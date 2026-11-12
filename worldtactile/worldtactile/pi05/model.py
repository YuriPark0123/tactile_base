"""WorldTactile on pi0.5 (openpi PyTorch).

PaliGemma (Gemma 2B) is the VL expert, openpi's Gemma-300M flow-matching action expert reads it through openpi's
layer-wise joint attention (per-expert QKV, one attention over the concatenated tokens). Added on the VLM side:
  current tactile  AnyTouch2 CLS (clip t-12..t) per finger -> Linear + finger embedding  (2 tokens)
  tactile queries  learnable, 4 per finger                                                  (8 tokens)
  tactile head     shared MLP on each finger's 4 query states (final norm) -> future AnyTouch2 CLS (512)

Token layout:  VLM stream [img head 256 | img wrist 256 | img 3rd slot (masked) | prompt 200 | tactile 2 | Q_L 4 | Q_R 4]
               expert     [actions 50]
Attention (row attends column):
  images + prompt   images + prompt (pi0.5's bidirectional prefix)
  tactile           images + prompt + tactile
  Q_L / Q_R         images + prompt + tactile + own finger's queries + ACTIONS   (bidirectional with the actions)
  actions           everything
Images, prompt and tactile never see queries or actions, so at inference their per-layer K,V are computed once
(`encode_static`) and every flow step only reruns queries (VLM weights) + actions (expert weights) (`dynamic_step`).
Training runs everything in one pass through the same `_layer` function.

Actions: robot actions (9: 7 arm + 2 fingers) normalized by the dataset (delta arm joints, quantile to [-1, 1]),
zero-padded to pi0.5's 32 dims; the flow loss uses the 9 real dims.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from transformers.models.gemma import modeling_gemma

from openpi.models.pi0_config import Pi0Config
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch, create_sinusoidal_pos_embedding

STATIC, TACTILE, QUERY_L, QUERY_R, ACTION = 0, 1, 2, 3, 4
NEG = -2.3819763e38  # openpi's masked logit
# key groups for the attention probe (inference): which modality each key token belongs to
ATTN_GROUPS = ("head_img", "wrist_img", "language", "state", "tactile_L", "tactile_R", "query_L", "query_R", "actions")


def build_mask(types, valid):
    """Boolean (B, N, N) attention mask from token types (B, N) and validity (B, N); see module docstring."""
    qi, kj = types[:, :, None], types[:, None, :]
    static_k = kj == STATIC
    m = (qi == STATIC) & static_k
    m |= (qi == TACTILE) & (static_k | (kj == TACTILE))
    for q in (QUERY_L, QUERY_R):
        m |= (qi == q) & (static_k | (kj == TACTILE) | (kj == q) | (kj == ACTION))
    m |= qi == ACTION
    m &= valid[:, None, :]
    return m


class WorldTactilePi05(nn.Module):
    def __init__(self, ckpt_dir=None, horizon=50, robot_action_dim=9, queries_per_finger=4, tactile_dim=512,
                 gradient_checkpointing=True):
        super().__init__()
        self.cfg = Pi0Config(pi05=True, action_horizon=horizon, dtype="bfloat16", pytorch_compile_mode=None)
        self.pi = PI0Pytorch(self.cfg)
        if ckpt_dir is not None:
            from safetensors.torch import load_file
            sd = load_file(f"{ckpt_dir}/model.safetensors")
            missing, unexpected = self.pi.load_state_dict(sd, strict=False)
            assert not unexpected, unexpected[:5]
            assert not [k for k in missing if "embed_tokens" not in k and "lm_head" not in k], missing[:5]
        pwe = self.pi.paligemma_with_expert
        self.lm, self.expert = pwe.paligemma.language_model, pwe.gemma_expert.model
        self.width = self.lm.config.hidden_size
        self.horizon, self.robot_dim, self.qpf = horizon, robot_action_dim, queries_per_finger
        self.gradient_checkpointing = gradient_checkpointing
        self.attn_probe = None  # set to {} to record per-layer attention mass by key group in dynamic_step
        self._probe_ctx = None

        # new modules (fp32), scaled to the RMS of PaliGemma's scaled input embeddings
        with torch.no_grad():
            emb = self.lm.embed_tokens.weight
            rows = emb[torch.randint(0, emb.shape[0], (4096,))].float()
            rms = float((rows * math.sqrt(self.width)).pow(2).mean().sqrt())
        self.emb_rms = rms
        self.tactile_in = nn.Linear(tactile_dim, self.width)
        nn.init.normal_(self.tactile_in.weight, std=rms / math.sqrt(tactile_dim))
        nn.init.zeros_(self.tactile_in.bias)
        self.finger_emb = nn.Parameter(torch.randn(2, self.width) * rms * 0.1)
        self.queries = nn.Parameter(torch.randn(2 * queries_per_finger, self.width) * rms)
        self.align_proj = nn.Sequential(nn.Linear(queries_per_finger * self.width, self.width), nn.GELU(),
                                        nn.Linear(self.width, tactile_dim))

    # ------------------------------------------------------------------ embeddings
    def embed_static(self, images, image_masks, tokens, token_masks, tactile):
        """images: list of (B, 3, 224, 224) in [-1, 1]; returns embeddings, validity and types of
        [images | prompt | tactile]."""
        embs, valid = [], []
        for img, m in zip(images, image_masks):
            e = self.pi.paligemma_with_expert.embed_image(img)  # float32 in [-1, 1]; patch embedding kept fp32
            embs.append(e)
            valid.append(m[:, None].expand(e.shape[0], e.shape[1]))
        lang = self.pi.paligemma_with_expert.embed_language_tokens(tokens) * math.sqrt(self.width)
        embs.append(lang)
        valid.append(token_masks)
        tac = self.tactile_in(tactile.float()) + self.finger_emb  # (B, 2, D)
        embs.append(tac.to(lang.dtype))
        B = tokens.shape[0]
        valid.append(torch.ones(B, 2, dtype=torch.bool, device=tokens.device))
        embs, valid = torch.cat(embs, 1), torch.cat(valid, 1)
        types = torch.full(valid.shape, STATIC, dtype=torch.long, device=valid.device)
        types[:, -2:] = TACTILE
        return embs.to(torch.bfloat16), valid, types

    def query_embs(self, B):
        q = self.queries.unsqueeze(0).expand(B, -1, -1).to(torch.bfloat16)
        types = torch.tensor([QUERY_L] * self.qpf + [QUERY_R] * self.qpf, device=q.device).expand(B, -1)
        return q, types

    def embed_actions(self, x_t, t):
        """x_t (B, H, robot_dim) normalized noisy actions -> expert inputs (B, H, D_e) and the adaRMS time cond."""
        pad = torch.zeros(*x_t.shape[:2], self.cfg.action_dim - x_t.shape[-1], device=x_t.device, dtype=x_t.dtype)
        a = self.pi.action_in_proj(torch.cat([x_t, pad], -1).to(self.pi.action_in_proj.weight.dtype))
        te = create_sinusoidal_pos_embedding(t, self.pi.action_in_proj.out_features, 4e-3, 4.0, device=t.device)
        te = te.to(self.pi.time_mlp_in.weight.dtype)
        cond = F.silu(self.pi.time_mlp_out(F.silu(self.pi.time_mlp_in(te))))
        return a.to(torch.bfloat16), cond

    # ------------------------------------------------------------------ one joint layer
    def _layer(self, li, streams, conds, mask4d, cos, sin, kv_static=None):
        """openpi's compute_layer_complete for a list of streams (None = absent), with optional cached K,V of earlier
        tokens prepended. Returns new hidden states per stream and this call's post-RoPE (k, v)."""
        models = [self.lm, self.expert]
        qs, ks, vs, gates, lens = [], [], [], [], []
        for i, h in enumerate(streams):
            if h is None:
                gates.append(None)
                lens.append(0)
                continue
            layer = models[i].layers[li]
            x, gate = layer.input_layernorm(h, cond=conds[i])
            shp = (*x.shape[:-1], -1, layer.self_attn.head_dim)
            qs.append(layer.self_attn.q_proj(x).view(shp).transpose(1, 2))
            ks.append(layer.self_attn.k_proj(x).view(shp).transpose(1, 2))
            vs.append(layer.self_attn.v_proj(x).view(shp).transpose(1, 2))
            gates.append(gate)
            lens.append(h.shape[1])
        q, k, v = torch.cat(qs, 2), torch.cat(ks, 2), torch.cat(vs, 2)
        q, k = modeling_gemma.apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1)
        k_new, v_new = k, v
        if kv_static is not None:
            k, v = torch.cat([kv_static[0], k], 2), torch.cat([kv_static[1], v], 2)
        attn = self.lm.layers[li].self_attn
        out, w = modeling_gemma.eager_attention_forward(attn, q, k, v, mask4d, attn.scaling)
        if self._probe_ctx is not None and kv_static is not None:
            onehot, n_q = self._probe_ctx  # (B, K, G), number of query rows (the rest are action rows)
            with torch.autocast(device_type=w.device.type, enabled=False):  # fp32 sums
                mass = torch.einsum("bhrk,bkg->bhrg", w.detach().float(), onehot.float())
                self.attn_probe["cur"].append(torch.stack([mass[:, :, :n_q].mean((1, 2)), mass[:, :, n_q:].mean((1, 2))], 1))
        out = out.reshape(q.shape[0], -1, attn.config.num_attention_heads * attn.head_dim)
        new, start = [], 0
        for i, h in enumerate(streams):
            if h is None:
                new.append(None)
                continue
            layer = models[i].layers[li]
            o = out[:, start:start + lens[i]].to(layer.self_attn.o_proj.weight.dtype)
            start += lens[i]
            o = modeling_gemma._gated_residual(h, layer.self_attn.o_proj(o), gates[i])  # noqa: SLF001
            res = o
            o, gate = layer.post_attention_layernorm(o, cond=conds[i])
            if layer.mlp.up_proj.weight.dtype == torch.bfloat16:
                o = o.to(torch.bfloat16)
            new.append(modeling_gemma._gated_residual(res, layer.mlp(o), gate))  # noqa: SLF001
        return new, (k_new, v_new)

    def _rope(self, h, pos):
        return self.lm.rotary_emb(h, pos)

    def _run(self, streams, conds, mask, pos, kv_static=None, keep_kv=False):
        """All layers. mask: bool (B, Nq, Nk). Returns final-normed streams and (optionally) per-layer K,V."""
        mask4d = torch.where(mask[:, None], 0.0, NEG).to(torch.float32)
        ref = next(h for h in streams if h is not None)
        cos, sin = self._rope(ref, pos)
        kvs = []
        for li in range(len(self.lm.layers)):
            kv_li = kv_static[li] if kv_static is not None else None
            if self.gradient_checkpointing and self.training:
                streams, kv = checkpoint(self._layer, li, streams, conds, mask4d, cos, sin, kv_li, use_reentrant=False)
            else:
                streams, kv = self._layer(li, streams, conds, mask4d, cos, sin, kv_li)
            if keep_kv:
                kvs.append(kv)
        models = [self.lm, self.expert]
        outs = [None if h is None else models[i].norm(h, cond=conds[i])[0] for i, h in enumerate(streams)]
        return outs, kvs

    # ------------------------------------------------------------------ heads
    def predict_tactile(self, q_out):
        """q_out (B, 2*qpf, D) final query states -> (B, 2, 512)."""
        B = q_out.shape[0]
        h = q_out.float().view(B, 2, self.qpf * self.width)
        return self.align_proj(h)

    def velocity(self, a_out):
        return self.pi.action_out_proj(a_out.to(self.pi.action_out_proj.weight.dtype)).float()[..., :self.robot_dim]

    # ------------------------------------------------------------------ training pass
    def full_pass(self, batch, tactile, x_t, t):
        """One joint pass over [static | tactile | queries] (VLM) and [actions] (expert).
        Returns flow velocity (B, H, robot_dim) and the tactile prediction (B, 2, 512)."""
        s_emb, s_valid, s_types = self.embed_static(batch["images"], batch["image_masks"], batch["tokens"],
                                                    batch["token_masks"], tactile)
        B = s_emb.shape[0]
        q_emb, q_types = self.query_embs(B)
        a_emb, cond = self.embed_actions(x_t, t)
        H = a_emb.shape[1]
        vlm = torch.cat([s_emb, q_emb], 1)
        valid = torch.cat([s_valid, torch.ones(B, q_emb.shape[1] + H, dtype=torch.bool, device=s_valid.device)], 1)
        types = torch.cat([s_types, q_types, torch.full((B, H), ACTION, device=s_types.device)], 1)
        mask = build_mask(types, valid)
        pos = torch.cumsum(valid.long(), 1) - 1
        (v_out, a_out), _ = self._run([vlm, a_emb], [None, cond], mask, pos)
        q_out = v_out[:, s_emb.shape[1]:]
        return self.velocity(a_out), self.predict_tactile(q_out)

    def flow_loss(self, v, target_v, action_mask):
        am = action_mask[..., None].float()
        return (((v - target_v) ** 2) * am).sum() / (am.sum() * v.shape[-1]).clamp(min=1)

    def forward(self, batch, stage=2, align_weight=1.0, adhere_weight=0.1, adhere_margin=5.0):
        actions = batch["actions"]  # (B, H, robot_dim), normalized
        B = actions.shape[0]
        t = torch.distributions.Beta(1.5, 1.0).sample((B,)).to(actions.device) * 0.999 + 0.001
        noise = torch.randn_like(actions)
        x_t = t[:, None, None] * noise + (1 - t[:, None, None]) * actions
        target_v = noise - actions
        v, pred = self.full_pass(batch, batch["tactile"], x_t, t)
        tm = batch["target_mask"].float()[:, None, None]
        align = (((pred - batch["tactile_target"].float()) ** 2) * tm).sum() / (tm.sum() * pred.shape[1] * pred.shape[2]).clamp(min=1)
        flow = self.flow_loss(v, target_v, batch["action_mask"])
        logs = dict(flow=flow.detach(), align=align.detach())
        if stage == 1:
            loss = align_weight * align
        else:
            loss = flow + align_weight * align
            if adhere_weight > 0 and "tactile_corrupt" in batch:
                v_bad, _ = self.full_pass(batch, batch["tactile_corrupt"], x_t, t)
                flow_bad = self.flow_loss(v_bad, target_v, batch["action_mask"])
                adhere = F.relu(adhere_margin - (flow_bad - flow))
                loss = loss + adhere_weight * adhere
                logs.update(flow_tac_corrupt=flow_bad.detach(), tac_gap=(flow_bad - flow).detach(), adhere=adhere.detach())
        logs["loss"] = loss.detach()
        return loss, logs

    # ------------------------------------------------------------------ inference
    @torch.no_grad()
    def encode_static(self, batch):
        """Per-layer K,V of [images | prompt | tactile] (they never attend to queries or actions)."""
        s_emb, s_valid, s_types = self.embed_static(batch["images"], batch["image_masks"], batch["tokens"],
                                                    batch["token_masks"], batch["tactile"])
        mask = build_mask(s_types, s_valid)
        pos = torch.cumsum(s_valid.long(), 1) - 1
        _, kvs = self._run([s_emb, None], [None, None], mask, pos, keep_kv=True)
        # key groups of the static tokens: 3 image slots of 256, prompt (language / state digits), tactile L, R
        n_img = len(batch["images"])
        per_img = (s_valid.shape[1] - batch["tokens"].shape[1] - 2) // n_img
        g = torch.full_like(s_types, -1)
        for slot, gid in zip(range(n_img), (0, 1, -1)):
            g[:, slot * per_img:(slot + 1) * per_img] = gid
        p0 = n_img * per_img
        tok_valid, is_state = batch["token_masks"], batch.get("state_token_mask", torch.zeros_like(batch["token_masks"]))
        g[:, p0:p0 + tok_valid.shape[1]] = torch.where(tok_valid, torch.where(is_state, 3, 2), -1)
        g[:, -2], g[:, -1] = 4, 5
        g = torch.where(s_valid, g, torch.full_like(g, -1))
        return dict(kv=kvs, valid=s_valid, types=s_types, next_pos=pos[:, -1] + 1, groups=g)

    def dynamic_step(self, cache, x_t, t):
        """Queries (VLM weights) + actions (expert) over the cached static K,V. Returns velocity and query states."""
        B = x_t.shape[0]
        q_emb, q_types = self.query_embs(B)
        a_emb, cond = self.embed_actions(x_t, t)
        H, Q = a_emb.shape[1], q_emb.shape[1]
        dyn_types = torch.cat([q_types, torch.full((B, H), ACTION, device=x_t.device)], 1)
        all_types = torch.cat([cache["types"], dyn_types], 1)
        all_valid = torch.cat([cache["valid"], torch.ones(B, Q + H, dtype=torch.bool, device=x_t.device)], 1)
        mask = build_mask(all_types, all_valid)[:, -(Q + H):]
        pos = cache["next_pos"][:, None] + torch.arange(Q + H, device=x_t.device)[None]
        if self.attn_probe is not None:
            dyn_g = torch.tensor([6] * self.qpf + [7] * self.qpf + [8] * H, device=x_t.device).expand(B, -1)
            groups = torch.cat([cache["groups"], dyn_g], 1)
            onehot = F.one_hot(groups.clamp(min=0), len(ATTN_GROUPS)).float() * (groups >= 0)[..., None]
            self._probe_ctx = (onehot, Q)
            self.attn_probe["share"] = (onehot.sum(1) / onehot.sum((1, 2))[:, None]).cpu()  # key-count share per group
            self.attn_probe["cur"] = []
        (q_out, a_out), _ = self._run([q_emb, a_emb], [None, cond], mask, pos, kv_static=cache["kv"])
        if self.attn_probe is not None:
            self.attn_probe.setdefault("steps", []).append(torch.stack(self.attn_probe.pop("cur"), 1).cpu())  # (B, L, 2, G)
            self._probe_ctx = None
        return self.velocity(a_out), q_out

    def sample_actions(self, cache, steps=10, noise=None, prev=None, weights=None, beta=20.0):
        """Euler flow integration (t: 1 -> 0), 10 steps by default. With `prev` (the previous chunk's unexecuted
        actions in this call's model space) and `weights` (rtc_weights), real-time chunking guidance (Black et al.,
        "Real-Time Execution of Action Chunking Flow Policies", 2025): each step adds -w * d<err, a_hat>/dx_t to the
        velocity, where a_hat = x_t - t * v is the predicted clean chunk and err the weighted gap to `prev`, with the
        guidance weight w clipped at `beta`. Returns normalized actions and the tactile prediction of the last step's
        query states."""
        B = cache["valid"].shape[0]
        dev = cache["valid"].device
        x = noise if noise is not None else torch.randn(B, self.horizon, self.robot_dim, device=dev)
        dt = 1.0 / steps
        q_out = None
        for i in range(steps):
            t_val = 1.0 - i * dt
            t = torch.full((B,), t_val, device=dev)
            if prev is None:
                with torch.no_grad():
                    v, q_out = self.dynamic_step(cache, x, t)
            else:
                with torch.enable_grad():
                    xg = x.detach().requires_grad_(True)
                    v, q_out = self.dynamic_step(cache, xg, t)
                    a_hat = xg - t_val * v
                    err = (prev - a_hat) * weights[None, :, None].to(dev)
                    g = torch.autograd.grad(a_hat, xg, grad_outputs=err)[0]
                tau = 1.0 - t_val
                r2 = (1 - tau) ** 2 / (tau ** 2 + (1 - tau) ** 2)
                w = beta if tau == 0 else min(beta, (1 - tau) / (tau * r2))
                v = v.detach() - w * g
            x = (x - dt * v).detach()
        with torch.no_grad():
            pred = self.predict_tactile(q_out.detach())
        return x, pred

    @staticmethod
    def rtc_weights(horizon, overlap, delay=0):
        i = torch.arange(horizon, dtype=torch.float32)
        c = (overlap - i) / (overlap - delay + 1)
        w = c * (torch.exp(c) - 1) / (math.e - 1)
        w = torch.where(i < delay, torch.ones_like(w), w)
        return torch.where(i >= overlap, torch.zeros_like(w), w)

    # ------------------------------------------------------------------ trainable parameter groups
    def add_lora(self, rank, alpha=None):
        from peft import LoraConfig, inject_adapter_in_model
        cfg = LoraConfig(r=rank, lora_alpha=alpha or 2 * rank, lora_dropout=0.0,
                         target_modules=r".*layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)")
        inject_adapter_in_model(cfg, self.lm)
        for n, p in self.lm.named_parameters():
            if "lora_" in n:
                p.data = p.data.float()

    def new_module_params(self):
        return [self.queries, self.finger_emb, *self.tactile_in.parameters(), *self.align_proj.parameters()]

    def set_stage(self, stage):
        """Stage 1: queries + tactile input + tactile head. Stage 2: + VLM LoRA + the whole action expert and the
        action/time projections. The SigLIP vision tower and the VLM base weights stay frozen."""
        for p in self.parameters():
            p.requires_grad_(False)
        for p in self.new_module_params():
            p.requires_grad_(True)
        if stage >= 2:
            for n, p in self.lm.named_parameters():
                if "lora_" in n:
                    p.requires_grad_(True)
            for p in self.expert.parameters():
                p.requires_grad_(True)
            for m in (self.pi.action_in_proj, self.pi.action_out_proj, self.pi.time_mlp_in, self.pi.time_mlp_out):
                for p in m.parameters():
                    p.requires_grad_(True)
        return [p for p in self.parameters() if p.requires_grad]
