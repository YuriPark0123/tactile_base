"""Numerical checks of the WorldTactile pi0.5 model (needs preprocessed insert_HDMI and insert_hole data).

  python scripts/check_model.py [--device cuda]

1. Our joint-layer loop = openpi's PaliGemmaWithExpertModel.forward on a plain pi0.5 prefix/suffix with openpi's mask.
2. Inference path = training path: encode_static + dynamic_step give the same velocity and query states as one joint pass.
3. Masks: queries see the actions (bidirectional) and only their own finger's queries; images/prompt see neither
   tactile, queries nor actions; tactile does not see queries.
4. Gradients: stage 1 reaches only the new modules; stage 2 also LoRA, the action expert and the projections;
   SigLIP and the base VLM weights get none.
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks  # noqa: E402
from worldtactile import paths  # noqa: E402
from worldtactile.dataset import UniVTACContactDataset  # noqa: E402
from worldtactile.pi05.collate import Pi05Collator, to_device  # noqa: E402
from worldtactile.pi05.model import NEG, WorldTactilePi05, build_mask  # noqa: E402

res = []


def check(name, ok, detail=""):
    res.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}", flush=True)


def rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm().clamp(min=1e-12)).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev = torch.device(args.device)
    torch.manual_seed(0)
    m = WorldTactilePi05(str(paths.PI05_BASE), gradient_checkpointing=False)
    m.add_lora(8)
    with torch.no_grad():  # make LoRA non-zero so it is actually exercised
        for n, p in m.lm.named_parameters():
            if "lora_B" in n:
                p.normal_(std=1e-3)
    for p in m.new_module_params():
        p.data = p.data.float()
    m.to(dev).eval()
    ds = UniVTACContactDataset(["insert_HDMI", "insert_hole"], cameras=["head", "wrist"], tactile_corrupt=True)
    batch = to_device(Pi05Collator()([ds[0], ds[len(ds) - 1]]), dev)
    B, H = 2, m.horizon
    t = torch.tensor([0.3, 0.8], device=dev)
    noise = torch.randn_like(batch["actions"])
    x_t = t[:, None, None] * noise + (1 - t[:, None, None]) * batch["actions"]
    ac = torch.autocast(args.device, dtype=torch.bfloat16)

    with torch.no_grad(), ac:
        # 1. plain pi0.5 (no tactile / queries) through openpi vs our loop
        s_emb, s_valid, s_types = m.embed_static(batch["images"], batch["image_masks"], batch["tokens"],
                                                 batch["token_masks"], batch["tactile"])
        pre, pre_valid = s_emb[:, :-2], s_valid[:, :-2]
        a_emb, cond = m.embed_actions(x_t, t)
        pad = torch.cat([pre_valid, torch.ones(B, H, dtype=torch.bool, device=dev)], 1)
        att = torch.cat([torch.zeros(B, pre.shape[1], device=dev), torch.cat([torch.ones(B, 1, device=dev),
                         torch.zeros(B, H - 1, device=dev)], 1)], 1)
        mask2d = make_att_2d_masks(pad, att)
        pos = torch.cumsum(pad.long(), 1) - 1
        (_, a_ref), _ = m.pi.paligemma_with_expert.forward(
            attention_mask=torch.where(mask2d[:, None], 0.0, NEG), position_ids=pos, past_key_values=None,
            inputs_embeds=[pre, a_emb], use_cache=False, adarms_cond=[None, cond])
        (_, a_ours), _ = m._run([pre, a_emb], [None, cond], mask2d, pos)
        e = rel(a_ours, a_ref)
        check("1. joint layer loop = openpi PaliGemmaWithExpertModel.forward", e < 2e-2, f"(rel err {e:.2e}, bf16)")

        # 2. inference path vs training pass
        v_full, pred_full = m.full_pass(batch, batch["tactile"], x_t, t)
        cache = m.encode_static(batch)
        v_dyn, q_dyn = m.dynamic_step(cache, x_t, t)
        e_v, e_p = rel(v_dyn, v_full), rel(m.predict_tactile(q_dyn), pred_full)
        check("2. cached static + dynamic step = joint pass", e_v < 2e-2 and e_p < 2e-2,
              f"(velocity {e_v:.2e}, tactile prediction {e_p:.2e}, bf16)")

        # 3. masks
        v2, q2 = m.dynamic_step(cache, x_t + torch.randn_like(x_t), t)
        check("3a. queries see the actions (bidirectional)", rel(q2, q_dyn) > 1e-3, f"(query change {rel(q2, q_dyn):.2e})")
        types = torch.cat([s_types, torch.tensor([2] * 4 + [3] * 4 + [4] * H, device=dev).expand(B, -1)], 1)
        valid = torch.cat([s_valid, torch.ones(B, 8 + H, dtype=torch.bool, device=dev)], 1)
        mk = build_mask(types, valid)
        st_rows = (types == 0) & valid
        check("3b. images/prompt attend to nothing but images/prompt", not mk[st_rows][:, (types[0] != 0)].any().item())
        tac_rows = types[0] == 1
        check("3c. tactile does not attend to queries or actions", not mk[0][tac_rows][:, types[0] >= 2].any().item())
        ql, qr = types[0] == 2, types[0] == 3
        check("3d. left queries do not see right queries", not mk[0][ql][:, qr].any().item())

    # 4. gradients
    m.train()

    def grads(stage):
        m.set_stage(stage)
        m.zero_grad(set_to_none=True)
        with ac:
            loss, _ = m(batch, stage=stage, adhere_weight=0.1)
        loss.backward()
        g = {}
        for name, sel in {"new": lambda n: n.startswith(("queries", "finger_emb", "tactile_in", "align_proj")),
                          "lora": lambda n: "lora_" in n,
                          "expert": lambda n: "gemma_expert" in n,
                          "proj": lambda n: n.startswith(("pi.action_", "pi.time_mlp")),
                          "siglip": lambda n: "vision_tower" in n,
                          "vlm_base": lambda n: "language_model" in n and "lora_" not in n}.items():
            g[name] = sum(float(p.grad.abs().sum()) for n, p in m.named_parameters() if sel(n) and p.grad is not None)
        return g

    g1 = grads(1)
    print("   stage 1 grad |sum|:", {k: f"{v:.2e}" for k, v in g1.items()})
    check("4a. stage 1: only the new modules",
          g1["new"] > 0 and all(g1[k] == 0 for k in ("lora", "expert", "proj", "siglip", "vlm_base")))
    g2 = grads(2)
    print("   stage 2 grad |sum|:", {k: f"{v:.2e}" for k, v in g2.items()})
    check("4b. stage 2: new modules, LoRA, expert, projections; not SigLIP / base VLM",
          all(g2[k] > 0 for k in ("new", "lora", "expert", "proj")) and g2["siglip"] == 0 and g2["vlm_base"] == 0)
    print(f"{sum(res)}/{len(res)} checks passed")


if __name__ == "__main__":
    main()
