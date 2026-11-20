import dataclasses
from typing import TYPE_CHECKING

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
import openpi.models.gemma as _gemma
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils

if TYPE_CHECKING:
    from openpi.models.pi0 import Pi0


@dataclasses.dataclass(frozen=True)
class Pi0Config(_model.BaseModelConfig):
    dtype: str = "bfloat16"
    paligemma_variant: _gemma.Variant = "gemma_2b"
    action_expert_variant: _gemma.Variant = "gemma_300m"

    # Set the model specific defaults.
    action_dim: int = 32
    action_horizon: int = 50
    max_token_len: int = None  # type: ignore
    # Pi05 has two differences from Pi0:
    # - the state input is part of the discrete language tokens rather than a continuous input that is part of the suffix
    # - the action expert uses adaRMSNorm to inject the flow matching timestep
    pi05: bool = False
    # This config option is not used directly by the model, but it is read by the ModelTransformFactory.
    discrete_state_input: bool = None  # type: ignore

    pytorch_compile_mode: str | None = "max-autotune"

    # Tactile token-level concatenation (OpenVTLA-style). If tactile_tokens > 0, the observation carries
    # `tactile` features of shape [tactile_tokens, tactile_feature_dim] that are projected to the VLM width and
    # prepended to the visual token sequence.
    tactile_tokens: int = 0
    tactile_feature_dim: int = 768
    # "linear" or "mlp" (Linear-GELU-Linear, LLaVA-style).
    tactile_projector: str = "mlp"

    # --- Tactile contrastive (margin) loss (option A: distance-adaptive margin) ------------------------------------
    # For the same vision + ground-truth action (shared noise/time), the model runs once with the real tactile and
    # once with a "wrong" tactile negative. We require the real-tactile flow loss to be below the wrong-tactile loss
    # by an adaptive margin that scales with how different the negative actually is (so a coincidentally-similar
    # negative -> margin ~ 0, avoiding false-negative pressure). Disabled when weight == 0 or tactile_tokens == 0.
    tactile_margin_weight: float = 0.0  # lambda; 0 disables the whole term (extra LLM forward is skipped)
    tactile_margin_base: float = 0.1  # base margin, reached when the negative is maximally different (d >= d_ref)
    tactile_neg_mode: str = "batch_far"  # "batch_far" (in-batch farthest) | "batch_roll" (random different sample)
    tactile_dref_pct: float = 50.0  # percentile of in-batch tactile distances used as the adaptive d_ref

    # --- Vision feature regularization (encourage the policy to rely on touch) -------------------------------------
    # Applied to post-SigLIP image tokens, train only. noise adds Gaussian jitter; mask randomly drops image tokens
    # from attention. Inference (sample_actions, train=False) is untouched.
    vision_feature_noise_std: float = 0.0
    vision_feature_mask_prob: float = 0.0

    # --- Tactile <-> end-effector-view alignment (option 1) --------------------------------------------------------
    # Auxiliary cross-modal InfoNCE that pulls the pooled tactile embedding toward the pooled image tokens of the view
    # that frames the end-effector/contact (the wrist camera), and pushes it away from other samples' wrist views and
    # from the head/context view. Uses clean (pre-noise) image tokens, so it adds no extra forward pass. 0 disables.
    tactile_align_weight: float = 0.0
    tactile_align_temp: float = 0.1
    tactile_align_eef_view: str = "left_wrist_0_rgb"  # the wrist camera = where the eef is visible (positive)
    tactile_align_context_view: str = "base_0_rgb"  # head camera = same scene, non-eef region (hard negative)

    def __post_init__(self):
        if self.max_token_len is None:
            object.__setattr__(self, "max_token_len", 200 if self.pi05 else 48)
        if self.discrete_state_input is None:
            object.__setattr__(self, "discrete_state_input", self.pi05)
        if self.pytorch_compile_mode is not None:
            assert self.pytorch_compile_mode in [
                "default",
                "reduce-overhead",
                "max-autotune",
                "max-autotune-no-cudagraphs",
            ]

    @property
    @override
    def model_type(self) -> _model.ModelType:
        if self.pi05:
            return _model.ModelType.PI05
        return _model.ModelType.PI0

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0":
        from openpi.models.pi0 import Pi0

        return Pi0(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        with at.disable_typechecking():
            observation_spec = _model.Observation(
                images={
                    "base_0_rgb": image_spec,
                    "left_wrist_0_rgb": image_spec,
                    "right_wrist_0_rgb": image_spec,
                },
                image_masks={
                    "base_0_rgb": image_mask_spec,
                    "left_wrist_0_rgb": image_mask_spec,
                    "right_wrist_0_rgb": image_mask_spec,
                },
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
                tactile=(
                    jax.ShapeDtypeStruct([batch_size, self.tactile_tokens, self.tactile_feature_dim], jnp.float32)
                    if self.tactile_tokens > 0
                    else None
                ),
                tactile_mask=jax.ShapeDtypeStruct([batch_size], jnp.bool_) if self.tactile_tokens > 0 else None,
            )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)

        return observation_spec, action_spec

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """Returns the freeze filter based on the model config."""
        filters = []
        has_lora = False
        gemma_params_filter = nnx_utils.PathRegex(".*llm.*")
        action_expert_params_filter = nnx_utils.PathRegex(".*llm.*_1.*")
        if "lora" in self.paligemma_variant:
            filters.append(
                gemma_params_filter,
            )
            if "lora" not in self.action_expert_variant:
                # If only freeze gemma params, exclude action expert params.
                filters.append(
                    nnx.Not(action_expert_params_filter),
                )
            has_lora = True
        elif "lora" in self.action_expert_variant:
            filters.append(
                action_expert_params_filter,
            )
            has_lora = True

        if has_lora:
            # If any lora is used, exclude all lora params.
            filters.append(
                nnx.Not(nnx_utils.PathRegex(".*lora.*")),
            )
        if not filters:
            return nnx.Nothing
        return nnx.All(*filters)
