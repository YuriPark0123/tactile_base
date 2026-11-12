"""Convert openpi's pi05_base JAX checkpoint to openpi's PyTorch format (CPU is enough).

  python scripts/convert_openpi_checkpoint.py --openpi <openpi repo> --src <pi05_base JAX dir> [--dst <out dir>]

openpi's examples/convert_jax_model_to_pytorch.py looks the model up through openpi.training.config, which imports
dataset packages that are not needed here; this calls its converter with the pi0.5 model config directly.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from worldtactile import paths  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--openpi", required=True, help="openpi repository root")
    ap.add_argument("--src", required=True, help="downloaded pi05_base checkpoint (JAX)")
    ap.add_argument("--dst", default=str(paths.PI05_BASE))
    args = ap.parse_args()
    import jax
    sys.path.insert(0, f"{args.openpi}/examples")
    from convert_jax_model_to_pytorch import convert_pi0_checkpoint
    from openpi.models.pi0_config import Pi0Config

    jax.config.update("jax_platforms", "cpu")
    convert_pi0_checkpoint(checkpoint_dir=args.src, precision="bfloat16", output_path=args.dst,
                           model_config=Pi0Config(pi05=True, action_horizon=50))


if __name__ == "__main__":
    main()
