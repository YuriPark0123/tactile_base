# OpenViTac client additions (simulator side)

Files to add to an OpenViTac checkout (https://github.com/FVL-Repo/OpenViTac, tested at `986a470`) to evaluate
tactile_base policies and to collect DAgger data. They run on the simulator machine (needs RT cores, e.g. RTX 3090),
not on the training server.

```bash
cd /path/to/OpenViTac
git apply /path/to/tactile_base/examples/openvitac/openvitac_client/openpi_tactile_history.patch
cp -r /path/to/tactile_base/examples/openvitac/openvitac_client/policy/* policy/
pip install websockets msgpack   # in the OpenViTac conda env (the OpenPI client protocol)
```

| File | What it is |
|---|---|
| `openpi_tactile_history.patch` | `policy/openpi/deploy_policy.py`: sends the tactile history + background that the AnyTouch2 models need (`tactile_history_frames`, `tactile_history_stride`), saves every episode's executed actions to `<eval dir>/trajectories/<seed>_<result>.npz` (`save_trajectory`, default on), and an `_on_executed_action` hook |
| `policy/openpi/abs_joint/deploy_openvtla_insert_USB.yml` | evaluation config for `openvtla_insert_USB*` servers (temporal ensemble, 4-frame tactile history, stride 2) |
| `policy/openpi_dagger/` | DAgger collection: the policy runs; on a jam (privileged sim state) a scripted expert lifts, re-aligns and inserts; every frame is recorded (`<eval dir>/dagger/<seed>_<result>.hdf5`) |
| `policy/replay_traj/` | re-executes saved trajectories (re-rendering, no policy server) |

Evaluation (20 episodes; the step timeout must be off on an RTX 3090, contact-rich UIPC steps exceed the 5 s default):

```bash
# training machine or same machine: uv run scripts/serve_policy.py --port 8000 policy:checkpoint \
#     --policy.config openvtla_insert_USB --policy.dir checkpoints/openvtla_insert_USB/baseline/19999
OPENVITAC_LIVESTREAM=0 OMNI_KIT_ACCEPT_EULA=yes python scripts/eval_policy.py insert_USB task_config/gelsight.yml \
    policy/openpi/abs_joint/deploy_openvtla_insert_USB.yml --start_seed 10000 --total_num 20 \
    --eval_step_timeout_seconds 0 --headless
```

DAgger collection (use seeds disjoint from the evaluation seeds):

```bash
OPENVITAC_LIVESTREAM=0 OMNI_KIT_ACCEPT_EULA=yes python scripts/eval_policy.py insert_USB task_config/gelsight.yml \
    policy/openpi_dagger/deploy_insert_USB.yml --start_seed 20000 --total_num 100 \
    --eval_step_timeout_seconds 0 --headless
```

Note: if an Isaac Sim run ends with an exception it can hang without exiting and keep its GPU memory; check
`nvidia-smi` before starting the next run.
