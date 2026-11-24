# OpenViTac client additions (simulator side)

Files to add to an OpenViTac checkout (https://github.com/FVL-Repo/OpenViTac, tested at `986a470`) to evaluate
tactile_base policies and to collect DAgger data. They run on the simulator machine (needs RT cores, e.g. RTX 3090),
not on the training server.

```bash
cd /path/to/OpenViTac
git apply /path/to/tactile_base/examples/openvitac/openvitac_client/openpi_tactile_history.patch
cp -r /path/to/tactile_base/examples/openvitac/openvitac_client/policy/* policy/
cp /path/to/tactile_base/examples/openvitac/openvitac_client/task_config/* task_config/
cp /path/to/tactile_base/examples/openvitac/openvitac_client/scripts/* scripts/
pip install websockets msgpack   # in the OpenViTac conda env (the OpenPI client protocol)
```

| File | What it is |
|---|---|
| `openpi_tactile_history.patch` | `policy/openpi/deploy_policy.py`: sends the tactile history + background that the AnyTouch2 models need (`tactile_history_frames`, `tactile_history_stride`), saves every episode's executed actions to `<eval dir>/trajectories/<seed>_<result>.npz` (`save_trajectory`, default on), and an `_on_executed_action` hook |
| `policy/openpi/abs_joint/deploy_openvtla_<task>.yml` | evaluation configs for `openvtla_<task>*` servers (temporal ensemble, 4-frame tactile history, stride 2; the dataset's task prompt) for insert_USB, gear_assembly, empty_can_selection, grasp_chip |
| `policy/openpi_dagger/` | DAgger collection: the policy runs; when the task plugin (`tasks/<task>.py`) detects its failure state from privileged sim state, a scripted expert takes over; every frame is recorded (`<eval dir>/dagger/<seed>_<result>.hdf5`). Plugins: `insert_USB` (jam at the slot opening -> lift, re-align, insert), `gear_assembly` (gear blocked above the shaft -> re-phase, re-insert, wiggle / release + re-grasp, turn), `empty_can_selection` (wrong weight judgement -> return the heavy can / re-grip the empty one, basket). Config: `deploy_<task>.yml` |
| `task_config/gelsight_slowreset.yml` | `gelsight.yml` with `reset_time_limit: 600` (gear_assembly / empty_can_selection resets take > 120 s on an RTX 3090) |
| `policy/replay_traj/` | re-executes saved trajectories (re-rendering, no policy server) |
| `scripts/run_force_seeds.py` | grasp_chip (force task) seed by seed with a solver-speed watchdog: a launch still in the slow UIPC state after 120 s (last 10 steps > 2 s/step) is killed and the same seed relaunched. Copy to `scripts/` of OpenViTac |

Evaluation (20 episodes; the step timeout must be off on an RTX 3090, contact-rich UIPC steps exceed the 5 s default):

```bash
# training machine or same machine: uv run scripts/serve_policy.py --port 8000 policy:checkpoint \
#     --policy.config openvtla_insert_USB --policy.dir checkpoints/openvtla_insert_USB/baseline/19999
OPENVITAC_LIVESTREAM=0 OMNI_KIT_ACCEPT_EULA=yes python scripts/eval_policy.py insert_USB task_config/gelsight.yml \
    policy/openpi/abs_joint/deploy_openvtla_insert_USB.yml --start_seed 10000 --total_num 20 \
    --eval_step_timeout_seconds 0 --headless
```

DAgger collection (use seeds disjoint from the evaluation seeds; serve the task's base model first):

```bash
OPENVITAC_LIVESTREAM=0 OMNI_KIT_ACCEPT_EULA=yes python scripts/eval_policy.py insert_USB task_config/gelsight.yml \
    policy/openpi_dagger/deploy_insert_USB.yml --start_seed 20000 --total_num 100 \
    --eval_step_timeout_seconds 0 --headless
OPENVITAC_LIVESTREAM=0 OMNI_KIT_ACCEPT_EULA=yes python scripts/eval_policy.py gear_assembly task_config/gelsight_slowreset.yml \
    policy/openpi_dagger/deploy_gear_assembly.yml --start_seed 20000 --total_num 50 \
    --eval_step_timeout_seconds 0 --headless
```

Then convert on the training side: `uv run examples/openvitac/convert_dagger.py --src <eval dir>/dagger
--repo-id openvitac/<task>_dagger_gelsight --event-name <jam|blocked>`.

Note: if an Isaac Sim run ends with an exception it can hang without exiting and keep its GPU memory; check
`nvidia-smi` before starting the next run.
