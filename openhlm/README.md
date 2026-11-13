# OpenHLM-data benchmark

Real Unitree G1 teleoperation recordings (`OpenHLM/OpenHLM-data`, five LeRobot sets), scored as open-loop action
prediction on held-out episodes; `scripts/check_openhlm.py` wants 20 distinct episodes at 50 % success or better.
The write-up, the contract and how to run it: `docs/openhlm.md`.

```bash
.venv-vla/bin/pip install -r openhlm/requirements.txt     # once: pyarrow next to the CUDA torch
python -m openhlm.fetch                                    # login node: tables + head-camera videos into $HF_HOME, table and frame caches
FAKE=1 sbatch openhlm/run.sbatch                           # 8-minute dry run on generated data
sbatch openhlm/run.sbatch                                  # two A40s, 30 minutes: train, select on val, test once, check
cat runs/openhlm/<job id>/report.md; python scripts/check_openhlm.py logs/openhlm
```

| file | what |
|---|---|
| `spec.py` | the contract: sets, split rule (salted hash, 4 test episodes per set), history 1/3 s, horizon 1 s, error, hold reference, success = skill >= 0.5, the record format; standard library only |
| `data.py` | reads a set (v2.x / v3.0 tables), names, hold map, training statistics, chunk starts, the normalised error |
| `policy.py` | hold, ridge, the chunk MLP (`mlp`, `mlp-wide`; `mlp-cam` with the camera encoder), the ensemble, scoring, checkpoints |
| `act_policy.py` | ACT from the LeRobot stack as a candidate (needs `lerobot`) |
| `train.py` | one candidate on one set inside a time budget, best-on-validation checkpoint |
| `pipeline.py` | split, train every candidate on every set (one worker per GPU), select per set, test once, report, check |
| `fetch.py`, `frames.py` | download and describe the data, build the table cache; decode one camera into a frame cache (every 3rd frame, 96 px) |
| `fake_data.py` | a stand-in dataset in the LeRobot v2.1 layout for the tests and `FAKE=1` runs |
| `run.sbatch` | the Slurm job (also runs as `bash openhlm/run.sbatch`) |
