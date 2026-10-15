# Reproducibility

Every number, table and figure in this repository is produced by code from logged runs:
- the README and `docs/` quote tables generated from those runs;
- CI fails if they ever differ;
- a single command re-runs every experiment.

This page covers what that guarantees, what it takes, and how to check it.

## What is guaranteed

- **Same platform, same results.** With the same configuration, operating system, CPU
  architecture and package versions, a run reproduces its training log (`train_episodes.csv`),
  periodic evaluations (`eval.csv`) and final evaluation (`final_eval.json`) byte for byte.
  Timings in `meta.json` and the internal metadata of checkpoint files are not reproduced.
- **Every reported number comes from logged data.** `scripts/report.py` rebuilds every table
  and figure from the run directories, and regenerating them from unchanged runs gives
  byte-identical files, figures and animations included.
- **The documentation quotes the generated tables.** Each table in the README and `docs/` is
  copied from `results/tables/` between include markers by `scripts/sync_docs.py`. CI runs
  `scripts/sync_docs.py --check` and fails if a document is out of date.

**Across platforms**, floating-point results can differ in the last bits: Box2D's physics,
BLAS kernels and PyTorch's CPU kernels differ between operating systems and CPUs, and RL
amplifies tiny differences over a million steps. A different platform should reproduce the
results statistically (the same conclusions, within the reported intervals), not
bit for bit.

## How determinism is achieved

- **Every source of randomness is seeded** from the run's seed: Python, NumPy, PyTorch, the
  training environment and the action space. Each consumer gets its own derived stream
  (`lunarlander_rl.seeding.derive_seed`), so adding random draws in one place cannot shift
  another.
- **Deterministic PyTorch.** Training runs on the CPU with `torch.use_deterministic_algorithms(True)`
  and a single thread per run (parallelism comes from running many runs at once).
- **Evaluation never disturbs training.** Policies act through a side-effect-free `predict`;
  stochastic evaluation draws from a separate, per-episode-seeded generator. Tests check that
  changing the evaluation schedule leaves the training log unchanged.
- **Evaluation episodes are fixed.** Periodic evaluations use episode seeds 20000–20009 and the
  final evaluation 10000–10099, for every run and method. Tuning used separate seeds
  (training 0–2, validation episodes 30000–30099).
- **Provenance is recorded.** Every run directory holds:
  - the fully resolved `config.yaml`;
  - a `meta.json` with the git commit, whether the working tree had uncommitted changes,
    Python and package versions, platform, CPU, command and timings.

## Evidence

The guarantee has been checked repeatedly, always by byte comparison of the deterministic files:

| Check | Runs compared | Result |
| :--- | ---: | :--- |
| Same seed, same config, twice (test suite, every agent including SB3) | all agents | identical |
| Phase 3 replication runs re-run after the training loop was refactored | 3 | identical |
| A1 ablation's benchmark cell against the benchmark's REINFORCE runs | 10 | identical |
| A2a ablation's benchmark cell against the benchmark's DQN runs | 5 | identical |
| A2b ablation's reference cell against the replication's DQN config 1 runs | 5 | identical |
| Wind study's calm re-evaluation against each benchmark run's final evaluation | 50 | identical |
| v2.0.0 release code, `train.py --from-run ... --check`, seed 100 of every benchmark method | 5 | identical |

The benchmark and ablation runs were produced during development, so their `meta.json` files
record `git_dirty: true`. The last row answers whether the released code still reproduces them.
One run of each of the five methods (Q-learning, DQN, REINFORCE, SB3 DQN and SB3 PPO) was re-run
with the v2.0.0 code from its saved config. All five reproduced their 1M-step training logs and
evaluations byte for byte.

## The environment the results come from

| | |
| :--- | :--- |
| Operating system | Windows 10 (build 19045) |
| CPU | AMD Ryzen, Zen 3 (12 cores, 24 threads) |
| Python | 3.11.16 |
| PyTorch | 2.14.1 (CPU) |
| Gymnasium / Box2D | 1.3.0 / 2.3.10 |
| NumPy | 2.4.6 |
| Stable-Baselines3 | 2.9.0 |

The full set of package versions is pinned in [requirements-lock.txt](../requirements-lock.txt).
Every run's own `meta.json` records the versions it actually ran with.

## Running everything

```bash
pip install -e ".[dev,sb3,media]"
python scripts/reproduce_all.py --workers 20    # every experiment, report and figure, then the docs
python scripts/reproduce_all.py --quick         # a few-minute smoke test into runs_quick/, results_quick/
```

Experiments run in dependency order: the wind study and the animations reuse the main
benchmark's policies. Completed runs are skipped, so the command can be stopped and restarted.
Wall-clock times on the machine above:

| Experiment | Runs | Workers | Time |
| :--- | ---: | ---: | ---: |
| `replication_v1` (v1.0 configurations, 5 seeds) | 55 | 20 | ~1.5 h (REINFORCE runs, ~30 min each) |
| `baseline_check` (SB3 implementation check) | 9 | 9 | ~20 min |
| `tuning` (28 candidates × 3 seeds × 1M steps) | 84 | 20 | ~1.5 h |
| `main_benchmark` (5 methods × 10 seeds × 1M steps) | 50 | 20 | ~1.3 h |
| `ablation_reinforce` (A1) | 60 | 20 | ~20 min |
| `ablation_dqn` (A2a) | 20 | 20 | ~1.3 h |
| `ablation_dqn_v1` (A2b) | 20 | 4 | ~1 h |
| `wind` (A3: 250 re-evaluations) | n/a | 4 | ~6 min |
| `showcase` (animations) | n/a | 1 | ~1 min |

In total, roughly 8 hours with 20 parallel workers on a 12-core CPU. DQN dominates: a 1M-step
DQN run takes about an hour when 20 runs share the CPU. All results can be rebuilt from existing
runs in under a minute with `python scripts/report.py all`.

## Checking a single run, or one row of a table

Re-run any run from its saved configuration and compare the results byte for byte:

```bash
python scripts/train.py --from-run runs/main_benchmark/q_learning/seed_100 --check
```

The re-run goes to `runs_verify/`, and the command exits with an error if the training log or
either evaluation differs.

To recompute one row of a results table (all seeds of one method), run one variant of the
experiment into a fresh directory, then build the report from it:

```bash
python scripts/sweep.py configs/experiments/main_benchmark.yaml --variant q_learning --runs-root runs_check
```

Re-evaluating a saved policy, for example under conditions it never trained in, does not
retrain anything:

```bash
python scripts/evaluate.py runs/main_benchmark/dqn/seed_100 --set env.kwargs.enable_wind=true
```

## Using the released runs instead of re-running

The v2.0.0 GitHub release has two archives:
- every run's records (configs, provenance, training logs and evaluations);
- the main benchmark's trained policies.

Unzip them in the repository root to recreate `runs/`. Then:

```bash
python scripts/report.py all                    # every table and figure, without training
python scripts/sync_docs.py --check             # the docs match them
python scripts/evaluate.py runs/main_benchmark/sb3_ppo/seed_100   # re-evaluate a released policy
```

The archives are built by `scripts/package_release.py`, reproducibly (sorted files, fixed
timestamps). Their SHA-256 checksums are published with them.
