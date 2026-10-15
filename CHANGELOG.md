# Changelog

## v2.0.0: reproducible research release

v2 rebuilds the v1.0 coursework project as a reproducible benchmark. The v1.0 code and results
are preserved under the tag `v1.0-course-project` and in [docs/legacy](docs/legacy).

### Fixed (v1.0 bugs, each covered by a regression test)

- **Q-learning:**
  - TD targets bootstrapped from terminal states.
  - Training never explored: the greedy policy was always used.
  - The time limit was ignored.
  - The module failed to import.
- **DQN:**
  - Time-limit truncations were stored as terminal transitions.
  - A debug print ran on every update.
  - The target network synced every 10 episodes regardless of their length.
- **REINFORCE:** the constructor ignored its learning-rate and network-size arguments.
- **All agents:** an existing checkpoint silently replaced training, and nothing was seeded.

### Added

- **Package.** A Python package (`lunarlander_rl`) with typed, validated YAML configs and a
  shared agent interface.
- **Training and evaluation.** One training loop for every agent, deterministic seeding, and
  run directories that record config, provenance, training log and evaluations.
- **Evaluation protocol.** Periodic evaluation on fixed seeds, and final evaluation on 100
  held-out test episodes with an outcome breakdown (landed, crashed, out of bounds, timed out).
- **Reference baselines.** Stable-Baselines3 DQN and PPO run through the same pipeline.
- **Studies:**
  - a five-seed replication of all nine v1.0 configurations ([docs/replication.md](docs/replication.md));
  - tuning at the full budget;
  - a 10-seed, 1M-step benchmark of five methods ([docs/benchmark.md](docs/benchmark.md));
  - ablations of REINFORCE's baseline, Double DQN, the TD loss and the discount factor;
  - a wind-robustness study ([docs/ablations.md](docs/ablations.md)).
- **Statistics.** IQM, optimality gap, performance profiles and probability of improvement,
  each with stratified-bootstrap intervals.
- **Figures and media.**
  - Publication-quality figures in one house style.
  - Animated demonstrations chosen by fixed rules ([docs/demos.md](docs/demos.md)).
- **Reproduction tooling:**
  - `scripts/reproduce_all.py` regenerates every number, table and figure, and has a quick
    smoke-test mode;
  - `scripts/train.py --from-run ... --check` re-runs any run and verifies the results byte
    for byte;
  - documentation tables are generated and checked in CI.
- **Engineering.** Tests (unit, integration and slow learning tests), Ruff, mypy, pre-commit
  and GitHub Actions.
- **Documents.** A technical report ([docs/report.md](docs/report.md)), a reproducibility
  guide ([docs/reproducibility.md](docs/reproducibility.md)), `CITATION.cff` and an MIT licence.

### Changed

- **Q-learning's tile coding** now uses asymmetric offsets, bounds fitted to the states the
  lander visits, a configurable resolution and a larger hash table.
- **Methods are compared at an equal number of environment steps** instead of v1.0's
  per-method episode budgets.
- **The README** now reports v2 results. Several v1.0 conclusions did not survive replication.

## v1.0: course project

Q-learning, DQN and REINFORCE on LunarLander-v3, three configurations each, one unseeded run
per configuration. Tagged `v1.0-course-project`.
