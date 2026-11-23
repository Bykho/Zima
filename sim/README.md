# Zima sim: running, scoring and comparing

MuJoCo model of the SO-101 arm with a wrist D405 over a chess board. The
controller (`pickup.py`, `episode.py`) moves pieces; everything here measures
it. Use `.venv/bin/python3`; on Linux leave `MUJOCO_GL` unset.

## Walls

`robot.open_robot(world, wall)` decides what the controller may know. The
world is physics only once a robot is open: scene edits raise, and a body or
joint written outside physics fails the next step.

| wall | controller sees |
|---|---|
| `real` (default everywhere) | camera, joint encoders and gripper load only; its pieces are a belief built from wrist looks |
| `perception` | same as `real` |
| `cheat` | the true piece poses as its belief, actuated motion |
| `off` | the world itself (no separation) |

Scores always come from the world (`evaluate_move.py`): a placed piece is at
rest on the board (within 3 mm of the surface, not touching the robot),
upright, and within 10 mm of the square centre (`ok`). `ok_loose` is the old
nearest-square rule, reported alongside. Moving any other piece fails the move.

## Running suites

```
python sim/run.py --list                                   # suites and walls
python sim/run.py --suite core --walls real,cheat --jobs 4 --tag mytag
python sim/run.py --suite core --walls real --only open_e2e4,opera_05 --tag probe
python sim/run.py --suite core --walls real --realism      # rough physics + noisy depth (rows: variant realism)
```

Each episode runs in its own process under a memory and time guard
(`ZIMA_MEM_CAP_GB` per worker, `ZIMA_TOTAL_MEM_CAP_GB` total). A run writes
`runs/<tag>/`:

- `rows.jsonl`: one row per case x wall (`ok`, `ok_loose`, `reason`, `stage`, `mover_final`, `disturbed`, phase times, ...);
- `meta.json`: commit, dirty diff hash, config hash, platform, MuJoCo/numpy versions, variant;
- `summary.txt`: totals and failures per suite x wall, plus each suite's own report.

Runs repeat exactly on one machine (the runner pins `PYTHONHASHSEED=0`). Mac
and Linux differ by a few flips; the committed baselines are from Linux with
`mujoco==4.0.1` (see `meta.json`).

Episode suites: `core` (opening + Opera, 52 moves), `opening`, `opera`,
`immortal`, `evergreen`, `games`, `all`, `grasp`, `matrix`, `starting`,
`offsets`, `perception_error`, `neighbors`, `transport`.

Kinematic suites (`layout`, `reach`) have no controller. They run once on
the pseudo-wall `none` whatever `--walls` says, and their rows carry
`metrics` (IK reach counts, open-gripper gaps) instead of a move verdict.

Suites measure an episode by subscribing to `diag` trace events (`phase` for
every `log_state` name, `scan_start`/`scan_end`, `look_start`/`look`) and
reading the world from the callback; the controller only emits.

## Stages and A/B variants

`execute_move` runs one move as a pipeline of stages, each a registered class
in `stages/` (`observe`, `grasp_plan`, `approach`, `close`, `carry`,
`release`, `retreat`, plus `recovery`, `board_observe` and `unstack_policy`).
`python sim/run.py --list` prints every stage's implementations, default
first. Selection, later overriding earlier: `stages.DEFAULTS`, the
`pipeline:` block in `config.yaml`, `ZIMA_PIPELINE`, then `--stage`.

```
python sim/run.py --suite core --walls cheat --stage carry=legacy_only --tag legacy
python sim/run.py --suite core --walls cheat --variants base,legacy=carry:legacy_only,plumb=carry:plumb_only --tag ab
python sim/compare.py runs/ab --by variant --ref base
```

A spec is `stage:impl[,stage:impl][;stage.key=value]`. In `--variants`, a
`label=` token starts a variant and a bare `stage:impl` token adds to the one
before; each variant is layered over `--stage`. Every case runs once per
variant with labelled rows; `--by variant` compares each
variant against `--ref` within the run. `meta.json` and each row record the
selected pipeline and a hash of its parameters. Baselines hold `base` rows only.

## Comparing and baselines

```
python sim/compare.py runs/mytag                 # against baselines/<suite>_<wall> (or baselines/<suite> for layout/reach)
python sim/compare.py runs/new runs/old          # two runs
python sim/compare.py runs/mytag --exact         # exit 1 on any ok/reason/stage/metric difference
python sim/compare.py runs/mytag --save-baseline # replace the baselines the run covers
```

The report gives totals per suite x wall, every flip (ok to fail and back),
same-verdict reason/stage changes, kinematic metric changes (counts exact,
`*_mm` within 0.5 mm), phase-time deltas, and a `run.py` command that reruns
the changed cases. Exit code 1 means a regression (an ok to fail flip or a worse
metric), 2 means the two sides ran on different `config.yaml`.

Save a baseline only for an accepted score change. Behaviour-preserving
changes must compare EXACT.

## Verify

```
python sim/verify.py --quick     # load, joints, IK
python sim/verify.py             # MuJoCo tests + pick-place on every wall
python sim/verify.py --studies   # plus each suite's quick subset vs its baseline
python sim/studies.py --suite core,layout   # just the quick subsets
```

`--studies` fails on a regression on each suite's gate wall (`cheat`, or
`none` for kinematic suites); `real` rows are compared and printed but do not
gate yet.

## Interactive

```
python sim/play.py e2e4 e7e5          # persistent board, typed moves (board | reset | quit)
python sim/game.py --game opera       # replay a famous game
python sim/pickup.py --headless       # one pick-and-place, d1 -> d4
```

All three default to `--wall real` and wrist vision; `--gt` uses the belief
pose without a look (truth only on `cheat`/`off`).

## Old entry points

`harness.py`, `bench.py`, `evaluate.py`, `grasp_bench.py`, `geom_sweep.py` and
`reach_map.py` are aliases for `run.py` with their old default suites and walls.
