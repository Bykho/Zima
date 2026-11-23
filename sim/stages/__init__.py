"""Move stages: a registry of implementations and the pipeline a move runs.

`pickup.execute_move` runs one implementation per stage, in `ORDER`, on a
shared `Move`. Implementations register under a stage and a name:

    @register("carry", "plumb_then_legacy")
    class PlumbThenLegacy: ...

`resolve(cfg)` picks one per stage: `DEFAULTS`, then `cfg["pipeline"]`, then
$ZIMA_PIPELINE (`--stage stage=impl` on the CLIs, `run.py --variants`).
Parameters for an implementation live in `cfg["pipeline"]["params"]["<stage>.<name>"]`
or come as `;stage.key=value` in a spec.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os

from stages.types import Fail, Move, Stop

# Pipeline order. observe also confirms the place; retreat also unhooks before observe;
# recovery runs only when a stage fails (and is gated by `pickup.recover_fallen`).
# board_observe and unstack_policy are move-level, run by `episode`.
ORDER = ("observe", "grasp_plan", "approach", "close", "carry", "release", "retreat")
DEFAULTS = {
    "observe": "wrist_vision",
    "grasp_plan": "piece_table",
    "approach": "vertical_column",
    "close": "load_check",
    "carry": "plumb_then_legacy",
    "release": "open_at_place",
    "retreat": "pad_backoff",
    "recovery": "nest",
    "board_observe": "wrist_scan",
    "unstack_policy": "pawns_first",
}

_REGISTRY: dict[str, dict[str, type]] = {s: {} for s in DEFAULTS}


def register(stage: str, name: str):
    if stage not in _REGISTRY:
        raise ValueError(f"unknown stage {stage!r}; stages: {', '.join(_REGISTRY)}")

    def deco(cls):
        if name in _REGISTRY[stage]:
            raise ValueError(f"{stage}:{name} registered twice")
        cls.stage, cls.name = stage, name
        _REGISTRY[stage][name] = cls
        return cls

    return deco


def registered() -> dict[str, list[str]]:
    _load()
    return {s: sorted(impls) for s, impls in _REGISTRY.items()}


ENV = "ZIMA_PIPELINE"


def parse_spec(text: str) -> tuple[dict[str, str], dict[str, dict]]:
    """`stage:impl[,stage:impl][;stage.key=value]` -> (choices, params by "stage.impl").

    A param names its stage; it applies to whichever implementation that stage resolves to.
    """
    choices: dict[str, str] = {}
    by_stage: dict[str, dict] = {}
    head, *settings = [t.strip() for t in (text or "").split(";")]
    for tok in filter(None, (t.strip() for t in head.split(","))):
        stage, sep, name = tok.partition(":")
        if not sep or not name:
            raise ValueError(f"bad stage choice {tok!r}; want stage:impl")
        choices[stage.strip()] = name.strip()
    for tok in filter(None, settings):
        key, sep, value = tok.partition("=")
        stage, dot, field = key.strip().partition(".")
        if not sep or not dot:
            raise ValueError(f"bad stage param {tok!r}; want stage.key=value")
        try:
            val = json.loads(value)
        except json.JSONDecodeError:
            val = value.strip()
        by_stage.setdefault(stage, {})[field] = val
    return choices, by_stage


def resolve(cfg: dict | None, pipeline: str | None = None) -> dict[str, object]:
    """Stage -> implementation instance.

    Precedence: DEFAULTS < cfg `pipeline:` < $ZIMA_PIPELINE (set by `--stage`/`--variants`) < `pipeline`.
    Params: cfg `pipeline.params["<stage>.<impl>"]`, then `;stage.key=value` settings from the specs.
    """
    _load()
    chosen = dict(DEFAULTS)
    conf = dict((cfg or {}).get("pipeline") or {})
    params = {k: dict(v) for k, v in (conf.pop("params", None) or {}).items()}
    stage_params: dict[str, dict] = {}
    layers = [(conf, {})] + [parse_spec(t) for t in (os.environ.get(ENV, ""), pipeline or "")]
    for choices, by_stage in layers:
        for stage, name in choices.items():
            if stage not in _REGISTRY:
                raise ValueError(f"unknown stage {stage!r}; stages: {', '.join(_REGISTRY)}")
            chosen[stage] = name
        for stage, kv in by_stage.items():
            stage_params.setdefault(stage, {}).update(kv)
    out = {}
    for stage, name in chosen.items():
        impls = _REGISTRY[stage]
        if name not in impls:
            raise ValueError(f"unknown {stage} implementation {name!r}; registered: {', '.join(sorted(impls))}")
        impl = impls[name]()
        impl.params = {**params.get(f"{stage}.{name}", {}), **stage_params.get(stage, {})}
        out[stage] = impl
    return out


def cli_arg(ap) -> None:
    """`--stage STAGE=IMPL` (repeatable) on any CLI that runs moves."""
    ap.add_argument("--stage", action="append", default=[], metavar="STAGE=IMPL",
                    help="pick a stage implementation (repeatable; run.py --list shows them)")


def select(choices: list[str]) -> str:
    """`--stage carry=legacy_only` -> $ZIMA_PIPELINE `carry:legacy_only`, checked against the registry."""
    spec = ",".join(c.replace("=", ":", 1) for c in choices)
    if spec:
        resolve(None, spec)
        os.environ[ENV] = spec
    return spec


def describe(cfg: dict | None) -> dict:
    """What a row records: the implementation per stage and a hash of their params."""
    P = resolve(cfg)
    chosen = {s: type(impl).name for s, impl in P.items()}
    blob = json.dumps({s: impl.params for s, impl in P.items()}, sort_keys=True, default=str)
    return {"pipeline": chosen, "pipeline_params_sha": hashlib.sha256(blob.encode()).hexdigest()[:12]}


MODULES = ("observe", "grasp_plan", "approach", "close", "carry", "release", "retreat", "recovery", "unstack")


def _load() -> None:
    for mod in MODULES:
        importlib.import_module(f"stages.{mod}")


__all__ = ["DEFAULTS", "ENV", "ORDER", "Fail", "Move", "Stop", "cli_arg", "describe", "parse_spec", "select", "register", "registered", "resolve"]
