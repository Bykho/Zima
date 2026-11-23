"""Move stages: a registry of implementations and the pipeline a move runs.

`pickup.execute_move` runs one implementation per stage, in `ORDER`, on a
shared `Move`. Implementations register under a stage and a name:

    @register("carry", "plumb_then_legacy")
    class PlumbThenLegacy: ...

`resolve(cfg)` picks one per stage from `cfg["pipeline"]` over `DEFAULTS`;
parameters for an implementation live in `cfg["pipeline"]["params"]["<stage>.<name>"]`.
"""

from __future__ import annotations

import importlib

from stages.types import Fail, Move, Stop

# Methods the runner calls, per stage (observe also confirms; retreat also unhooks).
ORDER = ("observe", "grasp_plan", "approach", "close", "carry", "release", "retreat")
DEFAULTS = {
    "observe": "wrist_vision",
    "grasp_plan": "piece_table",
    "approach": "vertical_column",
    "close": "load_check",
    "carry": "plumb_then_legacy",
    "release": "open_at_place",
    "retreat": "pad_backoff",
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


def resolve(cfg: dict | None, pipeline: dict | None = None) -> dict[str, object]:
    """Stage -> implementation instance. Precedence: DEFAULTS < cfg `pipeline:` < `pipeline` argument."""
    _load()
    chosen = dict(DEFAULTS)
    conf = dict((cfg or {}).get("pipeline") or {})
    params = conf.pop("params", None) or {}
    for src in (conf, pipeline or {}):
        for stage, name in src.items():
            if stage not in _REGISTRY:
                raise ValueError(f"unknown stage {stage!r}; stages: {', '.join(_REGISTRY)}")
            chosen[stage] = name
    out = {}
    for stage, name in chosen.items():
        impls = _REGISTRY[stage]
        if name not in impls:
            raise ValueError(f"unknown {stage} implementation {name!r}; registered: {', '.join(sorted(impls))}")
        impl = impls[name]()
        impl.params = dict(params.get(f"{stage}.{name}") or {})
        out[stage] = impl
    return out


def _load() -> None:
    importlib.import_module("stages.defaults")


__all__ = ["DEFAULTS", "ORDER", "Fail", "Move", "Stop", "register", "registered", "resolve"]
