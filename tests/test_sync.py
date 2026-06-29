"""Closed-loop tests — no hardware, no user. Run: python tests/test_sync.py"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "sim"))

from real.calib import Calib, JointMap, calibrate, reanchor  # noqa: E402


class FakeReal:
    def __init__(self, home_ticks: dict[str, int], ticks_per_rad: dict[str, float]):
        self._home = dict(home_ticks)
        self._tpr = dict(ticks_per_rad)
        self.ticks = dict(home_ticks)
        self.limits = {n: (0, 4095) for n in home_ticks}

    def read_ticks(self, joints):
        return {n: self.ticks[n] for n in joints}

    def read_limits(self, joints):
        return {n: self.limits[n] for n in joints}

    def write_ticks(self, cmd):
        pass  # hardware moves; truth already updated in move_to hook

    def wait_settled(self, target, tol=30, timeout=4.0):
        return {n: self.ticks[n] for n in target}


class FakeSim:
    def __init__(self, home: dict[str, float]):
        self.home = dict(home)
        self.state = dict(home)

    def reset_pose(self, pose):
        self.state = dict(pose)

    def set_joints(self, pose):
        self.state = {**self.state, **pose}

    def step(self, _dt):
        pass

    def move_to(self, target, duration=1.0):
        self.state.update(target)
        return True

    def joint_states(self):
        return dict(self.state)


def ok(name, cond, detail=""):
    s = "PASS" if cond else "FAIL"
    print(f"  [{s}] {name}" + (f" — {detail}" if detail else ""))
    return cond


def test_joint_map():
    m = JointMap(sim_home=0.0, real_home=2000, ticks_per_rad=650.0, min_tick=0, max_tick=4095)
    assert ok("home anchor", m.to_ticks(0.0) == 2000)
    assert ok("+0.5 rad", m.to_ticks(0.5) == 2000 + int(round(325)))
    assert ok("clamp", m.to_ticks(100.0) == 4095)


def test_calibrate_roundtrip():
    home_sim = {n: 0.0 for n in ["shoulder_pan", "elbow_flex"]}
    home_sim["gripper"] = 0.8
    tpr = {"shoulder_pan": 640.0, "elbow_flex": -640.0}
    home_tick = {"shoulder_pan": 1800, "elbow_flex": 2200}

    real = FakeReal(home_tick, tpr)

    def sim_pos(sim_rad):
        real.ticks = {
            n: int(round(home_tick[n] + tpr[n] * (sim_rad[n] - home_sim[n])))
            for n in home_tick
        }

    sim = FakeSim(home_sim)
    orig_move = sim.move_to

    def move_to(target, duration=1.0):
        sim.state.update(target)
        sim_pos(sim.state)
        return True

    sim.move_to = move_to  # type: ignore

    cal = calibrate(sim, real, home_sim, ["shoulder_pan", "elbow_flex"], nudge=0.25)
    for n in ["shoulder_pan", "elbow_flex"]:
        got = cal.joints[n].ticks_per_rad
        assert ok(f"fit {n}", abs(got - tpr[n]) < 5, f"got {got:.1f} want {tpr[n]}")


def test_sim_hold():
    from arm import SimArm, load_config

    home = load_config()["poses"]["home"]
    sim = SimArm(gui=False)
    sim.reset_pose(home)
    for _ in range(120):
        sim.set_joints(home)
        sim.step(1 / 240)
    err = max(abs(sim.joint_states()[n] - home[n]) for n in home)
    sim.disconnect()
    assert ok("sim hold home", err < 0.02, f"max err {err:.4f} rad")


def test_reanchor():
    home = {"a": 0.0, "b": 0.0}
    cal = Calib(
        joints={
            "a": JointMap(0.0, 1000, 650.0, 0, 4095),
            "b": JointMap(0.0, 2000, -650.0, 0, 4095),
        }
    )
    present = {"a": 1150, "b": 1850}
    cal2 = reanchor(cal, present, home, ["a", "b"])
    ticks = cal2.sim_to_ticks(home)
    assert ok("reanchor home", ticks == present)
    assert ok("+0.5 rad a", cal2.sim_to_ticks({"a": 0.5, "b": 0.0})["a"] == 1150 + int(round(325)))


def test_run_key_handler():
    sys.path.insert(0, str(ROOT / "sim"))
    from arm import JOINT_NAMES, load_config
    from controls import apply_key_event, nudge

    home = load_config()["poses"]["home"]
    limits = {n: (-3.0, 3.0) for n in JOINT_NAMES}
    state = {"targets": dict(home), "selected": 0, "step": 0.05, "grip_open": True}

    # same call pattern as sync.py run_cmd
    state["targets"], state["selected"], state["step"], state["grip_open"], _ = apply_key_event(
        state["targets"],
        limits,
        home,
        selected=state["selected"],
        step=state["step"],
        key=ord("2"),
        gripper_open=state["grip_open"],
    )
    assert ok("select joint 2", state["selected"] == 1)

    n = JOINT_NAMES[state["selected"]]
    state["targets"] = nudge(state["targets"], limits, n, state["step"])
    assert ok("nudge after select", state["targets"][n] > home[n])


def test_sync_module():
    import importlib.util

    spec = importlib.util.spec_from_file_location("sync", ROOT / "sync.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert ok("run_cmd defined", hasattr(mod, "run_cmd") and callable(mod.run_cmd))
    assert ok("calibrate_cmd defined", hasattr(mod, "calibrate_cmd"))


def test_cli():
    py = ROOT / ".venv" / "bin" / "python3"
    r = subprocess.run([str(py), str(ROOT / "sync.py"), "status"], capture_output=True, text=True, timeout=10)
    assert ok("sync status", r.returncode == 0, r.stderr[:120] if r.returncode else "")


def test_watchdog():
    from real.safety import SafetyConfig, Watchdog

    cfg = SafetyConfig(max_delta_ticks=10, max_tracking_error=50, max_load=500, fault_streak=3)
    wd = Watchdog(cfg, {"j": (0, 1000)})
    wd.seed({"j": 100})
    out = wd.filter({"j": 200})
    assert ok("ramp", out["j"] == 110)
    out2 = wd.filter({"j": 200})
    assert ok("ramp step 2", out2["j"] == 120)
    wd3 = Watchdog(cfg, {"j": (0, 1000)})
    wd3.seed({"j": 990})
    assert ok("limit clamp", wd3.filter({"j": 9999})["j"] == 1000)
    wd.seed({"j": 100})
    wd.filter({"j": 100})
    assert ok("tracking fault", wd.check({"j": 100}, {"j": 200}, {"j": 0}) is not None)
    wd.note(wd.check({"j": 100}, {"j": 200}, {"j": 0}))
    wd.note(wd.check({"j": 100}, {"j": 200}, {"j": 0}))
    assert ok("estop streak", wd.note(wd.check({"j": 100}, {"j": 200}, {"j": 0})))
    wd2 = Watchdog(cfg, {"j": (0, 1000)})
    wd2.seed({"j": 100})
    wd2.filter({"j": 100})
    assert ok("load fault", wd2.check({"j": 100}, {"j": 100}, {"j": 600}) is not None)


def main():
    print("test_sync")
    fails = 0
    for fn in [test_joint_map, test_calibrate_roundtrip, test_reanchor, test_sim_hold, test_watchdog, test_run_key_handler, test_sync_module, test_cli]:
        try:
            fn()
        except AssertionError:
            fails += 1
        except Exception as e:
            fails += 1
            print(f"  [FAIL] {fn.__name__} — {e}")
    print(f"\n{'ALL PASS' if fails == 0 else f'{fails} FAILED'}")
    sys.exit(fails)


if __name__ == "__main__":
    main()
