"""Today's controller, one registered implementation per stage. Bodies are in `pickup.py` for now."""

from __future__ import annotations

from stages import register


@register("observe", "wrist_vision")
class WristVision:
    """Reach an observe pose over the source and estimate the piece XY from the wrist depth image."""

    def locate(self, m):
        import pickup

        return pickup._stage_observe(m)

    def confirm(self, m):
        import pickup

        return pickup._stage_confirm(m)


@register("grasp_plan", "piece_table")
class PieceTable:
    """Grasp point, height and pregrasp from the piece table plus the fixed-pad shift; pitch the wrist vertical."""

    def plan(self, m):
        import pickup

        return pickup._stage_grasp_plan(m)


@register("approach", "vertical_column")
class VerticalColumn:
    """Collision-free vertical approach, pregrasp, re-look and re-verify, descend to the grasp."""

    def run(self, m):
        import pickup

        return pickup._stage_approach(m)


@register("close", "load_check")
class LoadCheck:
    """Close the jaws; the grasp counts from gripper load."""

    def close(self, m):
        import pickup

        return pickup._stage_close(m)


@register("carry", "plumb_then_legacy")
class PlumbThenLegacy:
    """Lift, transport and lower over the destination: plumb plan first, legacy lift/transport/lower otherwise."""

    def carry(self, m):
        import pickup

        return pickup._stage_carry(m)


@register("release", "open_at_place")
class OpenAtPlace:
    """Open to the release aperture at the place pose."""

    def release(self, m):
        import pickup

        return pickup._stage_release(m)


@register("retreat", "pad_backoff")
class PadBackoff:
    """Back the pads off the piece and rise clear; also frees the hand from the last move (unhook)."""

    def unhook(self, m):
        import pickup

        return pickup._stage_unhook(m)

    def back_off(self, m):
        import pickup

        return pickup._stage_retreat(m)
