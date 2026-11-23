"""Today's controller, one registered implementation per stage. Bodies are in `pickup.py` for now."""

from __future__ import annotations

from stages import register



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
