"""Synthetic StructureTile fixtures for M4 animal-mechanics tests."""


def structure_tile(kind="COOP", animal=None, placed_day=0, yield_units=0,
                    fed_today=False, consecutive_unfed=0, cared_today=False,
                    fertilizer_available=False, pending_care_bonus=0):
    return {
        "kind": kind,
        "animal": animal,
        "placed_day": placed_day,
        "yield_units": yield_units,
        "fed_today": fed_today,
        "consecutive_unfed": consecutive_unfed,
        "cared_today": cared_today,
        "fertilizer_available": fertilizer_available,
        "pending_care_bonus": pending_care_bonus,
    }
