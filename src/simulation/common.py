"""Module containing common utility functions for the simulation."""


def convert_minutes_to_hhmm(minutes: int, colon: bool = True) -> str:
    """Convert minutes since midnight to HHMM time format."""
    hh = (minutes // 60) % 24
    mm = minutes % 60
    if colon:
        return f"{hh:02d}:{mm:02d}"
    else:
        return f"{hh:02d}{mm:02d}"
