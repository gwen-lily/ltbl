"""Utilities for the color sub-module."""

import io

from PIL.Image import Image
from colorthief import ColorThief

from .color import Palette, Color


def clamp_rgb(rgb: tuple[int, int, int]) -> tuple[int, int, int]:
    """Clamp RGB values from 0 to 255.

    Necessary because ColorThief threw some wacky ValueErrors.

    Parameters
    ----------
    rgb : tuple[int, int, int]
        An RGB tuple.

    Returns
    -------
    tuple[int, int, int]
    """

    return tuple(min([max([0, v]), 255]) for v in rgb)


def colorthief_get_palette(__image: Image, /, k: int, **kwargs) -> Palette:
    """Get a palette from a Pillow Image using ColorThief

    This workaround is necessary as ColorThief only works with file objects.

    Parameters
    ----------
    __image : Image
        A Pillow Image object.
    k : int
        The number of colors.

    Returns
    -------
    Palette
    """

    with io.BytesIO() as pseudo_file:
        __image.save(pseudo_file, "PNG")
        ct = ColorThief(pseudo_file)

    # get the raw palette, clamp to 0-255, then transform to Color object.
    raw_palette = ct.get_palette(k, **kwargs)
    clamped_palette = tuple(clamp_rgb(v) for v in raw_palette)
    palette = [Color(*color) for color in clamped_palette]

    return palette
