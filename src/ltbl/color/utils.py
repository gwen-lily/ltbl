"""Utilities for the color sub-module."""

import io

import numpy as np
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


def distance_from_grey(__color: Color, /) -> float:
    """Return the distance between a color and the grey line in RGB space, normalized from 0 to 1.

    Parameters
    ----------
    __color : Color
        A color.

    Returns
    -------
    float
    """

    p0 = __color.rgb01

    x1 = np.asarray([0, 0, 0])
    x2 = np.asarray([1, 1, 1])
    x0 = np.asarray(p0)
    max_distance = np.sqrt(2 / 3)  # distance from grey line to any corner of R, G, B

    # calculate distance using classic formula and normalize to [0, 1]
    cross_vector = np.cross(x0 - x1, x0 - x2)
    distance = np.dot(cross_vector, cross_vector) ** (1 / 2) / np.sqrt(3)
    normalized_distance = distance / max_distance
    return normalized_distance


def distance_between_colors(__wumbo: Color, __scrumbo: Color, /) -> float:
    """Return the distance between two colors in RGB-256 space.

    Parameters
    ----------
    __wumbo : Color
        A color, we call it wumbo.
    __scrumbo : Color
        Another color, we call it scrumbo.

    Returns
    -------
    float
    """

    rmean = 0.5 * (__wumbo.r + __scrumbo.r)
    _dr, dg, db = np.asarray(__wumbo.rgb) - np.asarray(__scrumbo.rbg)
    color_delta = np.sqrt(
        (2 + rmean / 256) * rmean**2 + 4 * dg**2 + (2 + (255 - rmean) / 256) * db**2
    )
    return color_delta
