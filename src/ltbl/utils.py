"""utility functions for the ltbl module."""

from os import PathLike
from pathlib import Path

import PIL
from PIL import Image
from phue import Light, Bridge

from ltbl.data import LightsScheme

from .network import BRIDGE


def is_image_file(__file: PathLike, /) -> bool:
    """returns True if a provided filepath is an image.

    Parameters
    ----------
    filepath : PathLike
        a filepath object

    Returns
    -------
    bool
    """

    file = Path(__file)

    try:
        with Image.open(file):
            return True
    except PIL.UnidentifiedImageError:
        return False


def get_lights_list(__bridge: Bridge = BRIDGE, /) -> list[Light]:
    """Returns a type-validated list of Lights."""

    lights = __bridge.get_light_objects(mode="list")
    assert all(isinstance(li, Light) for li in lights)
    return lights


def get_lights_by_id(__bridge: Bridge = BRIDGE, /) -> dict[int, Light]:
    """Returns a type-validated dictionary of lights."""

    lights: dict[int, Light] = __bridge.get_light_objects(mode=LightsScheme.ID)

    try:
        assert isinstance(lights, dict)
    except AssertionError:
        print(lights, type(lights))

    return lights
