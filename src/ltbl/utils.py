"""utility functions for the ltbl module."""

from os import PathLike
from pathlib import Path

import PIL
from PIL import Image


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
