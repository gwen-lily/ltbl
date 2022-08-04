"""Define Color and Palette."""

from __future__ import annotations
from typing import List

from ..data import CONVERTER


class Color:
    """A color is an object with representative attributes for various systems.


    Attributes
    ----------

    r : int
        The red channel in an RGB representation.

    g : int
        The green channel in an RGB representation.

    b : int
        The blue channel in an RGB representation.

    x : float
        The X channel in an XY representation.

    y : float
        The Y channel in an XY representation.
    """

    def __init__(self, *args, **kwargs) -> None:
        """Define a color by providing variable arguments."""

        match args:
            case r, g, b:
                assert all(isinstance(v, int) for v in [r, g, b])
                x, y = CONVERTER.rgb_to_xy(r, g, b)

            case x, y:
                assert all(isinstance(v, float) for v in [x, y])
                r, g, b = CONVERTER.xy_to_rgb(x, y)

            case _:
                raise ValueError(f"Could not create Color from: {args}")

        self.r: int = r
        self.g: int = g
        self.b: int = b
        self.x: float = x
        self.y: float = y

    @property
    def rgb(self) -> tuple[int, int, int]:
        """Return the rgb representation of a color with a scale of 0-255.

        Returns
        -------
        tuple[int, int, int]
        """
        return self.r, self.g, self.b

    @property
    def rgb01(self) -> tuple[float, float, float]:
        """Return the rgb representation of a color with a scale of 0.0-1.0

        Returns
        -------
        tuple[float, float, float]
        """
        return tuple(channel / 255 for channel in self.rgb)

    @property
    def xy(self) -> tuple[float, float]:
        """Return the xy representation of a color with a scale of 0.0-1.0

        Returns
        -------
        tuple[float, float]
        """
        return self.x, self.y


Palette = List[Color]
