"""Define Color and Palette."""

from __future__ import annotations
from dataclasses import dataclass
from typing import List

from ..data import CONVERTER


@dataclass
class Color:
    _r: int | None = None
    _g: int | None = None
    _b: int | None = None
    _x: float | None = None
    _y: float | None = None

    def __post_init__(self) -> None:
        rgb = [self._r, self._g, self._b]
        xy = [self._x, self._y]

        if all(v is not None for v in rgb):
            x, y = CONVERTER.rgb_to_xy(*rgb)
            self._x = x
            self._y = y
        elif all(v is not None for v in xy):
            r, g, b = CONVERTER.xy_to_rgb(*xy)
            self._r = r
            self._g = g
            self._b = b
        else:
            raise ValueError(f"Could not create Color from rgb: {rgb} & xy: {xy}")

    # properties

    @property
    def r(self) -> int:
        """Return the red channel of an RGB representation, ranging from 0-255.

        Returns
        -------
        int
        """
        assert self._r is not None
        return self._r

    @property
    def g(self) -> int:
        """Return the green channel of an RGB representation, ranging from 0-255.

        Returns
        -------
        int
        """
        assert self._g is not None
        return self._g

    @property
    def b(self) -> int:
        """Return the blue channel of an RGB representation, ranging from 0-255.

        Returns
        -------
        int
        """
        assert self._b is not None
        return self._b

    @property
    def x(self) -> float:
        """Return the x value of an XY representation, ranging from 0.0 - 1.0.

        Returns
        -------
        float
        """
        assert self._x is not None
        return self._x

    @property
    def y(self) -> float:
        """Return the y value of an XY representation, ranging from 0.0 - 1.0.

        Returns
        -------
        float
        """
        assert self._y is not None
        return self._y

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
