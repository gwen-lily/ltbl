"""Define the LightRoutine abstract base class."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing_extensions import Self
from datetime import datetime
from time import sleep

from phue import Light

from ..color import Palette


@dataclass
class LightRoutine(ABC):
    """Abstract light routine which defines basic validation and structure.

    Attributes
    ----------
    lights : list[Light]
        A list of lights relevant to the LightRoutine.

    palette : Palette
        A list of Color objects.

    transition_time : int
        The time in deciseconds that lights take to transition between colors.

    brightness : int
        The brightness of the lights, from 0 (off) to 254 (on).

    time_limit: int
        The time in seconds before the routine terminates.

    verbose : bool
        If True, log each light change to the terminal window.
    """

    lights: list[Light]
    palette: Palette
    transition_time: int  # deciseconds
    brightness: int  # 0 - 254
    time_limit: int  # seconds
    verbose: bool = True

    def __post_init__(self) -> None:
        for light in self.lights:
            light.transitiontime = self.transition_time
            light.brightness = self.brightness

    @abstractmethod
    def loop(self) -> None:
        """Define color transition behavior."""


@dataclass
class LoopLights(LightRoutine):
    """A light routine that loops.

    Attributes
    ----------
    cycle_time : int
        The time between color changes.
    """

    cycle_time: int = 5  # seconds
    _index: int = field(init=False, default=0)

    def loop(self) -> None:
        """Loop through all colors at a constant rate."""

        start = datetime.utcnow()
        n = len(self.palette)

        while (datetime.utcnow() - start).seconds < self.time_limit:
            color = self.palette[self._index % n]
            self._index += 1

            for light in self.lights:
                light.xy = color.xy

                if self.verbose:
                    # TODO: Logging
                    print(light.name, f"xy: {light.xy}")

            sleep(self.cycle_time)
