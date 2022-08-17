"""Define the LightRoutine abstract base class."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from time import sleep
from random import choice, normalvariate

from phue import Light

from ..color import Palette
from .data import (
    DEFAULT_BRIGHTNESS,
    DEFAULT_LIGHT_TRANSITION_TIME,
    DEFAULT_MEAN_RANDOM_CYCLE_TIME,
    DEFAULT_STDEV_RANDOM_CYCLE_TIME,
    DEFAULT_LOOP_CYCLE_TIME,
    DEFAULT_TIME_LIMIT,
    MINIMUM_SLEEP_TIME,
)


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
    transition_time: int = DEFAULT_LIGHT_TRANSITION_TIME  # deciseconds
    brightness: int = DEFAULT_BRIGHTNESS  # 0 - 254
    time_limit: int = DEFAULT_TIME_LIMIT  # seconds
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

    cycle_time: int = DEFAULT_LOOP_CYCLE_TIME  # seconds
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


@dataclass
class RandomWalk(LightRoutine):
    """

    Attributes
    ----------
    mean_cycle_time : int
        The mean time between color changes, in seconds.
    stdev_cycle_time : int
        The standard deviation of the time between color changes, in seconds.

    """

    mean_cycle_time: int = DEFAULT_MEAN_RANDOM_CYCLE_TIME
    stdev_cycle_time: int = DEFAULT_STDEV_RANDOM_CYCLE_TIME

    def loop(self) -> None:
        """Loop through all colors, randomly, at random intervals."""

        start = datetime.utcnow()

        while (datetime.utcnow() - start).seconds < self.time_limit:
            color = choice(self.palette)

            for light in self.lights:
                light.xy = color.xy

                if self.verbose:
                    # TODO: Logging
                    print(light.name, f"xy: {light.xy}")

            # normal variable X with mean: μ and stdev: σ
            sleep_time = normalvariate(self.mean_cycle_time, self.stdev_cycle_time)
            clamped_sleep_time = max([MINIMUM_SLEEP_TIME, sleep_time])

            sleep(clamped_sleep_time)
