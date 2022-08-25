"""Test the routine sub-module."""

from dataclasses import dataclass, field
from typing import Any
from unittest import TestCase, main

from phue import Light, Bridge

from ltbl import BRIDGE, Color
from ltbl.routine import LoopLights, RandomWalk
from ltbl.routine.data import LightState
from ltbl.utils import get_lights_by_id

Red = Color(255, 0, 0)
Green = Color(0, 255, 0)
Blue = Color(0, 0, 255)

RGB_Palette = [Red, Green, Blue]

_TEST_TIME_LIMIT = 10  # seconds

LIGHT_ID_DICT: dict[int, LightState] = {}


class TestRoutine(TestCase):
    """Test the routine sub-module."""

    def test_record_lights(self):
        """Test the test's ability to save the light state"""
        lights = BRIDGE.lights_by_id

        for light_id, light in lights.items():
            assert isinstance(light_id, int) and isinstance(light, Light)
            state = LightState(Color(*light.xy), light.brightness)
            LIGHT_ID_DICT[light_id] = state

    def test_LoopLights(self):
        routine = LoopLights(BRIDGE.lights, RGB_Palette, time_limit=_TEST_TIME_LIMIT)
        routine.loop()

    def test_RandomWalk(self):
        routine = RandomWalk(BRIDGE.lights, RGB_Palette, time_limit=_TEST_TIME_LIMIT)
        routine.loop()

    def test_reset_lights(self):
        """Test the test's ability to reset the lights"""
        for light_id, (color, brightness) in LIGHT_ID_DICT.items():
            light = BRIDGE[light_id]
            assert isinstance(light, Light)

            light.xy = color.xy
            light.brightness = brightness


if __name__ == "__main__":
    main()  # run tests
