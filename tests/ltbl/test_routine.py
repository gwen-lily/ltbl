"""Test the routine sub-module."""

from unittest import TestCase

from ltbl import BRIDGE, Color
from ltbl.routine import LoopLights, RandomWalk

Red = Color(255, 0, 0)
Green = Color(0, 255, 0)
Blue = Color(0, 0, 255)

RGB_Palette = [Red, Green, Blue]

_TEST_TIME_LIMIT = 15  # seconds


class TestRoutine(TestCase):
    def test_LoopLights(self):
        routine = LoopLights(BRIDGE.lights, RGB_Palette, time_limit=_TEST_TIME_LIMIT)
        routine.loop()

    def test_RandomWalk(self):
        routine = RandomWalk(BRIDGE.lights, RGB_Palette, time_limit=_TEST_TIME_LIMIT)
        routine.loop()
