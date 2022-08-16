"""Test the Color module."""

from unittest import TestCase

from ...src.ltbl import Color

Red = Color(255, 0, 0)
Green = Color(0, 255, 0)
Blue = Color(0, 0, 255)
White = Color(255, 255, 255)
Black = Color(0, 0, 0)

color_1 = Color(0.5, 0.5)
color_2 = Color(0.3, 0.4)


class TestColor(TestCase):
    """"""

    def test_r(self):
        self.assertEqual(Red.r, 255)
        self.assertIsInstance(Red.r, int)

    def test_g(self):
        self.assertEqual(Green.g, 255)
        self.assertIsInstance(Green.g, int)

    def test_b(self):
        self.assertEqual(Blue.b, 255)
        self.assertIsInstance(Blue.b, int)

    def test_rgb(self):
        for channel in White.rgb:
            self.assertEqual(255, channel)

    def test_rgb01(self):
        for channel in Black.rgb:
            self.assertEqual(0, channel)

    def test_xy(self):
        for channel in color_1.xy:
            self.assertEqual(0.5, channel)

    def test_hex(self):
        white_hex = "ffffff"

        self.assertEqual(white_hex, White.hex.lower())
