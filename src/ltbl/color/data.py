"""Data and definitions for the color sub-module."""

from rgbxy import Converter, GamutC

# Define the color gamut, search online and match against your bulbs
CONVERTER = Converter(GamutC)
