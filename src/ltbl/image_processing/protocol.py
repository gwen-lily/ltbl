"""Define ExtractionProtocol for palette extraction."""

from typing import Protocol

from ..color import Palette


class ExtractionProtocol(Protocol):
    """The Extraction Protocol structure definition."""

    def extract(*args, **kwargs) -> Palette:
        """Define the extraction protocol."""
