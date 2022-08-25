"""Test the network sub-module."""

from unittest import TestCase, main

from ltbl.network.utils import get_bridge_IP, get_bridge_IP_old


class TestNetwork(TestCase):
    """"""

    def test_get_bridge_IP_old(self):
        """Test the nmap bridge IP scanner."""

        try:
            _ = get_bridge_IP_old()
        except RuntimeError:
            self.fail("could not find bridge IP")

    def test_get_bridge_IP(self):
        """Test the simple utility function."""

        try:
            _ = get_bridge_IP()
        except RuntimeError:
            self.fail("could not find bridge IP")


if __name__ == "__main__":
    main()
