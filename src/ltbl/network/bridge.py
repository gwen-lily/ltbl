"""Define the Philips Hue Bridge and activate it."""
from phue import Bridge

from .utils import get_bridge_IP

BRIDGE = Bridge(get_bridge_IP)
BRIDGE.connect()
