"""Utilities for the network sub-module."""

import requests as req
from time import sleep
from .data import NETWORK, BRIDGE_DEVICE_NAME


def get_bridge_IP_old() -> str:
    """# TODO: Document this

    Returns
    -------
    str
        _description_

    Raises
    ------
    RuntimeError
        _description_
    """
    import nmap

    # set up and scan network
    nmScan = nmap.PortScanner()
    nmScan.scan(NETWORK)

    # check each host, returning host IP if matched
    for host in nmScan.all_hosts():
        vendor = nmScan[host]["vendor"]

        try:
            values = list(vendor.values())
            name = values[0]
            if name == BRIDGE_DEVICE_NAME:
                return host

        except IndexError:
            continue

    # program can't run without the Bridge
    raise RuntimeError(nmScan.all_hosts())


def get_bridge_IP() -> str:
    """Returns the internal ip address of a hue bridge on your LAN"""

    site = "https://discovery.meethue.com"

    get = req.get(site)
    
    # sleep to prevent auto-insta query
    sleep(1)

    if get:
        # json comes in a list of length 1, why? idk
        return get.json()[0]['internalipaddress']
    
    raise RuntimeError(f"Bridge not found. Try connecting to {site}.")
