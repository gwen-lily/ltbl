"""Utilities for the network sub-module."""

import nmap

from .data import NETWORK, BRIDGE_DEVICE_NAME


def get_bridge_IP() -> str:
    # set up and scan network
    nmScan = nmap.PortScanner()
    nmScan.scan(NETWORK)

    # check each host, returning host IP if matched
    for host in nmScan.all_hosts():
        vendor = nmScan[host]["vendor"]

        try:
            name = list(vendor.values)[0]
            if name == BRIDGE_DEVICE_NAME:
                return host

        except IndexError:
            continue

    # program can't run without the Bridge
    raise RuntimeError(nmScan.all_hosts())
