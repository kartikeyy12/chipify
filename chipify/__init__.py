"""Chipify: convert instrumental audio into authentic NES-style chiptune."""
import os

# Headless plotting for WSL2/servers. Must be set before matplotlib is imported
# anywhere in the process, so it lives at package import time.
os.environ.setdefault("MPLBACKEND", "Agg")

__version__ = "0.2.0"
