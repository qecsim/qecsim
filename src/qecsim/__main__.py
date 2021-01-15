"""
This module exposes the qecsim command line interface (CLI) via module script.
"""

from qecsim.cli import cli

if __name__ == '__main__':
    cli(prog_name="python -m qecsim")
