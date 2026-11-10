"""python -m evaluation <command> ...   (run from src/)

  strength   gauntlet / head2head / roundrobin between agents from any checkpoint directories   (cli_strength.py)
  history    matrix / field: strength over a league's archived snapshots                         (cli_history.py)
  probe      record / run / report: policy movement on a fixed set of mid-game decisions         (cli_probe.py)

`python -m evaluation <command> --help` shows that command's own options."""
import importlib
import sys

COMMANDS = {"strength": "cli_strength", "history": "cli_history", "probe": "cli_probe"}


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] not in COMMANDS:
        print(__doc__)
        return 0 if not argv else 2
    importlib.import_module(f"evaluation.{COMMANDS[argv[0]]}").main(argv[1:])
    return 0


if __name__ == "__main__":
    sys.exit(main())
