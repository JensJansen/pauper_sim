"""Runnable self-check entry point for the drl_env package: `python -m drl_env`
from src/. The checks themselves live in drl_env/__init__.py (as
_run_self_checks, formerly the module's `if __name__ == "__main__"` block) so
they keep full access to every internal helper at package-module scope."""
import drl_env

drl_env._run_self_checks()
