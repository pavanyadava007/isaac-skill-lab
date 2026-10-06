"""Run a script's main() inside Kit and always exit cleanly.

Kit's shutdown (SimulationApp.close) hangs in its stop handler on this headless setup, so the process
exits hard once main() has returned and every output file is written.
"""

from __future__ import annotations

import os
import sys
import traceback


def run(main, app) -> None:
    try:
        main()
    except BaseException:  # noqa: BLE001
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    sys.stdout.flush()
    sys.stderr.flush()
    del app  # SimulationApp.close() is skipped on purpose, see module docstring
    os._exit(0)
