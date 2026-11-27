#!/usr/bin/env python3
"""Make the importing process ignore Ctrl+C. Imported only by the upscaler's forkserver process.

Ctrl+C reaches the whole process group, but only the main process should handle it (it cancels
the queued pages and lets the running ones finish). Python's forkserver ignores SIGINT only
after its preload imports are done, and each worker it forks starts with the handlers in place
at that point: imported first (see `upscaler._pool_context`), this module makes the server and
every worker ignore Ctrl+C from the start, so no import is cut off with a traceback.
"""

import signal

_ = signal.signal(signal.SIGINT, signal.SIG_IGN)
