"""Engine worker scripts. Each runs in its engine checkout's own Python (not this package's), as a
script: they import only the standard library, numpy, their engine and ``sfx_worker_protocol``."""
