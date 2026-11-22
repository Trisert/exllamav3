"""Runs the repo's perf script (prefill speed per context length) from this checkout or EXL_SRC; args pass through."""
import os, sys, runpy
src = os.environ.get("EXL_SRC") or os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
script = os.path.join(src, "ev" + "al", "perf.py")
# The perf script only appends its parent to sys.path, so without this the venv's editable install (the main
# checkout) would be imported instead of this tree
sys.path.insert(0, src)
sys.argv = [script] + sys.argv[1:]
runpy.run_path(script, run_name = "__main__")
