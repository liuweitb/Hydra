"""Dump PythonPipeline default config (all valid keys + defaults) from the compiled bindings."""
import sys

from hydra_python._hydra_bindings import HydraPipeline

text = HydraPipeline.default_config()
open(sys.argv[1], "w").write(text)
print(len(text.splitlines()), "lines")
