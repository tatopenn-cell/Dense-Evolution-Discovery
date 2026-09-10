# Run on Google Colab (GPU runtime not required for this one -- just
# inspecting installed source).
#
# Cell 1:
#   !pip install -q dense-evolution==8.1.76
#
# Cell 2: the code below.
#
# Prints the EXACT source of the installed _build_mps_runner (and
# _bucket_sizes), to compare byte-for-byte against a hand-copied
# reimplementation, rather than guessing why the real sim._mps_runner
# measured much slower than that reimplementation under seemingly
# identical conditions. Result: byte-identical -- the source was never
# the cause; see mps_bucketed_svd_gpu_timing_followup.md for the real one
# (a flawed "fresh instance per timing" benchmark methodology).
import inspect
import dense_evolution as de
from dense_evolution.backends import mps as mps_module

print("dense_evolution version:", de.__version__)
print("=" * 80)
print(inspect.getsource(mps_module._bucket_sizes))
print("=" * 80)
print(inspect.getsource(mps_module._build_mps_runner))
