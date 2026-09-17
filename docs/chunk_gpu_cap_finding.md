# Chunk's dynamic-sizing ceiling was discarding real GPU headroom

`dense_evolution.chunk.get_dynamic_chunk` already reads the ACTIVE compute
device's real available memory (`device.memory_stats()` on GPU/TPU, falling
back to `psutil.virtual_memory()` on CPU) -- that fix shipped earlier. But
the function's own final clamp, `max(16, min(max_bits, 27))`, threw away
real headroom regardless of what the device actually reported: 27 bits is
only 2.1 GB, so any GPU with more than ~2.5 GB free (nearly every real GPU,
Kaggle's T4/P100 included) was being capped well below what
`device.memory_stats()` itself said was safely usable.

## What was verified, not assumed

Ran a real Kaggle GPU kernel (T4/P100) that imports `dense_evolution`
directly and calls `get_dynamic_chunk(jnp.complex128)`:

- **Before** (existing PyPI code, ceiling at 27): would have printed `27`
  regardless of the device's real free memory.
- **After** (ceiling raised to 30 in a local branch, not yet promoted):
  printed **`29`** on the real GPU -- confirming the device-memory read was
  already correctly computing a value above 27, and the old hardcoded
  ceiling was the only thing suppressing it.

## Why 30, not unbounded

30 bits (~17 GB per chunk) is still a real ceiling -- avoids ever computing
a `chunk_size_bits` large enough to risk int32-indexing issues elsewhere in
the codebase -- it just no longer discards headroom a modern GPU actually
has. The function's own memory-based calculation already caps the *actual*
returned value at whatever the device really supports; raising the ceiling
only stops it from being overridden below that on hardware with more VRAM
than a 27-bit chunk needs.

## Status

Documented here first, per this repo's promotion convention -- not yet
merged into Dense-Evolution (the closed PR there, #278, explicitly deferred
to this write-up). Next step: re-propose the same fix as a new PR on
Dense-Evolution referencing this verification.
