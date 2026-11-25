# Operational amendment

The historical wrapper summed RSS over the Python worker and all descendants. Each process counts resident shared libraries, shared mappings, file-backed checkpoints and fork-inherited resident pages in its own RSS. The old receipts cannot identify which mappings were shared or whether either VM was physically short of memory.

The recovery notebooks embed the original worker and adapter with their original pins. Only the surrounding monitor, run receipts, identity, timing and export path change. The SHA-256 values of the original policy, adapter, amended policy, monitor and both notebooks appear in reports/phase4g1/manifest.json. The historical model hash is af0eaa71875e61f265284e45fb13b2770b1bd711e4ec7941873c53832be02c01.

Every two seconds the wrapper records per-process smaps rollups, host and cgroup headroom, swap inventory, cgroup OOM counters, host memory pressure and device-wide VRAM. PSS apportions shared pages among mapping processes. USS counts private resident pages. Missing rollups are marked and totals can be partial. Cgroup usage includes charged cache and other members; its visible hierarchy and host MemAvailable bound usable headroom. Sampling can miss spikes and short-lived processes.

Abort if host or any finite visible cgroup has at most max(1 GiB, 10% of its limit) available, a new cgroup OOM event occurs, full memory pressure averages at least 10% over ten seconds, or free GPU memory reaches 3% or less. Fail closed on missing essential constraints, including no finite visible cgroup limit. These are operational margins, not guarantees against sudden allocations. Limits are measured live. The preflight records the operator's remaining time estimate, the stopwatch bound, GPU, swap and process inventory. The stop path terminates the worker process group and exports evidence with 90 seconds reserved. A failed preflight launches no capture.

Start with Stairs in a fresh unmounted Colab T4. Save and review its archive before starting the second notebook in another fresh session. Preserve the parent separately when present. Do not repeat a failed capture without reviewing its evidence.
