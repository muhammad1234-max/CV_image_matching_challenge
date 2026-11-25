# Live preflight

No Phase 5A.6 live measurement has been returned. Phase 5A.5 measured Python 3.11.11, PyTorch 2.5.1+cu121, a single visible Tesla T4, a 30 GiB cgroup and 19.50 GiB of export space after setup. Its estimated scratch requirement was 6.79 GiB. Production measures scratch and export separately, reads OOM events and pressure, selects GPU 0 by its current UUID, checks the second GPU and uses the operator's remaining session time. The downloaded preflight JSON records each completed check. The runtime uses clean-environment isolation.
