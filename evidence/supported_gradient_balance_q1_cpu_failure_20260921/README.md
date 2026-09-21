# Q1 CPU verification failure and arithmetic repair

2026-09-21 10:29:22: all six endpoints completed; original Q1 exited 0. Original CPU verifier exited 1 at 10:29:28 with exact ratio equality assertion. Original pipeline remains STOPPED_AT_Q1_CPU; no restart or training update.

Full saved scalar scan: six files, 1560 steps, 3486 supported role rows. Every recorded ratio exactly matches runtime math.sqrt. Six rows differ from verifier exponentiation x**0.5 by one ULP, 2.220446049250313e-16. The minimal verifier repair uses math.sqrt while retaining exact equality and all existing thresholds. No change to training, config, checkpoint, scientific gates or results. Full repaired verification and independent audit remain pending. No scientific qualification asserted.
