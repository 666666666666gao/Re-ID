#!/usr/bin/env python3
"""Run geometry-aligned marginal-gain-routed Signal-preserving V7."""


def main() -> None:
    from run_signal_preserving_v5 import parse_args, run

    run(parse_args())


if __name__ == "__main__":
    main()
