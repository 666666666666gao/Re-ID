#!/usr/bin/env python3
"""Run the complementarity-activated Signal-preserving V6 experiment."""


def main() -> None:
    from run_signal_preserving_v5 import parse_args, run

    run(parse_args())


if __name__ == "__main__":
    main()
