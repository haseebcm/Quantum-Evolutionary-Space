"""Demonstrate the QSEE-11L isolated intelligence-stream architecture.

Run with:  python examples/qsee_11l_demo.py
"""
from __future__ import annotations

from qes.qsee import QSEE11L


def main() -> None:
    system = QSEE11L()

    system.evolve(
        {
            1: lambda s, add=1: add if s is None else s + add,
            2: lambda s, add=2: add if s is None else s + add,
            3: lambda s, add=3: add if s is None else s + add,
            4: lambda s, add=4: add if s is None else s + add,
            5: lambda s, add=5: add if s is None else s + add,
            6: lambda s, add=6: add if s is None else s + add,
            7: lambda s, add=7: add if s is None else s + add,
            8: lambda s, add=8: add if s is None else s + add,
            9: lambda s, add=9: add if s is None else s + add,
            10: lambda s, add=10: add if s is None else s + add,
            11: lambda s, add=11: add if s is None else s + add,
        }
    )

    output = system.request_output(lambda states: sum(states))
    print("QSEE-11L demo")
    print("=" * 40)
    print("Aligned states:", output.aligned_states)
    print("Combined output:", output.output)
    print("QEL stream 1 state:", system.gate.qels[0].state)


if __name__ == "__main__":
    main()
