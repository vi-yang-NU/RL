"""Unit tests (standard library unittest, no extra dependency).

Run with:  python run.py unittest            (or: python -m unittest discover -s tests/unit -t .)
           python run.py unittest -k solver  only tests whose name contains "solver"

These check the code's building blocks and edge cases quickly (seconds). The validation experiments the
handout asks for live in tests/test_dp.py, tests/test_pg_part1.py and tests/test_pg_part2.py and run with
`python run.py tests`.
"""
try:  # tiny networks: PyTorch's default thread count slows the tests down on many-core machines
    import torch

    torch.set_num_threads(2)
except ImportError:  # pragma: no cover
    pass
