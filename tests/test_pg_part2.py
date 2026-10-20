"""Validation experiments for Policy Gradient Part 2 (with baseline).

Run with:  python run.py tests --section pg2

Add one function per experiment (see tests/harness.py for everything ctx can record):

    @validation_test(
        section="pg2",
        title="Short name of the experiment",
        what="What the experiment changes or measures.",
        how="Why the outcome shows the implementation is correct, and what outcome is expected.",
    )
    def my_experiment(ctx):
        ...
        ctx.metric("some quantity", value)
        ctx.check("expected behaviour holds", condition, detail)
"""
from tests.harness import validation_test  # noqa: F401
