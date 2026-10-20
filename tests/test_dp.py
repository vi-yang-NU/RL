"""Validation experiments for Section 2: Value and Policy Iteration.

Run with:  python run.py tests --section dp

Add one function per experiment (see tests/harness.py for everything ctx can record):

    @validation_test(
        section="dp",
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
