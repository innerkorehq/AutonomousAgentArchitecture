"""A child run belongs to its parent's user: it inherits the parent's `meta` unless it is given its own."""

from aaa.algos.base import finish
from aaa.core.outcome import OutcomeKind
from aaa.runtime import Engine


class Leaf:
    name = "leaf"

    def subscriptions(self):
        return ["aaa.run.started"]

    async def on_event(self, e, ctx):
        return [finish(e, OutcomeKind.SUCCESS, {"meta": dict(e.meta)})]


class Parent:
    name = "parent"

    def __init__(self, child_meta):
        self.child_meta = child_meta

    def subscriptions(self):
        return ["aaa.run.started", "aaa.child.completed"]

    async def on_event(self, e, ctx):
        if e.type == "aaa.run.started":
            await ctx.engine.start(
                Leaf(),
                {},
                parent=ctx.run_id,
                tag="c",
                run_id=f"{ctx.run_id}-c",
                meta=self.child_meta,
            )
            return []
        return [finish(e, OutcomeKind.SUCCESS, e.data["outcome"]["value"])]


async def test_children_inherit_the_parents_user_and_an_explicit_meta_wins():
    async with Engine() as eng:
        inherited = await eng.run(
            Parent(None), {}, tenant="acme", meta={"user": "ann", "agent": "bot"}
        )
        own = await eng.run(Parent({"user": "bob"}), {}, tenant="acme", meta={"user": "ann"})
        bare = await eng.run(Parent(None), {}, tenant="acme")
    assert inherited.value == {"meta": {"user": "ann", "agent": "bot"}}
    assert own.value == {"meta": {"user": "bob"}}
    assert bare.value == {"meta": {}}
