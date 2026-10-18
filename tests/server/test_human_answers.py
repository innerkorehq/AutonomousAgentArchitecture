"""POST /v1/runs/{id}/answers: a person answers the question an AskHuman run is waiting on, with its own scope, tenant isolation and one answer per question."""

from __future__ import annotations

import asyncio
import json

from aaa_server import ServerConfig
from aaa_server.auth import ApiKeyAuthenticator

from aaa.algos import AskHuman
from aaa.llm import FakeLLMClient, LLMHandler
from aaa.llm.fake import text
from aaa.runtime import Budget, Engine
from helpers.server_env import KEYS, Env


async def make(turns):
    it = iter(turns)
    engine = Engine()
    LLMHandler(FakeLLMClient(lambda req: text(json.dumps(next(it))))).attach(engine)
    algo = AskHuman(max_questions=2)
    cfg = ServerConfig(
        engine=engine,
        authenticator=ApiKeyAuthenticator.from_keys(KEYS),
        algorithms={"ask": algo},
        budget=Budget(max_steps=20, max_tokens=10_000, deadline_s=30.0),
        keepalive_s=0.2,
    )
    return Env(engine, cfg, algo), engine  # type: ignore[arg-type]


async def pending(env, rid):
    for _ in range(200):
        for e in await env.engine.store.read(f"acme/{rid}"):
            if e.type == "aaa.human.requested":
                return e.data["question_id"]
        await asyncio.sleep(0.02)
    raise AssertionError("no question was asked")


ASK = {"action": "ask", "question": "For how many people?"}
FINAL = {"action": "answer", "answer": "Table booked for 4."}


async def test_an_answerer_unblocks_the_run_and_only_once():
    env, engine = await make([ASK, FINAL])
    async with (
        engine,
        env.client("key-acme") as starter,
        env.client("key-acme-answerer") as answerer,
    ):
        rid = (
            await starter.post("/v1/runs", json={"algo": "ask", "input": {"task": "Book a table"}})
        ).json()["run_id"]
        qid = await pending(env, rid)
        ok = await answerer.post(
            f"/v1/runs/{rid}/answers", json={"question_id": qid, "answer": "4"}
        )
        assert ok.status_code == 202
        for _ in range(200):
            run = (await starter.get(f"/v1/runs/{rid}")).json()
            if run["status"] == "completed":
                break
            await asyncio.sleep(0.02)
        assert run["outcome"]["value"]["answer"] == "Table booked for 4."
        assert run["outcome"]["value"]["questions"] == [
            {"question": "For how many people?", "answer": "4"}
        ]
        again = await answerer.post(
            f"/v1/runs/{rid}/answers", json={"question_id": qid, "answer": "5"}
        )
        assert again.status_code == 409 and again.json()["error"]["code"] == "no_pending_question"
        signalled = [
            e for e in await engine.store.read(f"acme/{rid}") if e.type == "aaa.human.answered"
        ]
        assert (
            len(signalled) == 1 and signalled[0].data["by"] == "amy"
        )  # who answered is in the log


async def test_the_answer_scope_is_required_and_is_not_the_approve_or_run_scope():
    env, engine = await make([ASK, FINAL])
    async with (
        engine,
        env.client("key-acme") as run_only,
        env.client("key-acme-approver") as approver,
    ):
        rid = (
            await run_only.post("/v1/runs", json={"algo": "ask", "input": {"task": "t"}})
        ).json()["run_id"]
        qid = await pending(env, rid)
        for client in (run_only, approver):
            r = await client.post(
                f"/v1/runs/{rid}/answers", json={"question_id": qid, "answer": "4"}
            )
            assert r.status_code == 403
        assert [
            e for e in await engine.store.read(f"acme/{rid}") if e.type == "aaa.human.answered"
        ] == []


async def test_a_question_that_was_never_asked_cannot_be_answered_in_advance():
    env, engine = await make([ASK, FINAL])
    async with (
        engine,
        env.client("key-acme") as starter,
        env.client("key-acme-answerer") as answerer,
    ):
        rid = (await starter.post("/v1/runs", json={"algo": "ask", "input": {"task": "t"}})).json()[
            "run_id"
        ]
        await pending(env, rid)
        for qid in ("q1", "q9", "guess"):
            r = await answerer.post(
                f"/v1/runs/{rid}/answers", json={"question_id": qid, "answer": "x"}
            )
            assert r.status_code == 409


async def test_another_tenant_gets_the_same_404_as_for_a_missing_run_and_input_is_validated():
    env, engine = await make([ASK, FINAL])
    async with (
        engine,
        env.client("key-acme") as starter,
        env.client("key-globex-answerer") as foreign,
        env.client("key-acme-answerer") as answerer,
    ):
        rid = (await starter.post("/v1/runs", json={"algo": "ask", "input": {"task": "t"}})).json()[
            "run_id"
        ]
        qid = await pending(env, rid)
        body = {"question_id": qid, "answer": "4"}
        a = await foreign.post(f"/v1/runs/{rid}/answers", json=body)
        b = await foreign.post("/v1/runs/does-not-exist/answers", json=body)
        assert (
            a.status_code == b.status_code == 404
            and a.json()["error"]["code"] == b.json()["error"]["code"]
        )
        for bad in (
            {"question_id": qid},
            {"question_id": qid, "answer": "x" * 2001},
            {"question_id": qid, "answer": "4", "tenant": "globex"},
        ):
            assert (await answerer.post(f"/v1/runs/{rid}/answers", json=bad)).status_code == 422


async def test_the_question_appears_on_the_event_stream():
    env, engine = await make([ASK, FINAL])
    async with (
        engine,
        env.client("key-acme") as starter,
        env.client("key-acme-answerer") as answerer,
    ):
        rid = (await starter.post("/v1/runs", json={"algo": "ask", "input": {"task": "t"}})).json()[
            "run_id"
        ]
        qid = await pending(env, rid)
        await answerer.post(f"/v1/runs/{rid}/answers", json={"question_id": qid, "answer": "4"})
        stream = (await starter.get(f"/v1/runs/{rid}/events")).text
    assert "event: human.requested" in stream and "For how many people?" in stream
    assert (
        '"answer"' not in stream.split("event: human.requested")[1].split("\n\n")[0]
    )  # the answer itself is not broadcast
