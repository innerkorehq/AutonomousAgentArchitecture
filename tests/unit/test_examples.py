"""Every example embedded in the documentation is executed here, so the docs cannot drift from the
code. Examples run offline against a scripted model (or ephemeral local state); the two cluster
examples need a database and run forever, so they are only compile-checked (the behavior they show is
covered by test_multiprocess.py, which launches real worker processes)."""

import os
import py_compile
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]
EXAMPLES = ROOT / "examples"

EXPECTED = {
    "quickstart.py": ["success - 17 x 23 = 391"],
    "inspect_log.py": [
        "0 aaa.run.started",
        "4 aaa.tool.started",
        "8 aaa.run.completed",
        "outcome: success",
    ],
    "data_codecs.py": [
        "ticket  f7199c89",
        "row     d578aa73",
        "event   81fa4071",
        "doc     2d87e6fc",
        "frame   78efaf7c",
        "col_a=1, col_b=x",
        "preflight: missing capabilities: scorer",
    ],
    "tool_approval.py": ["waiting for approval of: wire_money", "sent $250 to ACME"],
    "plan_and_execute.py": ["Tokyo + Delhi = 46 million people", "steps run: ['a', 'b', 'c']"],
    "generate_test_repair.py": ['success ["apple", "pear", "fig"] after 3 tries'],
    "router_supervisor.py": [
        "Refund issued.",
        "Findings: AAA is event-driven.",
        "Draft: AAA logs everything as events.",
    ],
    "workflow_fsm.py": ["workflow: 22", "fsm path: draft -> review -> draft -> review -> done"],
    "model_router.py": ["[small:fast-1] quick answer", "[big:deep-9] careful answer"],
    "streaming.py": ["Streaming works token by token.", "outcome: success"],
    "persistence_recovery.py": [
        "process 1 died with the run open",
        "process 2 recovered 1 run(s): result: value-of-x",
    ],
    "custom_algorithm.py": ["{'French': 'Bonjour', 'German': 'Hallo', 'Spanish': 'Hola'}"],
    "tool_guard.py": [
        "refund  -> at_most_once | max_attempts = 3",
        "search  -> at_least_once",
        "refunded A-100 (key=refund:A-100, attempt=1, fence=1)",
    ],
    "testing_replay.py": ["replayed identically: 5"],
    "rag_agent.py": ["Paris is the capital of France [d1]."],
    "large_payloads.py": [
        "got 60000 chars",
        "stored event keys: ['$claim', 'call_id', 'name']",
        "hydrated result length: 60000",
    ],
    "llm_gateways.py": [
        "POST https://api.portkey.ai/v1/chat/completions",
        "POST http://kong:8000/chat",
        "custom provider registered",
    ],
    "privacy_tenancy.py": [
        "answer: found C-7: [REDACTED:email], card [REDACTED:card]",
        "stored e-mail visible: False",
        "acme asking for globex's run: \"no run 'globex/ticket-1'\"",
        "erased 9 events; last event: aaa.run.erased",
        "payload left: False",
        "still finished: True",
        "globex untouched: True",
    ],
    "memory_agent.py": [
        "first run stored: ['Ann is vegetarian and lives in Lisbon']",
        "ann, new run: A vegetarian place in Lisbon: Ao 26.",
        "bob, same tenant: Which cuisine do you like?",
        "ann at another tenant: Which cuisine do you like?",
        "forgot: {'NativeMemory': 1}",
        "ann after forgetting: Which cuisine do you like?",
    ],
    "mcp_tools.py": [
        "weather__forecast        delivery=at_least_once approval=False",
        "weather__reset_station   delivery=at_most_once  approval=True",
        "The forecast says: Lisbon: sunny for the next 3 days",
    ],
    "context_management.py": [
        "without a policy the largest request was 13668 tokens (window 4000)",
        "with a policy the largest request was 3363 tokens",
        "the model was called 26 times and finished: Done: I looked up all 25 topics.",
        "compaction was applied on 19 calls, using: ['clear_tool_results', 'stored_summary', 'summarize']",
        "summaries written to the log: 4",
        "the log still holds every message: True",
    ],
    "chat_session.py": [
        "you: Hi, my name is Ann.",
        "bot: Noted (1 things said so far).",
        "bot: Your name is Ann.",
        "reattached: 3 turns on record",
        "messages sent to the provider per turn: [2, 2, 2]  (resending the history would be [2, 4, 6])",
        "turns continued from provider state: [False, True, True]",
        "erased 4 runs; provider state deleted: 3",
    ],
    "ensembles.py": [
        "vote: {'answer': '42', 'votes': 3, 'samples': 5, 'agreement': 0.6}",
        "critic loop: {'candidate': 'draft v2', 'iterations': 2}",
    ],
    "builtin_tools.py": [
        "fetch http://169.254.169.254/latest/meta-data/: refused (the host resolves to a non-public address)",
        "fetch http://localhost/admin: refused (the host resolves to a non-public address)",
        "fetch http://example.com:6379/: refused (port 6379 is not allowed)",
        "fetch https://evil.example.com/: refused (this host is not allowed)",
        "read notes.txt: project notes",
        "read ../secret.txt: refused (path is outside the allowed directory)",
        "code_exec waits for approval: code_exec",
        "The program printed: 5050",
    ],
    "reasoning_algorithms.py": [
        "reflexion: 42 km after 2 trials; lesson: ['State the unit.']",
        "debate: 12 from rounds [['12', '10', '12'], ['12', '12', '12']]",
        "swarm: Done: refunded A1 | handoffs: [['triage', 'billing']] | answered by: billing",
        "tree: (8 / (3 - 8 / 3)) | score 10.0 | at most 24 model calls",
    ],
    "eval_extras.py": [
        "from traces: {'task': 'Refund me, my email is [REDACTED:email]'}",
        "red team: attack success 90%, false positives 0%",
        "conversation: 2 turns; contains help = True",
    ],
    "environment_algorithms.py": [
        "computer: 20 euros a month | actions 1 | approvals asked 1",
        "codeact: The sum of the first ten squares is 385. | it ran: 385 | approvals asked 1",
        "anytime: 42 | tried ['quick', 'careful'] | stopped: accepted",
    ],
    "multi_agent_algorithms.py": [
        "mixture: Paris, the capital of France (3 proposers agreed) | layers [3, 3]",
        "blackboard: Launch on Tuesday, after the fix. | board ['researcher', 'critic']",
        "auction: winner expert | Done by the best-value bidder.",
        "protocol: {'approved': True, 'notes': 'three steps, fine'} | analyst document {'title': 'Migrate', 'steps': ['freeze', 'copy', 'switch']}",
        "human: the run is waiting on 'For how many people?'",
        "human: Table booked for 4. | asked 1",
    ],
    "memory_algorithms.py": [
        "skills: stored add_numbers | a failing test is refused | AssertionError",
        "skills: run -> 42 | other tenant sees []",
        "curriculum: [('add 20 and 22', True), ('add 1 and 2', True)] | skills ['add_numbers'] | done",
        "reflection: Ann is building a bakery in Lisbon. | sources ['m1', 'm0'] | stored 1",
        "self-managed: True (the next run starts with what the model chose to keep)",
        "optimise: 'Think step by step.' | dev 0.0 -> 1.0, held-out 0.0 -> 1.0 | improved: True",
    ],
    "retrieval_algorithms.py": [
        "corrective: Answered from web:w1 | route web | sources ['web:w1']",
        "corrective: Answered from fr | route index | graded 2 passages",
        "self-rag: Answered from fr | supported full | from fr",
        "adaptive: iterative with 2 searches; Answered from fr, de",
    ],
    "planning_algorithms.py": [
        "rewoo: Paris has 2.1 million people. | evidence {'E1': 'Paris', 'E2': '2.1 million', 'E3': '4'} | model calls: 2",
        "llm-compiler: Paris has 2.1 million people; 2 + 2 = 4. | rounds 2",
        "manager: Report: Paris is the capital of France. | sub-goals ['research', 'write']",
        "plan-and-execute with review: ['gather', 'r1-check']",
    ],
    "verification_algorithms.py": [
        "self-refine: 42 km (units added) after 1 rewrite; accepted",
        "verification: The capital of Australia is Canberra. | draft was: The capital of Australia is Sydney.",
        "best-of-n: the longest answer of all | scores [5.0, 10.0, 15.0, 25.0]",
        "least-to-most: 12 via ['Legs per dog?', 'How many dogs?']",
        "self-ask: 1969. after 1 follow-up",
        "graph-of-thoughts: a+b from 11 thoughts; 9 calls at most",
        "lats: 24 | path ['step 1: 8 / (3 - 8 / 3)', 'FINAL: 24'] | solved True",
    ],
    "mcp_everything.py": [
        "title: Everything (production)",
        "tools: ['getOrder', 'up_shout', 'word_count']",
        "up_shout: HELLO",
        "policies: ['up_shout', 'word_count']",
    ],
    "mcp_init.py": [
        "drift right after generating: none",
        "secret names (never values): ['ORDERS_API_TOKEN']",
        "after editing the Dockerfile: ['Dockerfile: differs from its template (and was edited by hand)']",
    ],
    "mcp_openapi.py": [
        "offered: getOrder | Fetch one order | read-only: True",
        "hidden (state-changing, need approval): ['cancelOrder', 'createOrder']",
        "1 operation(s) skipped as unsupported: uploadReceipt",
    ],
    "mcp_modules.py": [
        "tool: word_count | Count the words in a text. | ['text']",
        "hidden (need approval): ['delete_file']",
        "word_count -> 3",
    ],
    "mcp_resources.py": [
        "acme sees: ['aaa://files/handbook/handbook.md']",
        "acme reading globex's file: resource not found",
        "globex searches the faq: ['r2']",
        "prompt: Summarize the handbook for a new hire in 50 words.",
    ],
    "mcp_builder.py": [
        "tool: ask_support | None | {'openWorldHint': True}",
        "tool: lookup_order | Look up an order | {'readOnlyHint': True}",
        "answer: Order A1 shipped yesterday.",
    ],
    "mcp_generated.py": [
        "offers: desk_forecast | Weather forecast | {'readOnlyHint': True, 'openWorldHint': True}",
        "hidden (need approval): ['desk_erase_history']",
        'structured result: {"city": "Oslo", "celsius": 21.5}',
    ],
    "mcp_server.py": [
        "offered: ['add', 'lookup', 'run_agent']",
        "add: 5",
        "lookup: lookup failed",
        "wire_money: unknown tool 'wire_money' | error: True",
        'run_agent: {"kind": "success", "value": {"text": "Paris.", "turns": 1}, "reason": null}',
    ],
    "server_quickstart.py": [
        "no credentials : 401",
        "waiting for    : wire_money {'amount': 250}",
        "other tenant   : 404 not_found",
        "approved       : 202",
        "event stream   : ['run.started', 'tool.requested', 'approval.requested', 'tool.completed', 'run.completed', 'end']",
        'outcome        : "Done: sent $250"',
    ],
    "ingest_docs.py": [
        "first sync : added=2 updated=0 unchanged=0 removed=0 embedded_chunks=3",
        "second sync: added=0 updated=0 unchanged=2 removed=0 embedded_chunks=0",
        "after edit : added=0 updated=1 unchanged=1 removed=0 embedded_chunks=1",
        "after rm   : added=0 updated=0 unchanged=1 removed=1 embedded_chunks=0",
        "usage.md | Usage > Tips | Keep tools small.",
    ],
    "native_providers.py": ["anthropic: The sum is 5.", "gemini: The sum is 5."],
    "multimodal.py": [
        "answer: I can see 1 image (image/png, larger than the inline limit: True).",
        "picture bytes in the log: False",
        "blobs stored: 1",
        "blobs after erasing the run: 0",
    ],
    "vector_search.py": [
        "all: ['a', 'd', 'b', 'c']",
        "english, 2023 or later: ['a', 'c']",
        "acme sees: ['acme invoice terms']",
    ],
    "guardrails.py": [
        "the model saw the SSN: False",
        "the model saw the injection: False",
        "[tool result withheld by policy: prompt_injection]",
        "blocked by policy: tool_not_allowed",
    ],
    "evals.py": [
        "pass rate 67% over 3 runs",
        "contains: mean 0.67",
        "failed: ['vague']",
        "gate: pass rate 66.7% < 90.0%",
    ],
    "observability.py": [
        "invoke_agent tool_loop",
        "execute_tool lookup",
        "spans exported: 4 in 1 batch",
        "kinds: ['agent', 'llm', 'tool', 'llm']",
    ],
}
COMPILE_ONLY = {"cluster_worker.py", "cluster_client.py"}
OPTIONAL_DEPS = {
    "mcp_tools.py": "mcp",
    "mcp_everything.py": "mcp",
}  # skipped where the optional package is not installed


def test_every_example_is_accounted_for():
    on_disk = {p.name for p in EXAMPLES.glob("*.py")}
    assert on_disk == set(EXPECTED) | COMPILE_ONLY, "add new examples to EXPECTED (and the docs)"


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_example_runs_and_prints_what_the_docs_say(name):
    if dep := OPTIONAL_DEPS.get(name):
        pytest.importorskip(dep)
    proc = subprocess.run(
        [sys.executable, "-I", str(EXAMPLES / name)],
        check=False,
        capture_output=True,
        text=True,
        timeout=90,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
    )
    assert proc.returncode == 0, f"{name} failed:\n{proc.stdout}\n{proc.stderr}"
    for expected in EXPECTED[name]:
        assert expected in proc.stdout, f"{name}: missing {expected!r} in:\n{proc.stdout}"


@pytest.mark.parametrize("name", sorted(COMPILE_ONLY))
def test_cluster_examples_at_least_compile(name, tmp_path):
    py_compile.compile(str(EXAMPLES / name), cfile=str(tmp_path / "x.pyc"), doraise=True)


DOCS = ROOT / "docs"
MARKER = __import__("re").compile(r'^--8<-- "(?P<path>[^"]+)"$')


def test_docs_embed_markers_are_well_formed_and_point_at_real_files():
    """A mangled `--8<--` marker still builds (it is just text) but publishes a broken page."""
    problems = []
    for md in DOCS.rglob("*.md"):
        for n, line in enumerate(md.read_text().splitlines(), 1):
            if (
                line.startswith("--8")
                or "--8 <" in line
                or "8<--" in line
                and not line.lstrip().startswith("`")
            ):
                m = MARKER.match(line)
                if not m or not (ROOT / m["path"]).is_file():
                    problems.append(f"{md.relative_to(ROOT)}:{n}: {line!r}")
    assert not problems, "\n".join(problems)


def test_every_example_is_embedded_somewhere_in_the_docs():
    embedded = set()
    for md in DOCS.rglob("*.md"):
        embedded |= {
            m["path"].split("/")[-1]
            for line in md.read_text().splitlines()
            if (m := MARKER.match(line))
        }
    missing = {p.name for p in EXAMPLES.glob("*.py")} - embedded
    assert not missing, f"examples not shown in the docs: {sorted(missing)}"
