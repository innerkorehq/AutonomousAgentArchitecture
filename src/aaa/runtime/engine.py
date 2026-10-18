from __future__ import annotations

import asyncio
import fnmatch
import json
import logging
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Coroutine, Sequence
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, Protocol, Self, TypeVar

from pydantic import BaseModel, Field
from pydantic_core import to_jsonable_python

from aaa.core.aio import raise_if_cancelling
from aaa.core.errors import Redeliver, StoreUnavailable
from aaa.core.ids import new_id
from aaa.core.outcome import Outcome, OutcomeKind
from aaa.core.retry import RetryPolicy, retrying
from aaa.core.retry import is_transient as default_is_transient
from aaa.data.blob import BlobStore
from aaa.events import (
    Event,
    EventBus,
    EventStore,
    Handler,
    InMemoryEventStore,
    InProcessBus,
    Projection,
    SchemaRegistry,
)
from aaa.privacy import Redactor
from aaa.runtime import media as _media
from aaa.runtime.budget import Budget, Ledger
from aaa.runtime.retention import ErasureReport, MemoryReport, PurgeReport, RetentionPolicy
from aaa.runtime.tenancy import TenantScope, valid_tenant

log = logging.getLogger("aaa.engine")
S = TypeVar("S")

RUN_STARTED = "aaa.run.started"
RUN_COMPLETED = "aaa.run.completed"
RUN_CANCEL = "aaa.run.cancel"
CHILD_STARTED = "aaa.child.started"
CHILD_COMPLETED = "aaa.child.completed"
CLAIM_KEY = "$claim"
# Small, hot fields that stay on the stored event when its payload is claim-checked into a blob,
# so folds that only need them (spend, call ids, tags) never have to fetch the blob.
CLAIM_KEEP = ("usage", "cost", "tag", "call_id", "name", "state_id")


class RunMeta(BaseModel):
    """Everything the engine needs to know about a run, as a pure fold of its log: spend,
    whether/how it finished, and which child runs are still open. Because it is derived, any
    process can rebuild it (from a snapshot + tail) and no process-local state is authoritative."""

    ledger: Ledger = Field(default_factory=Ledger)
    completed: dict[str, Any] | None = None
    children: dict[str, str | None] = Field(default_factory=dict)  # open child run id -> tag


def _apply_meta(m: RunMeta, e: Event) -> RunMeta:
    m.ledger = m.ledger.fold(e)
    if e.type == RUN_COMPLETED and m.completed is None:
        m.completed = e.data
    elif e.type == CHILD_STARTED:
        m.children[e.data["child_run_id"]] = e.data.get("tag")
    elif e.type == CHILD_COMPLETED:
        m.children.pop(e.data["child_run_id"], None)
    return m


RUNMETA = Projection(RunMeta, _apply_meta, name="aaa.runmeta", version=1, state_type=RunMeta)


@dataclass
class _Run:
    budget: Budget
    future: asyncio.Future[Outcome[Any]]
    trace_id: str
    algo: Any = None
    parent: str | None = None
    tag: str | None = None
    tenant: str | None = None
    meta: RunMeta = field(default_factory=RunMeta)
    synced: int = -1  # last event seq folded into `meta`
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    done: bool = False
    tripped: bool = False
    timer: asyncio.Task[None] | None = None

    @property
    def ledger(self) -> Ledger:
        return self.meta.ledger


class Ctx:
    """Per-event context handed to algorithms. Algorithms hold no run state of their own:
    everything is derived from the run's event stream, so runs resume, replay and migrate
    between processes."""

    def __init__(self, engine: Engine, run_id: str | None) -> None:
        self.engine, self.run_id = engine, run_id

    async def history(self, from_seq: int = 0) -> list[Event]:
        """The run's events with any claim-checked payloads restored."""
        assert self.run_id
        rid = self.run_id
        raw = await self.engine.retry(lambda: self.engine.store.read(rid, from_seq), "read")
        return [await self.engine.hydrate(e) for e in raw]

    async def project(self, projection: Projection[S]) -> S:
        """Current state of `projection` for this run: latest snapshot + tail (O(tail))."""
        assert self.run_id
        return await self.engine.project(projection, self.run_id)

    async def started(self) -> Event:
        """The run's `aaa.run.started` event (input, budget, parent)."""
        assert self.run_id
        return await self.engine._started_of(self.run_id)

    def ledger(self) -> Ledger:
        run = self.engine._runs.get(self.run_id or "")
        return run.ledger if run else Ledger()


class Algorithm(Protocol):
    name: str

    def subscriptions(self) -> list[str]:
        """Event-type glob patterns this algorithm reacts to."""
        ...

    async def on_event(self, event: Event, ctx: Ctx, /) -> list[Event]:
        # positional-only so implementations may name the parameters anything (`e`, `evt`, ...)
        ...


class RunHandle:
    def __init__(self, engine: Engine, run_id: str) -> None:
        self._engine, self.run_id = engine, run_id

    async def result(self) -> Outcome[Any]:
        """Outcome of the run, wherever (whichever process) it was hosted or finished."""
        run = self._engine._runs.get(self.run_id)
        if run is not None and run.future.done():
            return run.future.result()
        waiters: list[asyncio.Future[Outcome[Any]]] = []
        if run is not None:
            waiters.append(asyncio.ensure_future(asyncio.shield(run.future)))
        waiters.append(asyncio.ensure_future(self._engine._await_completed(self.run_id)))
        try:
            done, _ = await asyncio.wait(waiters, return_when=asyncio.FIRST_COMPLETED)
            return next(iter(done)).result()
        finally:
            for w in waiters:
                if not w.done():
                    w.cancel()

    async def events(self, *, hydrate: bool = False) -> AsyncIterator[Event]:
        """Replay the run's stream from the start, then tail until the run completes."""
        async for e in self._engine.store.subscribe(self.run_id):
            yield await self._engine.hydrate(e) if hydrate else e
            if e.type == RUN_COMPLETED:
                return

    async def cancel(self, reason: str = "cancelled") -> None:
        await self.signal(RUN_CANCEL, {"reason": reason})

    async def signal(self, type: str, data: dict[str, Any] | None = None) -> Event | None:
        """Inject an external event (human approval, webhook, ...) into the run."""
        return await self._engine.emit(
            Event(
                type=type,
                data=data or {},
                stream=self.run_id,
                run_id=self.run_id,
                trace_id=self._engine._trace(self.run_id),
                source="external",
            )
        )


class Engine:
    """Owns store + bus + schemas (+ optional blob store). Everything persisted is appended to
    the log BEFORE it is delivered, so the log is the source of truth and delivery replayable.

    Cluster-ready: per-run state in `_runs` is only a cache of a fold of the log, rebuilt on
    demand ("adopted") by whichever process the bus routes a run's events to. With a
    distributed bus, several Engine processes sharing one store cooperate on the same runs."""

    def __init__(
        self,
        store: EventStore | None = None,
        bus: EventBus | None = None,
        schemas: SchemaRegistry | None = None,
        default_budget: Budget | None = None,
        blobs: BlobStore | None = None,
        claim_threshold: int = 16_384,
        media_threshold: int = _media.THRESHOLD,
        sweep_interval_s: float | None = None,
        store_retry: RetryPolicy | None = None,
        is_transient: Callable[[BaseException], bool] | None = None,
        redactor: Redactor | None = None,
        require_tenant: bool = False,
        retention: Sequence[RetentionPolicy] = (),
        memory: Sequence[Any] = (),
    ) -> None:
        self.store: EventStore = store or InMemoryEventStore()
        self.bus = bus or InProcessBus()
        self.distributed: bool = getattr(self.bus, "distributed", False)
        self.schemas = schemas or SchemaRegistry()
        self.default_budget = default_budget or Budget()
        self.blobs, self.claim_threshold = blobs, claim_threshold
        self.media_threshold = (
            media_threshold  # attachments of at least this many bytes go to the blob store
        )
        self.sweep_interval_s = sweep_interval_s
        # How long to ride out a store outage before asking the bus to redeliver instead.
        self.store_retry = store_retry or RetryPolicy()
        self.is_transient = is_transient or default_is_transient
        # Privacy: `redactor` scrubs every event before it is persisted or delivered; `require_tenant`
        # refuses runs that carry no tenant; `retention` is applied by `sweep()`.
        self.redactor, self.require_tenant, self.retention = (
            redactor,
            require_tenant,
            tuple(retention),
        )
        # Long-term memory backends (aaa.memory): `forget_memory` reaches the ones that can erase.
        self.memories: list[Any] = list(memory)
        # LLM handlers whose providers hold conversation state: erasing a run deletes it there too
        self.state_holders: list[Any] = []
        self._tenants: dict[str, str | None] = {}  # run id -> tenant, for runs not hosted here
        self._retention_seen: dict[int, float] = {}
        self._runs: dict[str, _Run] = {}
        self._algos: dict[str, Algorithm] = {}
        self._registered: set[int] = set()
        self._adopt_locks: dict[str, asyncio.Lock] = {}
        self._tasks: set[asyncio.Task[Any]] = set()
        self._sweeper: asyncio.Task[None] | None = None
        self.last_recovery: Any = None
        self.crashed = False  # set by abort(): models a process death
        self.subscribe(RUN_CANCEL, self._on_cancel, group="aaa.engine.cancel")

    async def __aenter__(self) -> Self:
        await self.bus.__aenter__()
        if self.sweep_interval_s:
            self._sweeper = asyncio.create_task(self._sweep_loop())
        return self

    async def __aexit__(self, *exc: object) -> None:
        if self._sweeper:
            self._sweeper.cancel()
        await self.bus.__aexit__(*exc)
        for run in self._runs.values():
            if run.timer:
                run.timer.cancel()
        for t in list(self._tasks):
            t.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)

    async def abort(self) -> None:
        """Hard stop without draining or completing anything — what a crash looks like to the
        log. Runs stay open in the store; another process (or `recover()`) continues them.
        Sets `crashed` so cleanup code (lease release, ...) behaves as if the process vanished."""
        self.crashed = True
        if self._sweeper:
            self._sweeper.cancel()
        await self.bus.abort()  # first: nothing may be acknowledged any more
        for run in self._runs.values():
            if run.timer:
                run.timer.cancel()
        for t in list(self._tasks):
            t.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)

    def spawn(self, coro: Coroutine[Any, Any, Any]) -> asyncio.Task[Any]:
        """Run background work (e.g. a long tool call) tracked for clean shutdown."""
        t = asyncio.create_task(coro)
        self._tasks.add(t)
        t.add_done_callback(self._tasks.discard)
        return t

    async def _tenant_of(self, run_id: str) -> str | None:
        run = self._runs.get(run_id)
        if run is not None:
            return run.tenant
        if run_id in self._tenants:
            return self._tenants[run_id]
        first = await self.retry(lambda: self.store.read(run_id, 0, 1), "read")
        if not first:
            return None  # not started (yet): do not cache a miss
        self._tenants[run_id] = first[0].tenant
        return first[0].tenant

    def _forget(self, run_id: str) -> None:
        run = self._runs.pop(run_id, None)
        if run and run.timer:
            run.timer.cancel()
        self._tenants.pop(run_id, None)

    # -- tenancy, erasure, retention (see aaa.runtime.tenancy / aaa.runtime.retention) -------
    def tenant(self, name: str) -> TenantScope:
        """A view of this engine bound to one tenant: namespaced run ids, ownership-checked lookups."""
        return TenantScope(self, name)

    async def erase(
        self, run_id: str, *, delete: bool = False, reason: str = "erasure", force: bool = False
    ) -> ErasureReport:
        """Erase a finished run: scrub its payloads (keeping the audit skeleton) or, with
        `delete=True`, remove it. Also removes its snapshots, blobs and bus copies."""
        from aaa.runtime.retention import erase_run

        return await erase_run(self, run_id, delete=delete, reason=reason, force=force)

    async def purge(
        self,
        *,
        older_than: timedelta,
        tenant: str | None = None,
        delete: bool = False,
        reason: str = "retention",
    ) -> PurgeReport:
        """Erase every finished run (of `tenant`, if given) that completed more than `older_than` ago."""
        from aaa.runtime.retention import purge

        return await purge(self, older_than=older_than, tenant=tenant, delete=delete, reason=reason)

    async def forget_memory(self, scope: Any) -> MemoryReport:
        """Erase everything the registered memory backends hold for `scope` (an `aaa.memory.Scope`:
        a tenant, or one user of it). Backends without a `forget` capability are reported, not skipped silently."""
        from aaa.memory.base import Forgettable

        report = MemoryReport()
        for m in self.memories:
            name = type(m).__name__
            if isinstance(m, Forgettable):
                report.forgotten[name] = report.forgotten.get(name, 0) + await m.forget(scope)
            else:
                report.unsupported.append(name)
        return report

    async def apply_retention(self) -> PurgeReport:
        """Evaluate the configured `retention` policies now (each is rate-limited by its `interval_s`)."""
        from aaa.runtime.retention import apply_retention

        return await apply_retention(self, self.retention)

    def handle(self, run_id: str) -> RunHandle:
        """A handle to any run by id (started here, in another process, or before a restart)."""
        return RunHandle(self, run_id)

    def _trace(self, run_id: str) -> str:
        run = self._runs.get(run_id)
        return run.trace_id if run else run_id

    async def retry(
        self,
        fn: Callable[[], Awaitable[Any]],
        what: str = "store operation",
        policy: RetryPolicy | None = None,
    ) -> Any:
        """Run a store/blob operation, riding out transient outages per `store_retry` (or `policy`).
        Raises `StoreUnavailable` if the outage outlasts the budget; other errors pass through."""
        return await retrying(fn, policy or self.store_retry, classify=self.is_transient, what=what)

    async def project(self, projection: Projection[S], run_id: str) -> S:
        loaded = await self.retry(
            lambda: projection.load(self.store, run_id, hydrate=self.hydrate), "projection load"
        )
        return loaded.state

    async def _started_of(self, run_id: str) -> Event:
        first = await self.retry(lambda: self.store.read(run_id, 0, 1), "read")
        if not first or first[0].type != RUN_STARTED:
            raise KeyError(f"no run {run_id!r}")
        return await self.hydrate(first[0])

    # -- blob claim-check -------------------------------------------------------------------
    async def _claim(self, event: Event) -> Event:
        if self.blobs is None or event.type == RUN_COMPLETED:
            return event
        raw = json.dumps(event.data, default=str).encode()
        if len(raw) <= self.claim_threshold:
            return event
        # The payload is namespaced by its stream so identical payloads of two runs (or tenants) are
        # different blobs: erasing one run can never remove, or reveal the existence of, another's.
        raw = json.dumps({"$ns": event.stream, "data": event.data}, default=str).encode()
        ref = await self.blobs.put(raw, "application/json")
        stub = {CLAIM_KEY: True, **{k: event.data[k] for k in CLAIM_KEEP if k in event.data}}
        return event.model_copy(update={"data": stub, "blobs": (*event.blobs, ref)})

    async def hydrate(self, event: Event) -> Event:
        """Restore a claim-checked payload (no-op for ordinary events)."""
        if self.blobs is None or event.data.get(CLAIM_KEY) is not True or not event.blobs:
            return event
        blobs = self.blobs
        data = json.loads(await self.retry(lambda: blobs.get(event.blobs[-1]), "blob read"))
        if isinstance(data, dict) and data.keys() == {"$ns", "data"}:
            data = data["data"]
        return event.model_copy(update={"data": data})

    async def media_bytes(self, part: Any) -> bytes:
        """The bytes of an attachment (inline or in the blob store)."""
        return await _media.read(part, self.blobs)

    # -- wiring -----------------------------------------------------------------------------
    def subscribe(
        self, pattern: str, handler: Handler, *, group: str | None = None, **kw: Any
    ) -> None:
        """Subscribe a handler whose returned events are persisted+published by the engine.
        Handlers receive hydrated events. Handler exceptions fail the run (as an event), never
        the bus. A handler that fails because the STORE is unavailable (beyond `store_retry`) is not
        a failed run: the engine raises `Redeliver` and the bus delivers the event again later.
        Events for runs that already finished are not delivered to handlers.

        `group` names a logical consumer. With a distributed bus, processes subscribing with the
        same group compete (each event is handled by exactly one of them, in order per run);
        `group=None` means broadcast to every process. Ignored by the in-process bus."""

        async def wrapped(e: Event) -> None:
            if e.run_id and self.is_done(e.run_id) and not e.type.startswith("aaa.run."):
                return
            try:
                e = await self.hydrate(e)
                for f in await handler(e) or ():
                    await self.emit(f)
            except Redeliver:
                raise
            except Exception as exc:
                if self.is_transient(exc):
                    # The STORE is unavailable (outlasting local retries); the handler is not at fault.
                    # Never fail the run for that: ask the bus to deliver this event again later.
                    log.warning(
                        "store unavailable while handling %s (%s: %s); asking the bus to redeliver",
                        e.type,
                        type(exc).__name__,
                        exc,
                    )
                    raise Redeliver(f"{type(exc).__name__}: {exc}") from exc
                log.exception("handler failed for %s", e.type)
                if e.run_id:
                    try:
                        await self.complete(
                            e.run_id, OutcomeKind.FAILED, reason=f"{type(exc).__name__}: {exc}"
                        )
                    except Exception as exc2:
                        if self.is_transient(exc2):
                            raise Redeliver(f"{type(exc2).__name__}: {exc2}") from exc2
                        log.exception("could not record the failure of run %s", e.run_id)

        if self.distributed:
            kw["group"] = group
        self.bus.subscribe(pattern, wrapped, **kw)

    def register(self, algo: Algorithm) -> None:
        """Subscribe an algorithm. It only ever sees events of runs it was started for, so
        several algorithms (and nested child runs) coexist on one bus. With a distributed bus,
        algorithm names must be unique and identical across processes."""
        if id(algo) in self._registered:
            return
        self._registered.add(id(algo))
        self._algos.setdefault(algo.name, algo)

        async def h(e: Event) -> list[Event]:
            rid = e.run_id or ""
            run = self._runs.get(rid) or await self._adopt(rid, algo)
            if run is None or run.algo is not algo:
                return []
            await self._sync(rid, run)  # absorb events other processes appended
            if run.done and not e.type.startswith("aaa.run."):
                return []
            return await algo.on_event(e, Ctx(self, rid))

        # ONE subscription (one queue, one worker): the algorithm sees its events strictly in
        # log order, so state folded from history is always consistent with what it handles.
        patterns = algo.subscriptions()
        self.subscribe(
            "*",
            h,
            group=f"aaa.algo.{algo.name}",
            predicate=lambda e: any(fnmatch.fnmatchcase(e.type, p) for p in patterns),
        )

    # -- run state: a cache of a fold of the log --------------------------------------------
    def is_done(self, run_id: str) -> bool:
        run = self._runs.get(run_id)
        return bool(run and run.done)

    async def _adopt(self, rid: str, algo: Algorithm | None) -> _Run | None:
        """Build local run state from the log (snapshot + tail). Returns None if `rid` is not a
        run, or belongs to a different algorithm than `algo`."""
        if run := self._runs.get(rid):
            return run
        async with self._adopt_locks.setdefault(rid, asyncio.Lock()):
            if run := self._runs.get(rid):
                return run
            first = await self.retry(lambda: self.store.read(rid, 0, 1), "read") if rid else []
            if not first or first[0].type != RUN_STARTED:
                return None
            started = await self.hydrate(first[0])
            if algo is not None and started.data.get("algo") != algo.name:
                return None
            loaded = await self.retry(
                lambda: RUNMETA.load(self.store, rid, hydrate=self.hydrate), "projection load"
            )
            run = _Run(
                Budget(**started.data.get("budget", {})),
                asyncio.get_running_loop().create_future(),
                trace_id=started.trace_id or rid,
                algo=algo,
                parent=started.data.get("parent"),
                tag=started.data.get("run_tag"),
                tenant=started.tenant,
                meta=loaded.state,
                synced=loaded.seq,
            )
            if run.meta.completed is not None:
                self._mark_done(run, run.meta.completed)
            elif run.budget.deadline_s is not None:
                elapsed = time.time() - started.time.timestamp()
                run.timer = asyncio.create_task(
                    self._deadline(
                        rid, max(0.0, run.budget.deadline_s - elapsed), run.budget.deadline_s
                    )
                )
            self._runs[rid] = run
            return run

    @staticmethod
    def _mark_done(run: _Run, outcome: dict[str, Any]) -> None:
        run.done = True
        if run.timer:
            run.timer.cancel()
        if not run.future.done():
            run.future.set_result(Outcome(**outcome))

    async def _absorb(self, rid: str, run: _Run, stored: Event | None = None) -> None:
        """Fold events appended since `run.synced` into the run's meta (cheap if nothing new)."""
        async with run.lock:
            if stored is not None and stored.seq == run.synced + 1:
                _apply_meta(run.meta, stored)
                run.synced = stored.seq
            elif await self.retry(lambda: self.store.last_seq(rid), "read") > run.synced:
                for e in await self.retry(lambda: self.store.read(rid, run.synced + 1), "read"):
                    _apply_meta(run.meta, e)
                    run.synced = e.seq
            if run.meta.completed is not None and not run.future.done():
                self._mark_done(run, run.meta.completed)

    async def _sync(self, rid: str, run: _Run) -> None:
        await self._absorb(rid, run)
        await self._check_budget(rid, run)

    async def _check_budget(self, rid: str, run: _Run) -> None:
        if run.done or run.tripped or not (why := run.ledger.exceeded(run.budget)):
            return
        run.tripped = True
        try:
            await self.emit(
                Event(
                    id=f"{rid}:budget",
                    type="aaa.budget.exceeded",
                    stream=rid,
                    run_id=rid,
                    trace_id=run.trace_id,
                    data={"reason": why},
                )
            )
            await self.complete(rid, OutcomeKind.BUDGET_EXCEEDED, reason=why)
        except BaseException:
            run.tripped = False  # not recorded (store outage): the next sync must try again
            raise

    # -- emit -------------------------------------------------------------------------------
    async def emit(self, event: Event, *, retry: RetryPolicy | None = None) -> Event | None:
        """Validate -> claim-check -> append (assign seq) -> account budget -> deliver.
        Returns the stored event, or None if dropped (duplicate terminal event).

        A store outage is ridden out per `retry` (default `store_retry`); if it outlasts that budget
        `StoreUnavailable` is raised and NOTHING has been recorded or marked done locally."""
        rid = event.run_id or ""
        run = self._runs.get(rid)
        marked_done = False
        if event.type == RUN_COMPLETED and run is not None:
            if run.done:
                return None
            run.done = marked_done = True  # no await between check and set: atomic under asyncio
        try:
            tenant = await self._tenant_of(rid) if event.tenant is None and rid else None
            if tenant is not None:
                event = event.model_copy(update={"tenant": tenant})
            event = self.schemas.validate(event)
            event = await _media.offload(
                event, self.blobs, self.media_threshold
            )  # attachment bytes -> blob store, before anything scrubs or measures the event
            if self.redactor is not None:
                event = self.redactor.redact(event)
            claimed = await self._claim(event)
            attempts = 0

            async def append() -> Event:
                nonlocal attempts
                attempts += 1
                return (await self.store.append(claimed.stream, [claimed]))[0]

            stored = await self.retry(append, "append", retry)
        except BaseException:
            if marked_done and run is not None:
                run.done = False  # the completion never reached the log: stay open, retryable
            raise
        # A duplicate normally means "already appended AND delivered". After a retry it may mean our
        # first attempt committed but we never saw the answer, so deliver anyway (repeats are harmless).
        if stored.dup and attempts == 1:
            if stored.type == RUN_COMPLETED:
                await self._after_completed(rid, stored)
            return stored
        if run is not None:
            await self._absorb(rid, run, stored)
        await self.bus.publish(stored)
        if stored.type == RUN_COMPLETED:
            await self._after_completed(rid, stored)
        elif run is not None:
            await self._check_budget(rid, run)
        return stored

    async def _after_completed(self, rid: str, done: Event) -> None:
        """Idempotent consequences of a run finishing, computed from the log alone: settle local
        state, cancel still-open children, announce the result to the parent."""
        run = self._runs.get(rid)
        if run is not None:
            self._mark_done(run, done.data)
        meta = await self.project(RUNMETA, rid)
        for cid in list(meta.children):  # a finished parent leaves no orphan work
            await self.complete(cid, OutcomeKind.CANCELLED, reason="parent run finished")
        try:
            started = await self._started_of(rid)
        except KeyError:
            return
        if parent := started.data.get("parent"):
            await self.emit(
                self._child_completed(
                    parent, done, started.trace_id or rid, started.data.get("run_tag"), meta.ledger
                )
            )

    @staticmethod
    def _child_completed(
        parent_id: str, done: Event, trace_id: str, tag: str | None, ledger: Ledger
    ) -> Event:
        return Event(
            id=f"child-{done.run_id}",
            type=CHILD_COMPLETED,
            stream=parent_id,
            run_id=parent_id,
            trace_id=trace_id,
            source="engine",
            data={
                "child_run_id": done.run_id,
                "tag": tag,
                "outcome": done.data,
                "ledger": ledger.model_dump(),
            },
        )

    async def complete(
        self, run_id: str, kind: OutcomeKind, value: Any = None, reason: str | None = None
    ) -> None:
        await self.emit(
            Event(
                id=f"{run_id}:completed",
                type=RUN_COMPLETED,
                stream=run_id,
                run_id=run_id,
                trace_id=self._trace(run_id),
                data={"kind": kind.value, "value": value, "reason": reason},
            )
        )

    async def _on_cancel(self, e: Event) -> None:
        if e.run_id:
            await self.complete(e.run_id, OutcomeKind.CANCELLED, reason=e.data.get("reason"))

    async def _deadline(self, run_id: str, seconds: float, total: float | None = None) -> None:
        await asyncio.sleep(seconds)
        try:
            await self.emit(
                Event(
                    type="aaa.timer.fired",
                    stream=run_id,
                    run_id=run_id,
                    trace_id=self._trace(run_id),
                    data={"kind": "deadline"},
                )
            )
            await self.complete(
                run_id, OutcomeKind.TIMEOUT, reason=f"deadline {total or seconds}s exceeded"
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            # e.g. the store is down past the retry budget. `sweep()` enforces deadlines too.
            log.exception("could not enforce the deadline of run %s; sweep() will retry", run_id)

    async def await_event(
        self, stream: str, match: Callable[[Event], bool], *, from_seq: int = 0
    ) -> Event:
        """Wait for the first event on `stream` (from `from_seq` on) that `match` accepts. A dropped store
        connection mid-read is ridden out like any other store call: resubscribe from the last event
        seen, backing off per `store_retry` (the budget restarts on progress)."""
        pos, attempt, started = from_seq, 0, time.monotonic()
        while True:
            try:
                async for e in self.store.subscribe(stream, pos):
                    pos, attempt, started = e.seq + 1, 0, time.monotonic()
                    if match(e):
                        return e
                raise AssertionError("unreachable")
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if not self.is_transient(exc):
                    raise
                attempt += 1
                delay, policy = self.store_retry.delay(attempt), self.store_retry
                spent = time.monotonic() - started
                if (policy.max_attempts is not None and attempt >= policy.max_attempts) or (
                    policy.max_elapsed_s is not None and spent + delay > policy.max_elapsed_s
                ):
                    raise StoreUnavailable(
                        f"waiting on {stream} still failing after {attempt} attempt(s): "
                        f"{type(exc).__name__}: {exc}"
                    ) from exc
                log.warning(
                    "subscribe to %s failed (%s); resubscribing in %.2fs", stream, exc, delay
                )
                await asyncio.sleep(delay)

    async def _await_completed(self, run_id: str) -> Outcome[Any]:
        done = await self.await_event(run_id, lambda e: e.type == RUN_COMPLETED)
        return Outcome(**done.data)

    # -- runs -------------------------------------------------------------------------------
    async def start(
        self,
        algo: Algorithm,
        input: dict[str, Any],
        *,
        budget: Budget | None = None,
        run_id: str | None = None,
        parent: str | None = None,
        tag: str | None = None,
        tenant: str | None = None,
        meta: dict[str, str | int | float | bool] | None = None,
    ) -> RunHandle:
        """Start a run (idempotent for a given `run_id`). With `parent`, the child's spend rolls
        up into the parent's budget, its completion is announced on the parent stream, and it is
        cancelled if the parent finishes first. `tenant` tags the run and everything derived from it
        (children inherit their parent's); see `Engine.tenant()` for the safe, namespaced way. `meta`
        is trusted application data (e.g. `{"user": "ann"}`) stamped on the run's events and inherited
        by everything derived from them (a child run with no `meta` of its own gets its parent's); memory scopes are built from it, never from model output."""
        self.register(algo)
        if tenant is not None:
            valid_tenant(tenant)
        rid = run_id or new_id()
        if rid in self._runs:
            return RunHandle(self, rid)
        if (
            await self.retry(lambda: self.store.last_seq(rid), "read") >= 0
        ):  # exists: finished, or open elsewhere
            await self._adopt(rid, None)
            return RunHandle(self, rid)
        parent_run = self._runs.get(parent) if parent else None
        if tenant is None and parent:
            tenant = parent_run.tenant if parent_run else await self._tenant_of(parent)
        if (
            meta is None and parent
        ):  # a child belongs to the same user (and agent) as its parent: memory scopes are built from this
            first = await self.retry(lambda: self.store.read(parent, 0, 1), "read")
            if first and first[0].meta:
                meta = dict(first[0].meta)
        if tenant is None and self.require_tenant:
            raise ValueError("this engine requires a tenant: use engine.tenant(name).start(...)")
        run = _Run(
            budget or self.default_budget,
            asyncio.get_running_loop().create_future(),
            trace_id=parent_run.trace_id if parent_run else rid,
            algo=algo,
            parent=parent,
            tag=tag,
            tenant=tenant,
        )
        self._runs[rid] = run
        try:
            if parent:
                await self.emit(
                    Event(
                        id=f"{rid}:childstarted",
                        type=CHILD_STARTED,
                        stream=parent,
                        run_id=parent,
                        tenant=tenant,
                        trace_id=run.trace_id,
                        source="engine",
                        data={"child_run_id": rid, "tag": tag},
                    )
                )
            if run.budget.deadline_s is not None:
                run.timer = asyncio.create_task(self._deadline(rid, run.budget.deadline_s))
            await self.emit(
                Event(
                    id=f"{rid}:started",
                    type=RUN_STARTED,
                    stream=rid,
                    run_id=rid,
                    tenant=tenant,
                    meta=meta or {},
                    trace_id=run.trace_id,
                    source=algo.name,
                    data={
                        "algo": algo.name,
                        "input": to_jsonable_python(
                            input, serialize_unknown=True
                        ),  # models (e.g. attachments) become their JSON form, as the log will hold them
                        "parent": parent,
                        "run_tag": tag,
                        "budget": run.budget.model_dump(),
                    },
                )
            )
        except BaseException:
            # Nothing was durably started: forget it so a retry (same or new run_id) starts cleanly.
            self._runs.pop(rid, None)
            if run.timer:
                run.timer.cancel()
            raise
        return RunHandle(self, rid)

    async def recover(self, algos: list[Algorithm]) -> dict[str, RunHandle]:
        """Continue every run left open in the store (single-process restart). See recovery.py."""
        from aaa.runtime.recovery import recover

        return await recover(self, algos)

    async def sweep(self) -> Any:
        """Stateless, idempotent cluster maintenance: expire deadlines of dead processes, cancel
        orphaned children, repair lost child announcements. Safe to run anywhere, any time."""
        from aaa.runtime.recovery import maintain

        return await maintain(self)

    async def _sweep_loop(self) -> None:
        while True:
            await asyncio.sleep(self.sweep_interval_s or 5)
            raise_if_cancelling()
            try:
                await self.sweep()
            except Exception:
                log.exception("sweep failed")

    async def run(self, algo: Algorithm, input: dict[str, Any], **kw: Any) -> Outcome[Any]:
        """Facade: start and await the outcome (no event knowledge needed)."""
        return await (await self.start(algo, input, **kw)).result()
