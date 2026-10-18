# Computer use, code as action, anytime

Three algorithms for acting beyond text: operate an **environment** (a web page, an emulator), act by **writing code**, and **escalate** from a cheap approach to a costly one only while the budget allows. The first two run things written by a model or by someone else's content, so their controls are enforced in code (see the [algorithms threat model](../../security/algorithms.md), T1, T4, T5). Like every [algorithm](custom.md) they are folds of the run's log, bounded at construction, and replay to the same outcome.

```python
--8<-- "examples/environment_algorithms.py"
```

```text
computer: 20 euros a month | actions 1 | approvals asked 1
codeact: The sum of the first ten squares is 385. | it ran: 385 | approvals asked 1
anytime: 42 | tried ['quick', 'careful'] | stopped: accepted
```

## Environments and ComputerUse

An `aaa.env.Environment` has `actions` (each an `ActionSpec` with a description, an argument schema and `side_effect`), and `reset()`, `observe()`, `act(action, arguments)` and `aclose()`, each returning an `Observation` (text, url, title, and a `fingerprint` that changes when the state does). Four rules hold for every environment:

- **What it shows is data.** The observation is untrusted; `ComputerUse` quotes it (`<screen>`) and says so in the prompt.
- **Actions that change something wait for approval.** `env_tools(env)` gives `env_act` (only for actions with `side_effect=False`) and `env_act_confirm` (only for the others, **requires approval**); each refuses the other kind, so a side effect cannot skip the gate. Arguments are validated against the action's schema.
- **An environment can refuse an action outright** (`ActionRefused`), whatever the model says and whoever approves.
- **One environment is one screen**: use it from one run at a time.

```python
agent = ComputerUse(env, max_steps=12, max_repeats=3, stall_limit=3)
ToolHandler(agent.tools).attach(engine)        # the environment's tools live in agent.tools
```

Each turn the model sees the goal, its recent actions, the screen and the action menu, and replies with one action (`{"action", "arguments"}`) or `finish`. The loop also ends, with `NO_SOLUTION` and what was done, when the same action would be repeated `max_repeats` times in a row (`stopped: "repeat"`), when the screen has not changed for `stall_limit` actions (`"stall"`), or after `max_steps` actions (`"max_steps"`). A denied or refused action is reported to the model as not having happened. Bound the wall-clock time with the run's `Budget(deadline_s=...)`. Result `{"answer", "steps", "stopped"}`. `FakeEnvironment` is a deterministic page-to-page state machine for tests and offline demos.

### The browser

`pip install "autonomous-agent-architecture[browser]"` and `playwright install chromium`, then:

```python
env = BrowserEnvironment(allowed_hosts=["docs.example.com", "*.example.com"], start_url="https://docs.example.com/")
```

- **Hosts.** `allowed_hosts` is required. Every request the page makes (the page, every redirect hop, scripts, images, frames) is checked **before it is sent** and aborted if its host is not listed; only `http`/`https` (and `about:blank`) pass. A redirect is never followed inside the browser: a redirected navigation is answered with a small page that moves the browser to the checked target, and a redirected subresource is followed hop by hop with each hop checked. Unless `allow_private_networks`, a host resolving to a private or loopback address is refused too. (A DNS-rebinding gap remains between the check and the browser's own lookup: do not allowlist hosts you do not control.)
- **A clean browser.** A fresh context: no cookies or stored logins, downloads off, service workers blocked, extra windows closed, dialogs dismissed.
- **Actions.** `goto`, `scroll`, `back` and `read` have no side effects; `click` and `type` do. Controls are numbered in the observation (`[3] a "Pricing" -> ...`) and addressed by number, not by selector.
- **Credential fields are refused.** `type` into a password input, or any field whose name, id, label, placeholder or autocomplete suggests a password, token, code or card number, raises `ActionRefused`. This is not something an approver can override.
- The observation is the page's text and its controls. A page can still say misleading things; the allow-list only limits where the browser can go.

## CodeAct

`CodeAct(sandbox, max_steps=6, requires_approval=True)`: the model replies with one ```` ```python ```` block, the code runs in the sandbox, what it printed comes back (quoted, cut to `observation_chars`), and the loop repeats until the model answers in plain text. Compared with a JSON tool call, one step can compute anything: loops, parsing, arithmetic.

- **A sandbox is required** (anything that is not a `Sandbox` is a `TypeError`); there is no in-process fallback. `SubprocessSandbox` is not isolation and refuses to exist without `unsafe=True`; use `DockerSandbox` for anything but development.
- **Every run waits for approval** (the `run_python` tool is `code_exec_tool(..., requires_approval=True)`); nothing runs while it waits. Pass `requires_approval=False` only for a sandbox you would trust with the worst code a model could write.
- Time, memory and output limits come from the sandbox `Limits`; code is capped at `max_code_chars`; runs at `max_steps`. Running out ends `NO_SOLUTION`.
- The code gets **nothing**: no network, files, tools or memory between runs. A bridge for calling your tools from inside the code is not part of this release; put what the code needs in the task.

Result `{"answer", "steps": [{"code", "exit_code", "stdout", "stderr"}], "turns"}`.

## Anytime

```python
Anytime([Rung("quick", fast_algo, budget=Budget(max_steps=5)),
         Rung("careful", slow_algo, budget=Budget(max_steps=80), needs_steps=40)],
        accept=lambda value: value["confident"])
```

Rungs run cheapest first, each as a child run under its own `Budget`. A rung's successful result is accepted when `accept(value)` is true (default: always); otherwise, or if the rung fails, the next rung starts **only if the run's remaining budget can afford it** (`needs_steps` left, and tokens, cost and time not already spent). When it cannot, the run ends with the best result so far instead of starting something it cannot finish. The remaining budget is computed from the run's own log, so the decision is the same on replay. Result `{"value", "rung", "tried", "degraded", "stopped"}` with `stopped` `accepted`, `exhausted` or `budget`; `degraded` is true unless the chosen rung was the first. A ladder has 2 to 6 rungs.

## Choosing

| You want | Use |
|---|---|
| To operate a website or other interface | `ComputerUse` with a `BrowserEnvironment` (or your own `Environment`) |
| Exact computation, parsing, data work | `CodeAct` (with a real sandbox) |
| Good answers cheaply, better ones when it matters and the budget allows | `Anytime` |
