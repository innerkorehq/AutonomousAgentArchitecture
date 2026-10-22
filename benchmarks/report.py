"""Turn benchmarks/results/**/*.json into markdown tables.

uv run python -m benchmarks.report > table.md
"""

from __future__ import annotations

import json
from pathlib import Path

RESULTS = Path(__file__).parent / "results"
TITLES = {
    "math": "Arithmetic word problems",
    "tools": "Tool use over a lookup table",
    "rag": "Retrieval-grounded questions",
}


def render() -> str:
    out: list[str] = []
    for model_dir in sorted(p for p in RESULTS.iterdir() if p.is_dir()):
        out.append(f"## {model_dir.name}\n")
        for fam in ("math", "tools", "rag"):
            rows = (
                [json.loads(p.read_text()) for p in sorted((model_dir / fam).glob("*.json"))]
                if (model_dir / fam).is_dir()
                else []
            )
            if not rows:
                continue
            rows.sort(key=lambda r: (-r["accuracy"], r["tokens"]))
            out += [
                f"### {TITLES[fam]} ({rows[0]['runs']} runs per algorithm)\n",
                "| Algorithm | Accuracy | Model calls | Tokens | Latency (s) | Accuracy per 1k tokens | What it does |",
                "|---|---:|---:|---:|---:|---:|---|",
            ]
            for r in rows:
                out.append(
                    f"| `{r['algorithm']}` | {r['accuracy']:.0%} | {r['calls']:.1f} | {r['tokens']:.0f} | {r['latency_s']:.1f} | {r['acc_per_1k_tokens']:.2f} | {r['about']} |"
                )
            out.append("")
    return "\n".join(out)


if __name__ == "__main__":
    print(render())
