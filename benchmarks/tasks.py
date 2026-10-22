"""Small, seeded task sets with answers that can be checked by code. Everything is invented or computed, so a model cannot have memorised it."""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Any

from aaa.data import Doc, InMemoryIndex
from aaa.eval import Dataset

ANSWER_NOTE = " Think it through briefly, then finish with a line 'Answer: <integer>'."


# ================================================================ math word problems
def math_cases(n: int = 18, seed: int = 7) -> Dataset:
    """Multi-step integer word problems from six templates. The answer is computed, never typed."""
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        r = rng
        kind = i % 6
        if kind == 0:
            a, b, c, d = (
                r.randint(12, 29),
                r.randint(14, 38),
                r.randint(60, 140),
                r.randint(80, 190),
            )
            text = f"A bakery bakes {a} trays with {b} rolls on each tray. It sells {c} rolls in the morning and {d} in the afternoon. How many rolls are left?"
            truth = a * b - c - d
        elif kind == 1:
            s1, t1, s2, t2 = r.randint(45, 95), r.randint(3, 9), r.randint(60, 120), r.randint(2, 7)
            text = f"A train travels at {s1} km/h for {t1} hours, then at {s2} km/h for {t2} hours. How many kilometres does it cover in total?"
            truth = s1 * t1 + s2 * t2
        elif kind == 2:
            k, c = r.randint(14, 45), r.randint(12, 60)
            text = f"Tom has {k * 5} marbles. He gives away a fifth of them, then buys {c} more. How many marbles does he have now?"
            truth = k * 5 - k + c
        elif kind == 3:
            length, width = r.randint(18, 45), r.randint(7, 26)
            text = f"A rectangle has a perimeter of {2 * (length + width)} metres and a length of {length} metres. What is its area in square metres?"
            truth = length * width
        elif kind == 4:
            boys, girls, cb, cg = (
                r.randint(9, 17),
                r.randint(10, 19),
                r.randint(3, 8),
                r.randint(4, 9),
            )
            text = f"A class has {boys + girls} students, {boys} of them boys. Each boy brings {cb} cookies and each girl brings {cg}. How many cookies are there in total?"
            truth = boys * cb + girls * cg
        else:
            price, group, free = r.randint(13, 48), r.randint(24, 60), r.randint(3, 9)
            text = f"Tickets cost {price} dollars each. A group of {group} people gets {free} tickets free. How many dollars does the group pay?"
            truth = (group - free) * price
        rows.append({"id": f"math-{i + 1:02d}", "input": text + ANSWER_NOTE, "expected": truth})
    return Dataset.from_list(rows, "math")


_NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def last_number(text: str) -> float | None:
    """The number after the last 'Answer:' if there is one, else the last number in the text."""
    tail = text.rsplit("Answer:", 1)[-1] if "Answer:" in text else text
    found = _NUM.findall(tail)
    if not found:
        return None
    try:
        return float(found[0 if "Answer:" in text else -1].replace(",", ""))
    except ValueError:
        return None


# ================================================================ tools over an invented knowledge base
CITIES = {
    "Veldor": (48_200, 120),
    "Marnis": (73_500, 150),
    "Quillon": (12_900, 43),
    "Brasko": (91_000, 130),
    "Tandrel": (35_400, 59),
    "Orvane": (56_700, 63),
    "Lushar": (27_300, 91),
    "Pellam": (64_800, 72),
    "Dunwick": (19_500, 65),
    "Karsten": (82_100, 41),
}


def kb_lookup(query: str) -> str:
    """Facts about the invented cities: ask for 'population of <city>' or 'area of <city> in km2'."""
    q = query.lower()
    for name, (pop, area) in CITIES.items():
        if name.lower() in q:
            if "area" in q or "km" in q or "size" in q:
                return str(area)
            if "popul" in q or "people" in q or "inhabit" in q:
                return str(pop)
    return "no data"


def tool_cases(n: int = 15, seed: int = 11) -> Dataset:
    rng = random.Random(seed)
    names = list(CITIES)
    rows = []
    for i in range(n):
        kind = i % 5
        a, b, c = rng.sample(names, 3)
        pa, ar_a = CITIES[a]
        pb, _ = CITIES[b]
        pc, _ = CITIES[c]
        if kind == 0:
            q, truth = f"What is the combined population of {a} and {b}?", pa + pb
        elif kind == 1:
            hi, lo = (a, b) if pa >= pb else (b, a)
            q, truth = f"How many more people live in {hi} than in {lo}?", abs(pa - pb)
        elif kind == 2:
            q, truth = f"What is the total population of {a}, {b} and {c}?", pa + pb + pc
        elif kind == 3:
            q, truth = f"The area of {a} in square kilometres times 10 is how much?", ar_a * 10
        else:
            q, truth = (
                f"What is the combined area of {a} and {b} in square kilometres?",
                ar_a + CITIES[b][1],
            )
        rows.append(
            {
                "id": f"tool-{i + 1:02d}",
                "input": q + " Use the tools, and finish with a line 'Answer: <integer>'.",
                "expected": truth,
            }
        )
    return Dataset.from_list(rows, "tools")


# ================================================================ retrieval over invented documents
DOCS = [
    ("d1", "Zyrlo Corp was founded in 1987 in the city of Tarmouth by Ilse Vandermeer."),
    (
        "d2",
        "Zyrlo Corp's headquarters moved to Brenwick in 2004. Its main product is the Halcyon routing engine.",
    ),
    (
        "d3",
        "The Halcyon routing engine was first released in 1999 and is written in the Ostrel language.",
    ),
    (
        "d4",
        "Ilse Vandermeer stepped down as chief executive in 2011. Her successor was Joaquim Ferreyra.",
    ),
    (
        "d5",
        "Joaquim Ferreyra holds the record for the longest tenure at Zyrlo Corp: fourteen years.",
    ),
    ("d6", "Brenwick is a port city known for its annual lantern festival held every October."),
    (
        "d7",
        "The Ostrel language was designed by Pavel Noreski and compiles to a bytecode called OVM.",
    ),
    ("d8", "Tarmouth is an inland city with a population of about 61,000 people."),
    ("d9", "Zyrlo Corp employs 2,300 people, of whom 410 work in the research lab in Tarmouth."),
    (
        "d10",
        "The Halcyon routing engine supports three protocols: BGP, OSPF and the proprietary Zyr-Link.",
    ),
    ("d11", "Zyr-Link was introduced in version 4 of Halcyon and replaced the older ZLP protocol."),
    ("d12", "The research lab in Tarmouth is led by Dr. Mirela Obuya, who joined in 2016."),
]
RAG_QA = [
    ("Who founded Zyrlo Corp?", "Vandermeer"),
    ("In which year was Zyrlo Corp founded?", "1987"),
    ("Where are Zyrlo Corp's headquarters?", "Brenwick"),
    ("What is Zyrlo Corp's main product?", "Halcyon"),
    ("In which language is the Halcyon routing engine written?", "Ostrel"),
    ("Who succeeded Ilse Vandermeer as chief executive?", "Ferreyra"),
    ("How many people work in the research lab in Tarmouth?", "410"),
    ("Which protocol replaced ZLP?", "Zyr-Link"),
    ("Who designed the language that Halcyon is written in?", "Noreski"),
    ("Who leads the research lab, and in which city is it?", "Obuya"),
    ("How long did the chief executive who succeeded the founder serve?", "fourteen"),
    ("What bytecode does the language of the Halcyon engine compile to?", "OVM"),
]
RAG_UNANSWERABLE = [
    "What is Zyrlo Corp's annual revenue?",
    "Who is Zyrlo Corp's chief financial officer?",
    "What was Zyrlo Corp's share price on its first trading day?",
    "In which year did Brenwick's lantern festival start?",
]
REFUSALS = (
    "not mention", "no information", "not provided", "cannot", "can't", "unable", "not enough", "insufficient", "does not say", "doesn't say",
    "not available", "no data", "don't know", "do not know", "not specified", "not stated", "no mention", "not found", "unknown", "not contain", "no relevant",
)  # fmt: skip


def rag_cases() -> Dataset:
    rows = [
        {"id": f"rag-{i + 1:02d}", "input": q, "expected": a} for i, (q, a) in enumerate(RAG_QA)
    ]
    rows += [
        {"id": f"rag-u{i + 1}", "input": q, "expected": "REFUSE"}
        for i, q in enumerate(RAG_UNANSWERABLE)
    ]
    return Dataset.from_list(rows, "rag")


async def rag_index() -> InMemoryIndex:
    idx = InMemoryIndex()
    await idx.upsert([Doc(id=i, text=t) for i, t in DOCS])
    return idx


def is_refusal(text: str) -> bool:
    low = text.lower()
    return any(p in low for p in REFUSALS)


@dataclass(frozen=True)
class Family:
    name: str
    dataset: Dataset
    note: str


def describe(cases: Dataset) -> dict[str, Any]:
    return {"cases": len(cases), "name": cases.name}
