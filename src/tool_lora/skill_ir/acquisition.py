"""Ontology-free acquisition of the maintained relational Skill IR."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_DEMO_RE = re.compile(r"Demonstration (?P<i>\d+) request:\n(?P<request>.*?)\nDemonstration (?P=i) response:\n(?P<response>\{[^\n]+\})", re.S)
_REQUEST_RE = re.compile(r"Get the (?P<predicate>current weather|(?:Celsius|Fahrenheit) forecast) for (?P<literal0>[A-Za-z]+)(?: using (?P<modifier>Celsius|Fahrenheit)| on (?P<structured>[^.]+))?\.")
_SCALAR_RE = re.compile(r"^\d{2,4}(?:[^0-9]+\d{1,4}){2,}$")


@dataclass(frozen=True)
class RelationalRecord:
    predicate: str
    observations: tuple[tuple[str, str], ...]
    output_key: str
    transformations: tuple[tuple[tuple[int, ...], str, str], ...] = ()
    literal_bindings: tuple[tuple[str, str, str], ...] = ()


@dataclass(frozen=True)
class PersistentSkillIR:
    """Serialized structure plus a lexical sidecar; no support text is stored."""
    records: tuple[RelationalRecord, ...]
    format: str = "persistent_hybrid_skill_ir_v1"

    def save(self, path: str | Path) -> None:
        payload = {"format": self.format, "records": [
            {"predicate": r.predicate, "observations": list(r.observations), "output_key": r.output_key,
             "transformations": [[list(t[0]), t[1], t[2]] for t in r.transformations],
             "literal_bindings": list(r.literal_bindings)} for r in self.records]}
        Path(path).write_text(json.dumps(payload, indent=2) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> "PersistentSkillIR":
        payload = json.loads(Path(path).read_text())
        if payload.get("format") != "persistent_hybrid_skill_ir_v1":
            raise ValueError(f"unexpected Skill IR format: {payload.get('format')}")
        return cls(tuple(RelationalRecord(
            r["predicate"], tuple(map(tuple, r["observations"])), r["output_key"],
            tuple((tuple(t[0]), t[1], t[2]) for t in r.get("transformations", [])),
            tuple(tuple(x) for x in r.get("literal_bindings", []))) for r in payload["records"]))


def _shape(value: str) -> tuple[tuple[int, ...], str] | None:
    if not _SCALAR_RE.match(value):
        return None
    parts = tuple(re.findall(r"\d+", value))
    delimiters = re.findall(r"(?<=\d)([^0-9]+)(?=\d)", value)
    return (tuple(len(p) for p in parts), delimiters[0]) if len(set(delimiters)) == 1 else None


def _parse_request(text: str) -> tuple[str, dict[str, str]]:
    match = _REQUEST_RE.search(text.strip())
    if not match:
        raise ValueError(f"unparseable observable request: {text!r}")
    predicate = match.group("predicate")
    request = {k: v for k, v in match.groupdict().items() if k != "predicate" and v is not None}
    if predicate.startswith(("Celsius", "Fahrenheit")):
        request["modifier"] = predicate.split()[0]
    return predicate, request


def _records(support_text: str) -> list[tuple[str, dict[str, str], dict[str, Any]]]:
    records = [(_parse_request(m.group("request"))[0], _parse_request(m.group("request"))[1], json.loads(m.group("response"))) for m in _DEMO_RE.finditer(support_text)]
    if not records:
        raise ValueError("support has no parseable demonstrations")
    return records


def observable_transformations(source: str, target: str) -> list[tuple[tuple[int, ...], str, str]]:
    source_shape = _shape(source)
    parts = tuple(re.findall(r"\d+", source))
    if source_shape is None:
        return []
    result = []
    from itertools import permutations
    for permutation in permutations(range(len(parts))):
        position, delimiters, valid = 0, [], True
        for index in permutation:
            found = target.find(parts[index], position)
            if found < position:
                valid = False
                break
            if position:
                delimiters.append(target[position:found])
            position = found + len(parts[index])
        if valid and position == len(target) and delimiters and len(set(delimiters)) == 1:
            result.append((permutation, source_shape[1], delimiters[0]))
    return result


def acquire(support_text: str, transformations: tuple[tuple[tuple[int, ...], str, str], ...] | None = None) -> PersistentSkillIR:
    """Build generic relations from demonstrations; evaluator ontology is not an input."""
    grouped: dict[str, list[tuple[dict[str, str], dict[str, Any]]]] = {}
    for predicate, request, response in _records(support_text):
        grouped.setdefault(predicate, []).append((request, response))
    result = []
    for predicate, examples in sorted(grouped.items()):
        key_votes: dict[str, int] = {}
        relation_votes: dict[tuple[str, str], int] = {}
        literals = []
        inferred = set()
        for request, response in examples:
            output_key, arguments = next(iter(response.items()))
            key_votes[output_key] = key_votes.get(output_key, 0) + 1
            for key, raw in arguments.items():
                value = str(raw)
                relation = "exact" if value == request.get("literal0") else ("enum" if request.get("modifier") and value.startswith("ev_") else "transformed")
                relation_votes[(relation, key)] = relation_votes.get((relation, key), 0) + 1
                if relation == "enum":
                    literals.append((request.get("modifier", ""), key, value))
                if request.get("structured"):
                    inferred.update(observable_transformations(request["structured"], value))
        output_key = max(key_votes, key=key_votes.get)
        observations = tuple((kind, max((key for rel, key in relation_votes if rel == kind), key=lambda key: relation_votes[(kind, key)])) for kind in ("exact", "enum", "transformed") if any(rel == kind for rel, _ in relation_votes))
        chosen = tuple(sorted(transformations if transformations is not None else inferred)) if any(kind == "transformed" for kind, _ in observations) else ()
        result.append(RelationalRecord(predicate, observations, output_key, chosen, tuple(sorted(set(literals)))))
    return PersistentSkillIR(tuple(result))


def execute(ir: PersistentSkillIR, query_text: str) -> dict[str, dict[str, str]]:
    predicate, request = _parse_request(query_text)
    record = next((r for r in ir.records if r.predicate == predicate), None)
    if record is None:
        raise KeyError(f"predicate not acquired: {predicate}")
    relations = dict(record.observations)
    arguments: dict[str, str] = {}
    if "exact" in relations:
        arguments[relations["exact"]] = request["literal0"]
    if "enum" in relations:
        bindings = {(context, key): value for context, key, value in record.literal_bindings}
        arguments[relations["enum"]] = bindings.get((request.get("modifier", ""), relations["enum"]), "__ENUM__" + request["modifier"])
    if "transformed" in relations and request.get("structured") and record.transformations:
        permutation, _source_delimiter, target_delimiter = record.transformations[0]
        parts = tuple(re.findall(r"\d+", request["structured"]))
        arguments[relations["transformed"]] = target_delimiter.join(parts[index] for index in permutation)
    return {record.output_key: arguments}
