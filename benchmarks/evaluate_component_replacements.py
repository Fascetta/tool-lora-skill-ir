"""Evaluate component replacements against the frozen P6-R1 suite.

This benchmark is comparative only. It does not retrain P6-R1 or modify the
reference checkpoint/suite.
"""
from __future__ import annotations

import hashlib
import json
import tempfile
import time
from pathlib import Path

from tool_lora.skill_ir.acquisition import _records, acquire, observable_transformations
from tool_lora.skill_ir.correspondence_matcher import (
    ExactPermutationMatcher, acquire_with_exact_matcher, acquire_with_matcher, load_p6,
)
from tool_lora.skill_ir.canonicalization import canonicalize

ROOT = Path(__file__).parents[1]
REFERENCE = ROOT / "results/stage3d_p6_r1"
CHECKPOINT_SHA256 = "d2721f3684ec7f0dc16bf46834098a35cf9e07d42499e89fe439ea47ac5cb827"
SUITE_SHA256 = "8a8b1d71b1a7dd61973086c1041d4503887caeddced3f1562ab91fd22e6263a5"


def expected(protocol: dict) -> tuple[tuple[int, ...], str, str]:
    return (tuple({"Y": 0, "M": 1, "D": 2}[x] for x in protocol["date_order"]), "-", protocol["date_separator"])


def evaluate(name, acquire_fn, suite):
    rows = []
    started = time.perf_counter()
    for item in suite:
        for condition in ("matched", "alternate", "counterfactual"):
            episode = item[condition]
            protocol = item["counterfactual_protocol"] if condition == "counterfactual" else item["protocol"]
            ir = acquire_fn(episode["support_text"])
            hypotheses = {hypothesis for record in ir.records for hypothesis in record.transformations}
            target = expected(protocol)
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "artifact.json"
                ir.save(path)
                reloaded = type(ir).load(path) == ir
            rows.append({"condition": condition, "any_correct": target in hypotheses,
                         "singleton_correct": hypotheses == {target}, "hypothesis_count": len(hypotheses),
                         "reload": reloaded})
    elapsed = time.perf_counter() - started
    return {"component": name, "cases": len(rows),
            "any_correct": sum(row["any_correct"] for row in rows) / len(rows),
            "singleton_correct": sum(row["singleton_correct"] for row in rows) / len(rows),
            "mean_hypotheses": sum(row["hypothesis_count"] for row in rows) / len(rows),
            "reload": sum(row["reload"] for row in rows) / len(rows),
            "seconds": elapsed, "milliseconds_per_case": elapsed / len(rows) * 1000}


def evaluate_parser_robustness(suite):
    """Compare the maintained parser with a JSON-decoder-based header parser."""
    import re
    decoder = json.JSONDecoder()
    header = re.compile(r"Demonstration (\d+) response:\n")
    successful = 0
    total = 0
    for item in suite:
        for condition in ("matched", "alternate", "counterfactual"):
            support = item[condition]["support_text"]
            total += 1
            try:
                records = []
                for match in header.finditer(support):
                    start = match.end()
                    response, _ = decoder.raw_decode(support[start:].lstrip())
                    records.append(response)
                successful += int(bool(records) and len(records) == len(_records(support)))
            except (ValueError, json.JSONDecodeError):
                pass
    return {"component": "JSONDecoder_response_validation", "cases": total,
            "parse_success": successful / total}


def p6_evidence(model, support_text):
    evidence = []
    for _predicate, request, response in _records(support_text):
        source = request.get("structured")
        if not source:
            continue
        for arguments in response.values():
            for value in arguments.values():
                candidates = observable_transformations(source, str(value))
                if len(candidates) != 1:
                    continue
                matrix = model.matrix(source, str(value))
                from tool_lora.skill_ir.correspondence_matcher import PERMUTATIONS
                matrix = matrix.detach()
                scored = sorted(((sum(float(matrix[permutation[j], j]) for j in range(3)), permutation) for permutation in PERMUTATIONS), reverse=True)
                score, permutation = scored[0]
                evidence.append({"hypothesis": (permutation, candidates[0][1], candidates[0][2]),
                                 "assignment_score": score,
                                 "confidence_margin": score - scored[1][0]})
    return evidence


def select_rule(evidence, rule):
    from collections import defaultdict
    groups = defaultdict(lambda: {"count": 0, "score": 0.0, "margin": 0.0})
    for item in evidence:
        group = groups[item["hypothesis"]]
        group["count"] += 1; group["score"] += item["assignment_score"]; group["margin"] += item["confidence_margin"]
    ranked = sorted(groups.items(), key=lambda item: {
        "A": (item[1]["margin"], item[1]["score"], item[0]),
        "B": (item[1]["score"], item[1]["margin"], item[0]),
        "C": (item[1]["count"], item[1]["score"], item[1]["margin"], item[0]),
        "D": (item[1]["count"], item[1]["score"], item[1]["margin"], item[0]),
    }[rule], reverse=True)
    if not ranked:
        return None
    if rule == "D" and len(ranked) > 1 and ranked[0][1]["count"] == ranked[1][1]["count"] and ranked[0][1]["score"] == ranked[1][1]["score"]:
        return None
    return ranked[0][0]


def canonicalization_evaluation(model, suite):
    results = []
    for rule in "ABCD":
        rows = []
        for item in suite:
            for condition in ("matched", "alternate", "counterfactual"):
                episode = item[condition]
                protocol = item["counterfactual_protocol"] if condition == "counterfactual" else item["protocol"]
                raw = acquire_with_matcher(model, episode["support_text"])
                evidence = p6_evidence(model, episode["support_text"])
                winner = select_rule(evidence, rule)
                canonical = canonicalize(raw, evidence) if rule == "D" else type(raw)(tuple(
                    record if not record.transformations or winner is None else record.__class__(record.predicate, record.observations, record.output_key, (winner,), record.literal_bindings)
                    for record in raw.records))
                hypotheses = {hypothesis for record in canonical.records for hypothesis in record.transformations}
                rows.append({"correct": hypotheses == {expected(protocol)}, "abstained": winner is None})
        results.append({"rule": rule, "cases": len(rows), "canonical_singleton": sum(row["correct"] for row in rows) / len(rows), "abstention": sum(row["abstained"] for row in rows) / len(rows)})
    return {"component": "canonicalization_rules", "results": results}


def main():
    checkpoint = REFERENCE / "p6_r1_checkpoint.pt"
    suite_path = REFERENCE / "immutable_suite.json"
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == CHECKPOINT_SHA256
    assert hashlib.sha256(suite_path.read_bytes()).hexdigest() == SUITE_SHA256
    suite = json.loads(suite_path.read_text())
    p6 = load_p6(checkpoint)
    exact = ExactPermutationMatcher()
    results = [
        evaluate("P6-R1 learned correspondence", lambda text: acquire_with_matcher(p6, text), suite),
        evaluate("exact observable permutation", lambda text: acquire_with_exact_matcher(exact, text), suite),
        evaluate("deterministic acquisition baseline", acquire, suite),
        evaluate_parser_robustness(suite),
        canonicalization_evaluation(p6, suite),
    ]
    output = {"reference": {"checkpoint_sha256": CHECKPOINT_SHA256, "suite_sha256": SUITE_SHA256},
              "results": results,
              "interpretation": {
                  "exact_permutation": "candidate replacement only when component occurrences are observable",
                  "json_decoder": "parser robustness check, not a semantic replacement",
                  "deterministic_acquisition": "existing ontology-free acquisition control",
              }}
    destination = ROOT / "results/component_replacement_evaluation.json"
    destination.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
