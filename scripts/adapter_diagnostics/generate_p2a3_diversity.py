#!/usr/bin/env python
"""Build verified P2-A.3 surface/function diversity datasets from oracle rows.

Each directory is a trainable task ID. A latent-group manifest maps multiple
surface IDs to one free latent, while validation rows use a held-out surface
realization for every family.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path


FAMILIES = [
    "userinfoforsoundcloud",
    "playlistfordeezer",
    "gettriviafactfornumbers",
    "genrefordeezer",
    "getdatefactfornumbers",
    "getmathfactfornumbers",
    "getyearfactfornumbers",
    "artistfordeezer",
    "playlistinfoforsoundcloud",
    "songinfoforsoundcloud",
    "getstandardmaptileformaptiles",
    "timeframeforcurrencyapinet",
    "findplacebytextfortruewayplaces",
    "getmaptilewithfrenchlabelsformaptiles",
    "5dayforecastforweather",
    "commentsearchforsocialgrep",
]


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def surface_row(row: dict, family_index: int, variant: int) -> dict:
    target = json.loads(row["target"])
    old_tool = str(target["tool"])
    old_arguments = dict(target.get("arguments") or {})
    old_keys = list(old_arguments)
    if variant == 0:
        new_tool = old_tool
        new_keys = old_keys
    else:
        new_tool = f"family_{family_index:02d}_surface_{variant}_call"
        new_keys = [f"argument_{index}" for index in range(len(old_keys))]

    prompt = str(row["prompt"])
    # Replace the tool identity first: argument names can occur inside an
    # underscored tool name (for example, ``query``).
    prompt = prompt.replace(old_tool, new_tool)
    for old_key, new_key in zip(old_keys, new_keys):
        prompt = re.sub(
            rf"(?<![A-Za-z0-9_]){re.escape(old_key)}(?![A-Za-z0-9_])",
            new_key,
            prompt,
        )
    if new_tool not in prompt or any(new_key not in prompt for new_key in new_keys):
        raise ValueError(
            f"Source row {row.get('example_id')} has target/schema mismatch after surface transformation"
        )
    transformed_target = {
        "type": target.get("type", "CALL"),
        "tool": new_tool,
        "arguments": {new: old_arguments[old] for old, new in zip(old_keys, new_keys)},
    }
    result = dict(row)
    result["example_id"] = f"{row.get('example_id', 'row')}_f{family_index}_v{variant}"
    result["tool_id"] = f"family_{family_index:02d}_surface_{variant}"
    result["prompt"] = prompt
    result["target"] = json.dumps(transformed_target, ensure_ascii=False, separators=(",", ":")) + "\n"
    result["full_text"] = prompt + result["target"]
    return result


def write_jsonl(path: Path, values: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(value, ensure_ascii=False) for value in values) + "\n")


def build(
    base_root: Path,
    output_root: Path,
    families: list[str],
    train_variants: list[int],
    validation_variant: int,
) -> None:
    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)
    latent_groups: dict[str, str] = {}
    eval_tool_ids: list[str] = []
    manifest = []
    for family_index, family in enumerate(families):
        source = base_root / family
        train = read_rows(source / "train.jsonl")
        validation = read_rows(source / "val.jsonl")
        group = f"family_{family_index:02d}"
        manifest.append(
            {
                "family": group,
                "source_tool": family,
                "train_variants": train_variants,
                "validation_variant": validation_variant,
            }
        )
        for variant in train_variants:
            task_id = f"family_{family_index:02d}_surface_{variant}"
            destination = output_root / task_id
            destination.mkdir()
            latent_groups[task_id] = group
            write_jsonl(destination / "train.jsonl", [surface_row(row, family_index, variant) for row in train])
            write_jsonl(destination / "val.jsonl", [surface_row(row, family_index, validation_variant) for row in validation])
            (destination / "metadata.json").write_text(
                json.dumps(
                    {
                        "family": group,
                        "source_tool": family,
                        "train_variant": variant,
                        "validation_variant": validation_variant,
                    },
                    indent=2,
                )
                + "\n"
            )
        eval_tool_ids.append(f"family_{family_index:02d}_surface_{train_variants[0]}")
    (output_root / "latent_groups.json").write_text(json.dumps(latent_groups, indent=2) + "\n")
    (output_root / "eval_tool_ids.json").write_text(json.dumps(eval_tool_ids, indent=2) + "\n")
    (output_root / "family_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--families", nargs="+", default=FAMILIES)
    parser.add_argument("--train-variants", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--validation-variant", type=int, default=3)
    args = parser.parse_args()
    build(args.base_root, args.output_root, args.families, args.train_variants, args.validation_variant)
    print(
        json.dumps(
            {
                "families": len(args.families),
                "train_variants": args.train_variants,
                "output": str(args.output_root),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
