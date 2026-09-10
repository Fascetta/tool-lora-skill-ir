from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tool_lora.functional_hypernet.data import (
    build_functional_prompt_from_oracle_row,
    load_jsonl,
    load_split_tools,
)


@dataclass(frozen=True)
class SkillEpisode:
    experience_id: str
    support_text: str
    query_row: dict[str, Any]


class BehavioralEpisodeSampler:
    """Support/query sampler which never exposes tool identity to the model.

    Split metadata is used only to group examples from the same executable tool
    environment. Adapter paths, oracle metrics, categories, and labels are not
    loaded into episode state.
    """

    def __init__(
        self,
        split_path: str | Path,
        *,
        seed: int,
        maximum_tools: int | None = None,
    ) -> None:
        raw_tools = load_split_tools(split_path)
        if maximum_tools is not None:
            raw_tools = raw_tools[:maximum_tools]
        self.tools = [
            {
                "tool_id": str(item["tool_id"]),
                "documentation": str(item["compact_tool_documentation"]),
                "dataset_path": str(item["dataset_path"]),
            }
            for item in raw_tools
        ]
        if not self.tools:
            raise ValueError("Skill episode split has no tools")
        self.rng = random.Random(seed)
        self._rows: dict[str, list[dict[str, Any]]] = {}

    def _tool_rows(self, tool: dict[str, str], split: str = "train") -> list[dict[str, Any]]:
        key = f"{tool['tool_id']}:{split}"
        if key not in self._rows:
            self._rows[key] = load_jsonl(Path(tool["dataset_path"]) / f"{split}.jsonl")
        return self._rows[key]

    def sample(self, *, split: str = "train") -> SkillEpisode:
        eligible = self.tools[:]
        self.rng.shuffle(eligible)
        for tool in eligible:
            rows = self._tool_rows(tool, split)
            if len(rows) < 2:
                continue
            support, query = self.rng.sample(rows, 2)
            support_prompt = build_functional_prompt_from_oracle_row(support).rstrip()
            support_text = (
                "Persistent capability evidence. Infer the reusable behavior from the "
                "tool specification and observed example.\n\n"
                f"Tool specification:\n{tool['documentation'].strip()}\n\n"
                f"Observed request:\n{support_prompt}\n\n"
                f"Observed assistant behavior:\n{str(support['target']).strip()}"
            )
            return SkillEpisode(
                experience_id=f"{support.get('example_id')}->{query.get('example_id')}",
                support_text=support_text,
                query_row=query,
            )
        raise ValueError(f"No tool in {split} has two examples")

    def state_dict(self) -> dict[str, Any]:
        return {"random_state": self.rng.getstate()}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.rng.setstate(state["random_state"])
