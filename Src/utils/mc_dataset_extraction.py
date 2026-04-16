from __future__ import annotations

import json
from collections import defaultdict
from typing import Any, DefaultDict, Dict, List, Tuple

from datasets import Dataset, concatenate_datasets, load_dataset


JsonLeaf = Dict[str, object]
JsonByPool = Dict[str, JsonLeaf]
JsonByCategory = Dict[str, JsonByPool]
JsonByVideo = Dict[str, JsonByCategory]


def main() -> None:
    # Load the Italian pre-split MAIA dataset for the multiple-choice task.
    train_ds: Dataset = load_dataset("caput/MAIA_ita", "mc", split="train")
    test_ds: Dataset = load_dataset("caput/MAIA_ita", "mc", split="test")

    # Merge train and test so that all rows are processed together.
    ds: Dataset = concatenate_datasets([train_ds, test_ds])

    # Convert the dataset to a list of rows to allow deterministic sorting.
    rows: List[Dict[str, Any]] = [ds[i] for i in range(len(ds))]
    rows.sort(key=lambda x: (x["video_id"], x["question_category"], x["id"]))

    # Fallback counter used only when "pool_pos" is not present in a row.
    # The counter is maintained independently for each (video_id, question_category) pair.
    fallback_pool_counter: DefaultDict[Tuple[str, str], int] = defaultdict(int)

    # Final nested JSON structure:
    # {
    #   video_id: {
    #       question_category: {
    #           pool_pos_X: {
    #               "0": answer1,   # MAIA option A
    #               "1": answer2,   # MAIA option B
    #               "target": target
    #           }
    #       }
    #   }
    # }
    result: JsonByVideo = {}

    for row in rows:
        video_id: str = str(row["video_id"])
        question_category: str = str(row["question_category"])

        # In MAIA multiple-choice:
        # - answer1 is option A
        # - answer2 is option B
        # We want:
        # - "0" <-> option A
        # - "1" <-> option B
        answer_a: str = str(row["answer1"])
        answer_b: str = str(row["answer2"])
        target: int = int(row["target"])

        # Use the dataset pool_pos when available.
        # Otherwise, generate a progressive pool position per (video_id, question_category).
        if "pool_pos" in row and row["pool_pos"] is not None:
            pool_pos_value: int = int(row["pool_pos"])
        else:
            key: Tuple[str, str] = (video_id, question_category)
            fallback_pool_counter[key] += 1
            pool_pos_value = fallback_pool_counter[key]

        pool_key: str = f"pool_pos_{pool_pos_value}"

        # Create missing nested dictionaries as needed.
        result.setdefault(video_id, {})
        result[video_id].setdefault(question_category, {})

        # Store MAIA option A as "0" and MAIA option B as "1".
        # The original target is kept unchanged:
        # target == 0 -> correct answer is option A -> key "0"
        # target == 1 -> correct answer is option B -> key "1"
        result[video_id][question_category][pool_key] = {
            "0": answer_a,
            "1": answer_b,
            "target": target,
        }

    output_path: str = "Data/Dataset/maia_ita_mc_by_video_category_pool.json"

    # Save the final structure as a readable UTF-8 JSON file.
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()