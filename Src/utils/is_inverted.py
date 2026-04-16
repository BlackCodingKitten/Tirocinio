from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Tuple

from datasets import Dataset, concatenate_datasets, load_dataset


def verify_maia_mc_json_mapping(
    json_path: str | Path,
    dataset_name: str = "caput/MAIA_ita",
    dataset_config: str = "mc",
) -> None:
    """
    Verify that the generated JSON preserves the MAIA multiple-choice mapping:

    - answer1 (MAIA option A) <-> key "0"
    - answer2 (MAIA option B) <-> key "1"
    - target is unchanged

    This function checks structural consistency only.
    It does not try to determine whether answer1/answer2 are semantically correct.
    """
    train_ds: Dataset = load_dataset(dataset_name, dataset_config, split="train")
    test_ds: Dataset = load_dataset(dataset_name, dataset_config, split="test")
    ds: Dataset = concatenate_datasets([train_ds, test_ds])

    rows: List[Dict[str, Any]] = [ds[i] for i in range(len(ds))]
    rows.sort(key=lambda x: (x["video_id"], x["question_category"], x["id"]))

    with Path(json_path).open("r", encoding="utf-8") as f:
        generated: Dict[str, Any] = json.load(f)

    fallback_pool_counter: DefaultDict[Tuple[str, str], int] = defaultdict(int)

    checked = 0
    mismatches: List[Dict[str, Any]] = []

    for row in rows:
        video_id = str(row["video_id"])
        question_category = str(row["question_category"])
        answer1 = str(row["answer1"])   # MAIA option A
        answer2 = str(row["answer2"])   # MAIA option B
        target = int(row["target"])

        if "pool_pos" in row and row["pool_pos"] is not None:
            pool_pos_value = int(row["pool_pos"])
        else:
            key = (video_id, question_category)
            fallback_pool_counter[key] += 1
            pool_pos_value = fallback_pool_counter[key]

        pool_key = f"pool_pos_{pool_pos_value}"

        try:
            node = generated[video_id][question_category][pool_key]
        except KeyError:
            mismatches.append(
                {
                    "type": "missing_entry",
                    "video_id": video_id,
                    "question_category": question_category,
                    "pool_key": pool_key,
                }
            )
            continue

        json_0 = str(node.get("0", ""))
        json_1 = str(node.get("1", ""))
        json_target = int(node.get("target", -999))

        if json_0 != answer1 or json_1 != answer2 or json_target != target:
            mismatches.append(
                {
                    "type": "value_mismatch",
                    "video_id": video_id,
                    "question_category": question_category,
                    "pool_key": pool_key,
                    "expected_0": answer1,
                    "found_0": json_0,
                    "expected_1": answer2,
                    "found_1": json_1,
                    "expected_target": target,
                    "found_target": json_target,
                }
            )

        checked += 1

    extra_entries = []
    for video_id, categories in generated.items():
        if not isinstance(categories, dict):
            extra_entries.append(
                {"type": "invalid_video_node", "video_id": video_id}
            )
            continue

        for question_category, pools in categories.items():
            if not isinstance(pools, dict):
                extra_entries.append(
                    {
                        "type": "invalid_category_node",
                        "video_id": video_id,
                        "question_category": question_category,
                    }
                )
                continue

            for pool_key in pools.keys():
                # This is just for visibility if there are weird unexpected keys.
                if not isinstance(pool_key, str) or not pool_key.startswith("pool_pos_"):
                    extra_entries.append(
                        {
                            "type": "unexpected_pool_key",
                            "video_id": video_id,
                            "question_category": question_category,
                            "pool_key": pool_key,
                        }
                    )

    print(f"Checked entries: {checked}")
    print(f"Mismatches: {len(mismatches)}")
    print(f"Extra/invalid nodes: {len(extra_entries)}")

    if mismatches:
        print("\nFirst 10 mismatches:")
        for item in mismatches[:10]:
            print(json.dumps(item, ensure_ascii=False, indent=2))

    if extra_entries:
        print("\nFirst 10 extra/invalid nodes:")
        for item in extra_entries[:10]:
            print(json.dumps(item, ensure_ascii=False, indent=2))

    if not mismatches and not extra_entries:
        print("\nVerification passed:")
        print('- MAIA answer1 == JSON key "0"')
        print('- MAIA answer2 == JSON key "1"')
        print('- target unchanged')


if __name__ == "__main__":
    verify_maia_mc_json_mapping("Data/Dataset/falzo.json")