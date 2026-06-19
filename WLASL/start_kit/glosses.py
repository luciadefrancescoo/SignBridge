import json
from pathlib import Path

BASE = Path(__file__).parent


def load_id_set(path: Path) -> set:
    return {line.strip() for line in path.read_text().splitlines() if line.strip()}


def filter_glosses(data: list, excluded: set) -> tuple[list, list]:
    """
    Returns (filtered_data, results) where:
      filtered_data: list of entries with only valid instances
      results: list of (gloss, [video_ids]) sorted by count desc
    """
    filtered_data = []
    results = []
    for entry in data:
        gloss = entry["gloss"]
        valid_instances = [
            inst for inst in entry["instances"]
            if inst["video_id"] not in excluded
        ]
        if valid_instances:
            filtered_data.append({"gloss": gloss, "instances": valid_instances})
        results.append((gloss, [inst["video_id"] for inst in valid_instances]))
    results.sort(key=lambda x: len(x[1]), reverse=True)
    return filtered_data, results


def build_nslt_json(glosses_path: Path, output_path: Path) -> dict:
    """
    Converts glosses_valid.json to the NSLT format expected by I3D:
      { video_id: { "subset": split, "action": [class_id, frame_start, frame_end] } }
    Saves to output_path and returns the dict.
    """
    content = json.loads(Path(glosses_path).read_text())
    nslt = {}
    for class_id, entry in enumerate(content):
        for inst in entry["instances"]:
            nslt[inst["video_id"]] = {
                "subset": inst["split"],
                "action": [class_id, inst["frame_start"], inst["frame_end"]],
            }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(nslt, indent=2))

    counts = {}
    for v in nslt.values():
        counts[v["subset"]] = counts.get(v["subset"], 0) + 1
    print(f"NSLT JSON guardado: {output_path}")
    print(f"  Total: {len(nslt)} videos")
    for split, n in sorted(counts.items()):
        print(f"  {split}: {n}")
    return nslt


def main():
    data = json.loads((BASE / "WLASL_v0.3.full.json").read_text())
    excluded = load_id_set(BASE / "corrupted.txt") | load_id_set(BASE / "missing.txt")
    filtered_data, results = filter_glosses(data, excluded)

    output_path = BASE / "glosses_valid.json"
    output_path.write_text(json.dumps(filtered_data, indent=2, ensure_ascii=False))
    total_instances = sum(len(e["instances"]) for e in filtered_data)
    print(f"Guardado: {output_path} ({len(filtered_data)} glosses, {total_instances} instancias válidas)")

    print(f"\n{'Rank':<6} {'Gloss':<30} {'Videos':<8} IDs")
    print("-" * 80)
    for rank, (gloss, ids) in enumerate(results[:20], 1):
        print(f"{rank:<6} {gloss:<30} {len(ids):<8} {', '.join(ids)}")


if __name__ == "__main__":
    main()
