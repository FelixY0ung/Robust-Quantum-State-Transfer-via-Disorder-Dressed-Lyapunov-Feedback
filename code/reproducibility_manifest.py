"""Generate a reproducibility manifest for scripts and result artifacts.

The manifest records SHA-256 hashes, file sizes, CSV row counts, and CSV column
names for the public code/results bundle.  It is intentionally read-only with
respect to scientific outputs: it does not rerun simulations or alter existing
CSV files.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from paths import result_path


ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = ROOT / "code"
RESULTS_DIR = ROOT / "results"
MANIFEST_OUTPUTS = {
    "results/reproducibility_manifest.json",
    "results/reproducibility_manifest.md",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def csv_metadata(path: Path) -> dict[str, Any]:
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            return {"row_count": 0, "columns": []}
        row_count = sum(1 for _ in reader)
    return {"row_count": row_count, "columns": header}


def file_record(path: Path) -> dict[str, Any]:
    relative = path.relative_to(ROOT).as_posix()
    record: dict[str, Any] = {
        "path": relative,
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }
    if path.suffix == ".csv":
        record.update(csv_metadata(path))
    return record


def collect_files() -> tuple[list[Path], list[Path]]:
    code_files = sorted(CODE_DIR.glob("*.py"))
    result_files = sorted(
        path
        for path in RESULTS_DIR.rglob("*")
        if path.is_file()
        and path.suffix in {".csv", ".md", ".pdf", ".png"}
        and path.relative_to(ROOT).as_posix() not in MANIFEST_OUTPUTS
        and "__pycache__" not in path.parts
    )
    return code_files, result_files


def manifest() -> dict[str, Any]:
    code_files, result_files = collect_files()
    return {
        "schema": "disorder-dressed-reproducibility-manifest-v1",
        "repository_root": ".",
        "code_files": [file_record(path) for path in code_files],
        "result_files": [file_record(path) for path in result_files],
    }


def write_markdown(data: dict[str, Any]) -> None:
    path = result_path("reproducibility_manifest.md")
    csv_rows = [
        item
        for item in data["result_files"]
        if item["path"].endswith(".csv")
    ]
    with path.open("w", encoding="utf-8") as f:
        f.write("# Reproducibility Manifest\n\n")
        f.write(f"Schema: `{data['schema']}`\n\n")
        f.write(f"Code files hashed: {len(data['code_files'])}\n\n")
        f.write(f"Result artifacts hashed: {len(data['result_files'])}\n\n")
        f.write("## CSV Row Counts\n\n")
        f.write("| path | rows | sha256 |\n")
        f.write("| --- | ---: | --- |\n")
        for item in csv_rows:
            f.write(
                f"| `{item['path']}` | {item['row_count']} | `{item['sha256']}` |\n"
            )


def main() -> None:
    data = manifest()
    json_path = result_path("reproducibility_manifest.json")
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")
    write_markdown(data)
    print(f"wrote {json_path}")
    print(f"wrote {result_path('reproducibility_manifest.md')}")


if __name__ == "__main__":
    main()
