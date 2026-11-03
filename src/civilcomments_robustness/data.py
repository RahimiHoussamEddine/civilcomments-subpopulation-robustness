"""Acquire and audit the official WILDS CivilComments v1.0 archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = ROOT / "data/source_manifest.json"
ARCHIVE_PATH = ROOT / "data/raw/civilcomments-v1.0.tar.gz"
CSV_PATH = ROOT / "data/raw/all_data_with_identities.csv"
QUALITY_PATH = ROOT / "reports/data_quality.json"
CHUNK_BYTES = 1024 * 1024


def _read_manifest() -> dict[str, Any]:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def _write_manifest(manifest: dict[str, Any]) -> None:
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_archive(path: Path, manifest: dict[str, Any]) -> str:
    expected_bytes = int(manifest["bundle"]["expected_compressed_bytes"])
    if path.stat().st_size != expected_bytes:
        raise ValueError(f"WILDS archive size mismatch: {path.stat().st_size} != {expected_bytes}")
    actual_sha256 = _digest(path)
    expected_sha256 = manifest["bundle"].get("sha256")
    if expected_sha256 and actual_sha256 != expected_sha256:
        raise ValueError(f"WILDS archive SHA-256 mismatch: {actual_sha256}")
    return actual_sha256


def download_archive() -> Path:
    manifest = _read_manifest()
    ARCHIVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not ARCHIVE_PATH.exists():
        temporary = ARCHIVE_PATH.with_suffix(".tar.gz.part")
        request = urllib.request.Request(
            manifest["bundle"]["url"],
            headers={"User-Agent": "p22-civilcomments-robustness/1.0"},
        )
        try:
            with (
                urllib.request.urlopen(request, timeout=120) as response,
                temporary.open("wb") as output,
            ):
                if response.status != 200:
                    raise ValueError(f"Unexpected WILDS archive HTTP status {response.status}")
                content_type = response.headers.get("Content-Type", "").split(";", 1)[0].lower()
                if content_type not in {"application/gzip", "application/x-gzip"}:
                    raise ValueError(f"Unexpected WILDS archive Content-Type: {content_type}")
                disposition = response.headers.get("Content-Disposition", "")
                if "civilcomments_v1.0.tar.gz" not in disposition:
                    raise ValueError("CodaLab response did not identify the expected v1.0 archive")
                while chunk := response.read(CHUNK_BYTES):
                    output.write(chunk)
            actual_sha256 = _verify_archive(temporary, manifest)
            temporary.replace(ARCHIVE_PATH)
        except (OSError, urllib.error.URLError, TimeoutError, ValueError):
            temporary.unlink(missing_ok=True)
            raise
        manifest["bundle"]["sha256"] = actual_sha256
        _write_manifest(manifest)
        print(f"Downloaded and SHA-256 pinned WILDS bundle: {actual_sha256}")
    else:
        actual_sha256 = _verify_archive(ARCHIVE_PATH, manifest)
        if not manifest["bundle"].get("sha256"):
            manifest["bundle"]["sha256"] = actual_sha256
            _write_manifest(manifest)

    _extract_source_csv(ARCHIVE_PATH)
    return ARCHIVE_PATH


def _extract_source_csv(archive_path: Path) -> Path:
    """Extract only the expected CSV member after checking its path and type."""
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, mode="r:gz") as archive:
        members = [
            member for member in archive.getmembers() if Path(member.name).name == CSV_PATH.name
        ]
        if len(members) != 1 or not members[0].isfile():
            raise ValueError("Expected exactly one all_data_with_identities.csv regular file")
        member = members[0]
        if member.size <= 0:
            raise ValueError("CivilComments source CSV is empty")
        source = archive.extractfile(member)
        if source is None:
            raise ValueError("Could not read CivilComments source CSV from the archive")
        temporary = CSV_PATH.with_suffix(".csv.part")
        try:
            with source, temporary.open("wb") as output:
                while chunk := source.read(CHUNK_BYTES):
                    output.write(chunk)
            if temporary.stat().st_size != member.size:
                raise ValueError("Extracted CivilComments CSV size does not match the tar member")
            temporary.replace(CSV_PATH)
        except (OSError, ValueError):
            temporary.unlink(missing_ok=True)
            raise
    manifest = _read_manifest()
    csv_info = manifest["source_csv"]
    actual_bytes = CSV_PATH.stat().st_size
    actual_sha256 = _digest(CSV_PATH)
    if csv_info.get("bytes") and int(csv_info["bytes"]) != actual_bytes:
        raise ValueError("Extracted CivilComments CSV byte count differs from its recorded value")
    if csv_info.get("sha256") and str(csv_info["sha256"]) != actual_sha256:
        raise ValueError("Extracted CivilComments CSV SHA-256 differs from its recorded value")
    csv_info["bytes"] = actual_bytes
    csv_info["sha256"] = actual_sha256
    _write_manifest(manifest)
    return CSV_PATH


def _load_source() -> pd.DataFrame:
    if not CSV_PATH.exists():
        raise FileNotFoundError(
            "Run `make fetch` to download and extract the official WILDS archive"
        )
    manifest = _read_manifest()
    columns = [
        "split",
        "comment_text",
        "toxicity",
        "article_id",
        *manifest["identity_columns"],
    ]
    return pd.read_csv(CSV_PATH, usecols=columns, low_memory=False)


def _normalized_text(frame: pd.DataFrame) -> pd.Series:
    return frame["comment_text"].fillna("").astype(str).str.casefold().str.split().str.join(" ")


def leakage_safe_splits(frame: pd.DataFrame) -> tuple[dict[str, pd.DataFrame], dict[str, int]]:
    """Remove empty text and repeated normalized text from train/evaluation neighbors."""
    normalized = _normalized_text(frame)
    valid_text = normalized.ne("")
    split_values = frame["split"].astype(str)
    validation_text = set(normalized[(split_values == "val") & valid_text])
    test_text = set(normalized[(split_values == "test") & valid_text])
    evaluation_text = validation_text | test_text

    keep = {
        "train": (split_values == "train") & valid_text & ~normalized.isin(evaluation_text),
        "val": (split_values == "val") & valid_text & ~normalized.isin(test_text),
        "test": (split_values == "test") & valid_text,
    }
    splits = {name: frame.loc[mask].copy() for name, mask in keep.items()}
    before = {name: int((split_values == name).sum()) for name in keep}
    after = {name: len(rows) for name, rows in splits.items()}
    removed = {name: before[name] - after[name] for name in keep}
    return splits, {f"{name}_removed_for_empty_or_duplicate_text": removed[name] for name in keep}


def _group_counts(frame: pd.DataFrame, labels: np.ndarray, identities: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for identity in identities:
        mentioned = pd.to_numeric(frame[identity], errors="coerce").to_numpy() >= 0.5
        for label in (0, 1):
            counts[f"{identity}|{label}"] = int(np.sum(mentioned & (labels == label)))
    return counts


def validate_data() -> dict[str, Any]:
    manifest = _read_manifest()
    if not ARCHIVE_PATH.exists():
        raise FileNotFoundError("Run `make fetch` before `make validate-data`")
    archive_sha256 = _verify_archive(ARCHIVE_PATH, manifest)
    if not manifest["bundle"].get("sha256"):
        manifest["bundle"]["sha256"] = archive_sha256
        _write_manifest(manifest)
    if not CSV_PATH.exists():
        _extract_source_csv(ARCHIVE_PATH)
    manifest = _read_manifest()
    csv_info = manifest["source_csv"]
    if csv_info.get("bytes") and int(csv_info["bytes"]) != CSV_PATH.stat().st_size:
        raise ValueError("Local CivilComments CSV byte count differs from the source manifest")
    if csv_info.get("sha256") and _digest(CSV_PATH) != str(csv_info["sha256"]):
        raise ValueError("Local CivilComments CSV SHA-256 differs from the source manifest")

    frame = _load_source()
    expected_columns = {
        "split",
        "comment_text",
        "toxicity",
        "article_id",
        *manifest["identity_columns"],
    }
    missing_columns = expected_columns - set(frame.columns)
    if missing_columns:
        raise ValueError(f"WILDS columns missing: {sorted(missing_columns)}")
    if len(frame) != int(manifest["expected_rows"]["total"]):
        raise ValueError(f"Unexpected CivilComments row count: {len(frame)}")
    if frame["article_id"].isna().any():
        raise ValueError("CivilComments contains missing source article IDs")

    split_counts = {str(name): int(count) for name, count in frame["split"].value_counts().items()}
    expected_split_counts = {
        str(name): int(count)
        for name, count in manifest["expected_rows"].items()
        if name != "total"
    }
    if split_counts != expected_split_counts:
        raise ValueError(f"CivilComments split counts changed: {split_counts}")

    toxicity = pd.to_numeric(frame["toxicity"], errors="coerce").to_numpy(dtype=np.float64)
    if not np.isfinite(toxicity).all() or np.any((toxicity < 0) | (toxicity > 1)):
        raise ValueError("CivilComments toxicity scores must be finite and lie in [0, 1]")
    labels = (toxicity >= float(manifest["label"]["threshold"])).astype(np.int8)

    identity_missing: dict[str, int] = {}
    for identity in manifest["identity_columns"]:
        values = pd.to_numeric(frame[identity], errors="coerce").to_numpy(dtype=np.float64)
        identity_missing[identity] = int((~np.isfinite(values)).sum())
        if identity_missing[identity] or np.any((values < 0) | (values > 1)):
            raise ValueError(f"Invalid score in identity annotation {identity}")

    article_split_counts = frame.groupby("article_id", sort=False)["split"].nunique()
    article_leakage_count = int((article_split_counts > 1).sum())
    if article_leakage_count:
        raise ValueError(f"Source articles cross the official splits: {article_leakage_count}")

    normalized = _normalized_text(frame)
    nonempty = normalized.ne("")
    duplicate_frame = pd.DataFrame(
        {"normalized_text": normalized[nonempty], "split": frame.loc[nonempty, "split"]}
    ).drop_duplicates()
    cross_split_duplicate_groups = int(
        (duplicate_frame.groupby("normalized_text", sort=False)["split"].nunique() > 1).sum()
    )
    leakage_safe, leakage_exclusions = leakage_safe_splits(frame)

    summary = {
        "dataset": manifest["dataset"],
        "license": manifest["license"],
        "source_archive_sha256": archive_sha256,
        "source_csv_sha256": _digest(CSV_PATH),
        "source_csv_bytes": CSV_PATH.stat().st_size,
        "rows": int(len(frame)),
        "split_counts": {name: int(count) for name, count in split_counts.items()},
        "toxicity_positive_rule": manifest["label"]["positive_rule"],
        "label_counts": {
            "non_toxic": int(np.sum(labels == 0)),
            "toxic": int(np.sum(labels == 1)),
        },
        "missing_or_empty_text_rows_by_split": {
            name: int(((frame["split"] == name) & ~nonempty).sum())
            for name in ("train", "val", "test")
        },
        "identity_columns": list(manifest["identity_columns"]),
        "identity_label_group_counts": _group_counts(frame, labels, manifest["identity_columns"]),
        "unique_articles": int(frame["article_id"].nunique()),
        "cross_split_article_leaks": article_leakage_count,
        "cross_split_normalized_text_duplicate_groups": cross_split_duplicate_groups,
        "leakage_safe_rows": {name: len(rows) for name, rows in leakage_safe.items()},
        "leakage_exclusions": leakage_exclusions,
        "text_rows_committed": False,
        "row_level_predictions_committed": False,
    }
    QUALITY_PATH.parent.mkdir(parents=True, exist_ok=True)
    QUALITY_PATH.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


def load_data() -> pd.DataFrame:
    """Return the source table locally after requiring a successful manifest audit."""
    manifest = _read_manifest()
    archive_sha256 = _verify_archive(ARCHIVE_PATH, manifest)
    if archive_sha256 != manifest["bundle"].get("sha256"):
        raise ValueError("Archive SHA-256 differs from the pinned source manifest")
    if not CSV_PATH.exists():
        _extract_source_csv(ARCHIVE_PATH)
        manifest = _read_manifest()
    csv_info = manifest["source_csv"]
    if csv_info.get("bytes") and int(csv_info["bytes"]) != CSV_PATH.stat().st_size:
        raise ValueError("Local CivilComments CSV byte count differs from the source manifest")
    if csv_info.get("sha256") and _digest(CSV_PATH) != str(csv_info["sha256"]):
        raise ValueError("Local CivilComments CSV SHA-256 differs from the source manifest")
    return _load_source()


def load_leakage_safe_splits() -> tuple[dict[str, pd.DataFrame], dict[str, int]]:
    """Load official splits after removing empty text and cross-split duplicate inputs."""
    return leakage_safe_splits(load_data())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("fetch", "validate"))
    command = parser.parse_args().command
    if command == "fetch":
        download_archive()
        validate_data()
    else:
        validate_data()


if __name__ == "__main__":
    main()
