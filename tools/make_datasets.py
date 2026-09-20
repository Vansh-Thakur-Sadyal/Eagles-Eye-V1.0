#!/usr/bin/env python
"""Sentinel dataset builder.

    python tools/make_datasets.py structure
    python tools/make_datasets.py behaviour --per-category 40 --render
    python tools/make_datasets.py occlusion --source datasets/source_faces
    python tools/make_datasets.py reasoning
    python tools/make_datasets.py validate
    python tools/make_datasets.py all --per-category 40

Everything is written under --out (default: ./datasets).
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sentinel_datasets import behaviour, occlusion, reasoning, structure, validate  # noqa: E402

DEFAULT_OUT = Path(__file__).resolve().parents[1] / "datasets"

PROVENANCE = {
    "behaviour": {
        "kind": "simulated",
        "suitable_for": [
            "training and evaluating the behaviour, relationship and object agents on "
            "trajectory features (the representation they use in production)",
            "threshold tuning",
            "end-to-end pipeline testing",
        ],
        "not_suitable_for": [
            "training a detector - a model trained on these renders will not generalise "
            "to real CCTV; use Open Images / MOT17 / UCF-Crime for that",
        ],
    },
    "occlusion": {
        "kind": "augmented from real photographs you supplied",
        "suitable_for": [
            "face-occlusion classification",
            "appearance-robustness training and testing",
        ],
        "not_suitable_for": [
            "building a permanent identity database - every sample traces back to a "
            "source identity and should be deleted with it",
        ],
    },
    "reasoning": {
        "kind": "derived from incidents this deployment actually recorded",
        "suitable_for": [
            "fine-tuning incident narration and the conversational assistant",
            "regression-testing the guardrails",
        ],
        "not_suitable_for": [
            "anything, until you have collected a few hundred real incidents",
        ],
    },
}


def _write_manifest(out: Path, section: str, result: dict) -> None:
    manifest_path = out / "manifest.json"
    manifest = {}
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            manifest = {}
    manifest.setdefault("generated", {})
    manifest["generated"][section] = {
        "at": datetime.now(timezone.utc).isoformat(),
        "provenance": PROVENANCE.get(section),
        "result": result,
    }
    manifest["read_this_first"] = (
        "Simulated trajectory data, augmented imagery and real footage have different "
        "standing. Check `provenance` for each section before training on it."
    )
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Sentinel AI datasets")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="dataset root")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("structure", help="create the folder tree and export camera/zone metadata")

    p_behaviour = sub.add_parser("behaviour", aliases=["behavior"],
                                 help="generate the custom behaviour scenarios")
    p_behaviour.add_argument("--per-category", type=int, default=40)
    p_behaviour.add_argument("--categories", nargs="*", default=list(behaviour.GENERATORS))
    p_behaviour.add_argument("--render", action="store_true", help="also render MP4 clips")
    p_behaviour.add_argument("--seed", type=int, default=20260919)

    p_occlusion = sub.add_parser("occlusion", help="augment real photographs into "
                                                   "occlusion + appearance sets")
    p_occlusion.add_argument("--source", type=Path, required=True,
                             help="folder of <identity>/<photo>.jpg")
    p_occlusion.add_argument("--compose", type=int, default=3)
    p_occlusion.add_argument("--no-appearance", action="store_true")

    p_reasoning = sub.add_parser("reasoning", help="export incident records and Q&A pairs")
    p_reasoning.add_argument("--database-url", default=None)
    p_reasoning.add_argument("--limit", type=int, default=5000)

    p_validate = sub.add_parser("validate", help="check the dataset before training")
    p_validate.add_argument("--json", action="store_true", help="machine-readable output")

    p_all = sub.add_parser("all", help="structure + behaviour + reasoning, then validate")
    p_all.add_argument("--per-category", type=int, default=40)
    p_all.add_argument("--render", action="store_true")
    p_all.add_argument("--source", type=Path, default=None,
                       help="optional face folder for the occlusion set")

    args = parser.parse_args()
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)

    if args.command == "structure":
        result = structure.build(out)
        _write_manifest(out, "structure", result)
        print(json.dumps(result, indent=2))
        return 0

    if args.command in ("behaviour", "behavior"):
        result = behaviour.build(
            out, categories=args.categories, per_category=args.per_category,
            seed=args.seed, render=args.render,
        )
        _write_manifest(out, "behaviour", result)
        print(json.dumps(result, indent=2))
        return 0

    if args.command == "occlusion":
        result = occlusion.build(
            args.source, out, compose=args.compose,
            include_appearance=not args.no_appearance,
        )
        _write_manifest(out, "occlusion", result)
        print(json.dumps(result, indent=2))
        return 1 if result.get("error") else 0

    if args.command == "reasoning":
        result = reasoning.build(out, database_url=args.database_url, limit=args.limit)
        _write_manifest(out, "reasoning", result)
        print(json.dumps(result, indent=2))
        return 0

    if args.command == "validate":
        report = validate.validate(out)
        if args.json:
            print(json.dumps(report.to_dict(), indent=2))
        else:
            print(validate.format_report(report))
        return 0 if report.ok else 1

    if args.command == "all":
        print("== structure ==")
        result = structure.build(out)
        _write_manifest(out, "structure", result)
        print(f"  {result['folders_created']} folders, metadata from {result['metadata']['source']}")

        print("== behaviour ==")
        result = behaviour.build(
            out, categories=list(behaviour.GENERATORS),
            per_category=args.per_category, render=args.render,
        )
        _write_manifest(out, "behaviour", result)
        for name, stats in result["categories"].items():
            print(f"  {name}: {stats['scenes']} scenes "
                  f"({stats['positive']} positive / {stats['negative']} negative)")

        if args.source:
            print("== occlusion ==")
            result = occlusion.build(args.source, out)
            _write_manifest(out, "occlusion", result)
            print(f"  {result.get('total_generated', 0)} samples from "
                  f"{result.get('identities', 0)} identity/identities")

        print("== reasoning ==")
        result = reasoning.build(out)
        _write_manifest(out, "reasoning", result)
        print(f"  {result['incidents']} incidents, {result['qa_pairs']} Q&A pairs")
        if result.get("warning"):
            print(f"  ! {result['warning']}")

        print("\n== validation ==")
        report = validate.validate(out)
        print(validate.format_report(report))
        return 0 if report.ok else 1

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
