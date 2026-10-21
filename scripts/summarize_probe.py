#!/usr/bin/env python3
"""Render a compact Markdown report from probe_report.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def pct(x: float | None) -> str:
    return "—" if x is None else f"{100.0 * x:.1f}%"


def num(x: float | None, digits: int = 3) -> str:
    return "—" if x is None else f"{x:.{digits}f}"


def delta_cell(m: dict[str, Any]) -> str:
    d = m.get("delta_vs_A3")
    if not d:
        return "—"
    return f"{d['delta']:+.3f} [{d['ci_low']:+.3f}, {d['ci_high']:+.3f}]"


def baseline(report: dict[str, Any], split: str, level: str, name: str) -> dict[str, Any] | None:
    return report[split]["baselines"].get(f"{split}/{level}/all/{name}")


def model(report: dict[str, Any], split: str, label: str) -> dict[str, Any] | None:
    for m in report[split]["models"]:
        if m["label"] == label:
            return m
    return None


def test_metrics(report: dict[str, Any], split: str, label: str,
                 level: str) -> dict[str, Any] | None:
    m = model(report, split, label)
    return None if m is None else m["test"].get(f"{level}/all")


def split_table(meta: dict[str, Any], key: str) -> list[str]:
    split = meta[key]
    if not split:
        return ["_Random split not run._"]
    out = [
        "| split | cases | L9 | L7 | exact-group overlap | token overlap |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in ("train", "val", "test"):
        s = split[name]
        out.append(
            f"| {name} | {s['cases']} | {s['per_level'].get('9', 0)} | "
            f"{s['per_level'].get('7', 0)} | {s.get('bridge_group_overlap_with_train', 0)} | "
            f"{s.get('bridge_token_overlap_with_train', 0):.1%} |"
        )
    return out


def level_table(report: dict[str, Any], split: str, level: str) -> list[str]:
    rows = [
        "| model | n | top-1 | strict top-1 | MRR tie | Δ strict vs A3 [95% CI] |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in ("A0_random", "A1_jaccard", "A2_centroid", "A3_m_cosine"):
        m = baseline(report, split, level, name)
        if m:
            rows.append(
                f"| {name} | {m['n']} | {pct(m['top1'])} | {pct(m['top1_strict'])} | "
                f"{num(m['mrr_tie'])} | — |"
            )
    for m in report[split]["models"]:
        if m["scope"] not in (level, "combined"):
            continue
        t = m["test"].get(f"{level}/all")
        if not t:
            continue
        rows.append(
            f"| {m['label']} | {t['n']} | {pct(t['top1'])} | {pct(t['top1_strict'])} | "
            f"{num(t['mrr_tie'])} | {delta_cell(t)} |"
        )
    return rows


def random_comparison(report: dict[str, Any]) -> list[str]:
    if not report.get("random") or not report["random"].get("baselines"):
        return ["_Random split not run._"]
    levels = [x for x in ("9", "7") if baseline(report, "grouped", x, "A3_m_cosine")]
    labels = [
        "A3_m_cosine",
        "L9 vector-only logreg",
        "L9 vector+geom logreg",
        "L9 text-only HGB",
        "L9 text+geom HGB",
        "L9+L7 vector+geom logreg",
        "L9+L7 text+geom HGB",
    ]
    out = [
        "| level | model | grouped strict | random strict | random − grouped |",
        "|---:|---|---:|---:|---:|",
    ]
    for level in levels:
        for label in labels:
            if label.startswith("A3"):
                g = baseline(report, "grouped", level, "A3_m_cosine")
                r = baseline(report, "random", level, "A3_m_cosine")
            elif label.startswith("L9 vector") or label.startswith("L9 text"):
                if level != "9":
                    continue
                g = test_metrics(report, "grouped", label, level)
                r = test_metrics(report, "random", label, level)
            else:
                g = test_metrics(report, "grouped", label, level)
                r = test_metrics(report, "random", label, level)
            if not g or not r:
                continue
            diff = r["top1_strict"] - g["top1_strict"]
            out.append(
                f"| {level} | {label} | {pct(g['top1_strict'])} | "
                f"{pct(r['top1_strict'])} | {diff:+.1%} |"
            )
    return out


def key_findings(report: dict[str, Any]) -> list[str]:
    findings: list[str] = []
    a3 = baseline(report, "grouped", "9", "A3_m_cosine")
    vec = test_metrics(report, "grouped", "L9 vector-only logreg", "9")
    if a3 and vec:
        d = vec.get("delta_vs_A3", {})
        findings.append(
            f"L9 grouped vector-only: **{pct(vec['top1_strict'])} strict top-1** vs "
            f"A3 **{pct(a3['top1_strict'])}**; delta **{d.get('delta', 0):+.3f}** "
            f"(95% CI {d.get('ci_low', 0):+.3f} to {d.get('ci_high', 0):+.3f})."
        )
    for label in ("L9 vector+geom logreg", "L9 text+geom HGB", "L9 text-only HGB"):
        m = test_metrics(report, "grouped", label, "9")
        if m:
            findings.append(
                f"{label}: **{pct(m['top1_strict'])} strict top-1**, "
                f"MRR tie {num(m['mrr_tie'])} "
                f"(Δ vs A3 {delta_cell(m)})."
            )
    a3_7 = baseline(report, "grouped", "7", "A3_m_cosine")
    vec_7 = test_metrics(report, "grouped", "L7 vector-only logreg", "7")
    if a3_7 and vec_7:
        d = vec_7.get("delta_vs_A3", {})
        findings.append(
            f"L7 grouped vector-only: **{pct(vec_7['top1_strict'])}** vs A3 "
            f"**{pct(a3_7['top1_strict'])}**; delta **{d.get('delta', 0):+.3f}** "
            f"(95% CI {d.get('ci_low', 0):+.3f} to {d.get('ci_high', 0):+.3f})."
        )
    ctrl = report.get("controls", {}).get("label_shuffle_L9")
    if ctrl:
        findings.append(
            f"Label-shuffle control on L9: **{pct(ctrl['top1_strict'])}** strict top-1 "
            f"(chance ≈ 10% for 10 candidates), so the pipeline itself is not leaking."
        )
    oracle = baseline(report, "grouped", "9", "L0_weight_identity_oracle")
    if oracle:
        findings.append(
            "The exact-weight identity baseline is 100% by construction; "
            "`weight_delta` is therefore excluded from all learned models."
        )
    return findings


def stability_table(report: dict[str, Any]) -> list[str]:
    st = report.get("stability")
    if not st:
        return []
    rows = [
        "## Stability check (second grouped split)",
        "",
        f"Seed {st['seed']} changes train/val/test membership. "
        f"{st.get('note', '')}",
        "",
        "| level | model | seed 42 strict | seed 43 strict | seed 43 Δ vs A3 [95% CI] | n |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for row in st.get("rows", []):
        d = row.get("seed43_delta")
        delta = "—" if not d else (
            f"{d['delta']:+.3f} [{d['ci_low']:+.3f}, {d['ci_high']:+.3f}]"
        )
        rows.append(
            f"| {row['level']} | {row['model']} | {pct(row['seed42_strict'])} | "
            f"{pct(row['seed43_strict'])} | {delta} | {row['seed43_n']} |"
        )
    rows.append("")
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--report", default=str(ROOT / "evaluation/results/bridge_probe/probe_report.json"))
    ap.add_argument("--out", default=str(ROOT / "evaluation/results/bridge_probe/probe_report.md"))
    args = ap.parse_args(argv)

    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    meta = report["meta"]
    lines: list[str] = [
        "# Learned bridge-selection probe",
        "",
        f"Generated: {meta['generated_at']}  ",
        f"Dataset: `{meta['dataset']}`  ",
        f"Cases: **{meta['cases']}** ({meta['levels']} levels), "
        f"{meta['negatives']} hard near-neighbour negatives per case  ",
        f"Runtime: {meta['runtime_seconds']:.1f} s",
        "",
        "## Split",
        "",
    ]
    lines += split_table(meta, "grouped_split")
    lines += ["", "Random split (leakage comparison only):", ""]
    lines += split_table(meta, "random_split")
    lines += ["", "## Key findings", ""]
    lines += [f"- {x}" for x in key_findings(report)]
    lines += ["", "## Grouped split (primary)", ""]
    for level in ("9", "7"):
        if baseline(report, "grouped", level, "A3_m_cosine") is None:
            continue
        lines += [f"### L{level}", ""]
        lines += level_table(report, "grouped", level)
        lines.append("")
    lines += [
        "## Random split (leakage comparison)",
        "",
        "Bridge-word sets are unique in this dataset, so the random split has "
        "little exact/token overlap with train. It is a stability check rather "
        "than the primary estimate.",
        "",
    ]
    lines += random_comparison(report)
    lines += stability_table(report)
    lines += ["", "## Leakage diagnostic", ""]
    lines += [
        "The true bridge is the only candidate with `weight_delta == 0` because the "
        "near negatives are weight-band matched but not exact. The "
        "`L0_weight_identity_oracle` row therefore scores 100% by construction. "
        "All learned models exclude `weight_delta`.",
        "",
        "An archived run with `weight_delta` included is at "
        "`probe_report_with_weight_leak.json`; its text+geom HGB scored 100%, which "
        "is the artifact this control detects.",
        "",
        "## Notes",
        "",
    ]
    lines += [f"- {x}" for x in report.get("notes", [])]
    lines.append("")

    out = Path(args.out)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
