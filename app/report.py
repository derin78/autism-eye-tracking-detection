"""
Generate a human-readable test report from screening results.
"""

import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


def generate_report(test_results, classification_results, save=True, subject_name=""):
    """
    Generate a text report from test and classification results.

    Parameters
    ----------
    subject_name : str
        Name of the test subject for report identification.

    Returns the report as a string and optionally saves to file.
    """
    ts = test_results.get("timestamp", datetime.now().strftime("%Y%m%d_%H%M%S"))

    lines = []
    lines.append("=" * 60)
    lines.append("  AUTISM EYE TRACKING SCREENING REPORT")
    lines.append("=" * 60)
    lines.append(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if subject_name:
        lines.append(f"  Subject: {subject_name}")
    lines.append(f"  Session ID: {ts}")
    lines.append("")

    # ── Disclaimer ──
    lines.append("-" * 60)
    lines.append(classification_results.get("disclaimer", ""))
    lines.append("-" * 60)
    lines.append("")

    # ── Overall Result ──
    lines.append("CLASSIFICATION RESULT")
    lines.append("~" * 40)
    classification = classification_results.get("classification", "N/A")
    risk_score = classification_results.get("risk_score", 0)

    lines.append(f"  Classification:     {classification}")
    lines.append(f"  Risk Score:         {risk_score:.1f}%")
    lines.append(f"  Face Attention:     {classification_results.get('face_attention_ratio', 0):.1%}")
    lines.append("")

    # ── Per-Image Results ──
    lines.append("PER-IMAGE RESULTS")
    lines.append("~" * 40)
    lines.append(f"  {'#':>3}  {'Face Side':>10}  {'Samples':>8}  {'Face %':>8}  {'Object %':>9}")
    lines.append(f"  {'---':>3}  {'----------':>10}  {'--------':>8}  {'------':>8}  {'---------':>9}")

    per_image = test_results.get("per_image", [])
    for r in per_image:
        total = r["total_samples"] or 1
        face_pct = r["face_samples"] / total * 100
        obj_pct = r["object_samples"] / total * 100
        lines.append(
            f"  {r['stimulus_id']:>3}  {r['face_side']:>10}  {r['total_samples']:>8}  "
            f"{face_pct:>7.1f}%  {obj_pct:>8.1f}%"
        )
    lines.append("")

    # ── Feature Details ──
    features = classification_results.get("features", [])
    names = classification_results.get("feature_names", [])
    if features and names:
        lines.append("EXTRACTED FEATURES")
        lines.append("~" * 40)
        for name, val in zip(names, features):
            lines.append(f"  {name:40s}  {val:.4f}")
        lines.append("")

    # ── Additional Info ──
    lines.append("ADDITIONAL INFORMATION")
    lines.append("~" * 40)
    lines.append(f"  Total blinks during test:  {test_results.get('total_blinks', 'N/A')}")
    lines.append(f"  ML classification:         {classification_results.get('ml_classification', 'N/A')}")
    lines.append(f"  Threshold classification:  {classification_results.get('threshold_classification', 'N/A')}")
    lines.append(f"  Raw data file:             {test_results.get('raw_data_path', 'N/A')}")
    lines.append("")

    # ── Interpretation Guide ──
    lines.append("INTERPRETATION GUIDE")
    lines.append("~" * 40)
    lines.append("  Typical:        Face attention > 55%, Risk score < 40%")
    lines.append("  Moderate Risk:  Face attention 35-55%, Risk score 40-65%")
    lines.append("  High Risk:      Face attention < 35%, Risk score > 65%")
    lines.append("")
    lines.append("  Higher face attention ratio indicates more typical gaze")
    lines.append("  patterns. Lower ratios may suggest atypical visual")
    lines.append("  attention patterns that warrant professional evaluation.")
    lines.append("")
    lines.append("=" * 60)

    report_text = "\n".join(lines)

    if save:
        os.makedirs(config.REPORTS_DIR, exist_ok=True)
        # Include subject name in filename for easy identification
        safe_name = "".join(c for c in subject_name if c.isalnum() or c in "_ -").strip().replace(" ", "_")
        if safe_name:
            report_path = os.path.join(config.REPORTS_DIR, f"report_{safe_name}_{ts}.txt")
        else:
            report_path = os.path.join(config.REPORTS_DIR, f"report_{ts}.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_text)
        print(f"✅ Report saved to {report_path}")

    return report_text
