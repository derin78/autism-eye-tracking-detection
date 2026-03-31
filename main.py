"""
main.py — Unified launcher for the Autism Eye Tracking System.

Usage
-----
    python main.py               Launch GUI application (default)
    python main.py gui           Launch GUI application
    python main.py test          Run autism screening test
    python main.py calibrate     Run 9-point screen calibration
    python main.py track         Start eye tracking + real-time gaze overlay
    python main.py record        Record raw gaze data only (no overlay)
    python main.py analyze       Process recorded data and generate heatmap
    python main.py heatmap       Generate heatmap from existing data
"""

import argparse
import sys
import os

# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(__file__))


def cmd_calibrate(args):
    """Run the calibration routine."""
    from tracking.calibration import run_calibration
    run_calibration()


def cmd_track(args):
    """Start tracking with real-time gaze overlay."""
    from app.visualizer import run_visualizer
    run_visualizer()


def cmd_record(args):
    """Record raw tracking data without overlay."""
    from tracking.tracking import run_standalone
    run_standalone()


def cmd_analyze(args):
    """Process the latest raw data and extract features + heatmap."""
    from data.processing import process_raw
    from data.features import extract_and_save
    from app.heatmap import generate_heatmap

    print("=" * 50)
    print("  STEP 1: Processing raw gaze data")
    print("=" * 50)
    df, processed_path = process_raw(args.input)

    print()
    print("=" * 50)
    print("  STEP 2: Extracting features")
    print("=" * 50)
    extract_and_save(processed_path)

    print()
    print("=" * 50)
    print("  STEP 3: Generating heatmap")
    print("=" * 50)
    generate_heatmap(processed_path)

    print()
    print("✅ Analysis complete!")


def cmd_heatmap(args):
    """Generate heatmap only."""
    from app.heatmap import generate_heatmap
    generate_heatmap(args.input)


def cmd_gui(args):
    """Launch the GUI application."""
    from app.gui import launch_gui
    launch_gui()


def cmd_test(args):
    """Run the autism screening test."""
    from app.test_session import run_test_session
    from app.classifier import AutismGazeClassifier
    from app.report import generate_report

    results = run_test_session()
    if results is None:
        print("Test was cancelled.")
        return

    classifier = AutismGazeClassifier()
    classification = classifier.classify(results)
    report = generate_report(results, classification)
    print(report)


def main():
    parser = argparse.ArgumentParser(
        description="Autism Detection Using Eye Tracking System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py calibrate          Run 9-point calibration
  python main.py track              Track + visualize gaze in real time
  python main.py record             Record raw gaze data to CSV
  python main.py analyze            Process data → features → heatmap
  python main.py analyze -i data/raw/gaze_20260129_192312.csv
  python main.py heatmap            Generate heatmap from latest data
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # calibrate
    sub_cal = subparsers.add_parser("calibrate", help="Run screen calibration")
    sub_cal.set_defaults(func=cmd_calibrate)

    # track
    sub_track = subparsers.add_parser("track", help="Track gaze with overlay")
    sub_track.set_defaults(func=cmd_track)

    # record
    sub_rec = subparsers.add_parser("record", help="Record raw gaze data")
    sub_rec.set_defaults(func=cmd_record)

    # analyze
    sub_ana = subparsers.add_parser("analyze", help="Process data & extract features")
    sub_ana.add_argument("-i", "--input", default=None,
                         help="Path to specific raw CSV to analyze")
    sub_ana.set_defaults(func=cmd_analyze)

    # heatmap
    sub_heat = subparsers.add_parser("heatmap", help="Generate gaze heatmap")
    sub_heat.add_argument("-i", "--input", default=None,
                          help="Path to gaze CSV")
    sub_heat.set_defaults(func=cmd_heatmap)

    # gui
    sub_gui = subparsers.add_parser("gui", help="Launch GUI application")
    sub_gui.set_defaults(func=cmd_gui)

    # test
    sub_test = subparsers.add_parser("test", help="Run autism screening test")
    sub_test.set_defaults(func=cmd_test)

    args = parser.parse_args()

    if args.command is None:
        # Default: launch GUI
        cmd_gui(args)
    else:
        args.func(args)


if __name__ == "__main__":
    main()
