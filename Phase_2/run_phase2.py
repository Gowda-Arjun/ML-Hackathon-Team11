"""Phase 2 Master Runner.

Runs all Phase 2 steps in sequence:
  Step 1: Bayesian Hyperparameter Optimization (Optuna)
  Step 2: HMM + Viterbi Evaluation (Before vs After)
  Step 3: Folium Map Generation for all student-test trips

Usage:
  cd /path/to/mlhack
  python Phase_2/run_phase2.py [--skip-optuna] [--skip-viterbi] [--skip-maps]
"""

import os
import sys
import argparse
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def run_step(name: str, fn, *args, **kwargs):
    print(f"\n{'='*60}")
    print(f"  STEP: {name}")
    print(f"{'='*60}")
    t0 = time.time()
    result = fn(*args, **kwargs)
    elapsed = time.time() - t0
    print(f"\n✅ {name} completed in {elapsed:.1f}s")
    return result


def main():
    parser = argparse.ArgumentParser(description="PathSnap Phase 2 Runner")
    parser.add_argument("--skip-optuna",  action="store_true", help="Skip Bayesian optimization")
    parser.add_argument("--skip-viterbi", action="store_true", help="Skip HMM/Viterbi evaluation")
    parser.add_argument("--skip-maps",    action="store_true", help="Skip Folium map generation")
    parser.add_argument("--partition",    default="Student folder/student_test",
                        help="Partition dir for Viterbi + maps")
    parser.add_argument("--max-trips",    type=int, default=None,
                        help="Limit trips (useful for quick testing)")
    args = parser.parse_args()

    print("\n" + "🗺️  PathSnap Phase 2 — Full Pipeline".center(60))
    print("=" * 60)

    tuned_model_path = "artifacts/models/HistGradientBoosting_tuned.joblib"
    fallback_model   = "artifacts/models/HistGradientBoosting.joblib"

    # ── Step 1: Bayesian Optimization ────────────────────────────
    if not args.skip_optuna:
        from Phase_2.bayesian_opt.bayesian_tune import run_bayesian_optimization
        run_step("Bayesian Hyperparameter Optimization (Optuna)", run_bayesian_optimization)
    else:
        print("\n[SKIP] Bayesian Optimization")

    # ── Step 2: Viterbi Evaluation ────────────────────────────────
    if not args.skip_viterbi:
        from Phase_2.hmm.viterbi import evaluate_viterbi_on_partition

        model_path = tuned_model_path if os.path.exists(tuned_model_path) else fallback_model
        out_csv    = "artifacts/results/viterbi_evaluation.csv"

        df = run_step(
            "HMM + Viterbi Evaluation (Before vs After)",
            evaluate_viterbi_on_partition,
            args.partition,
            model_path,
            max_trips=args.max_trips,
        )

        os.makedirs("artifacts/results", exist_ok=True)
        df.to_csv(out_csv, index=False)
        print(f"\nSaved Viterbi evaluation to {out_csv}")

        # Print summary
        if not df.empty:
            print("\n── Aggregate Summary ──")
            print(f"  Mean Classifier Top-1 Acc : {df['clf_top1_acc'].mean():.4f}")
            print(f"  Mean HMM Viterbi Top-1 Acc: {df['hmm_top1_acc'].mean():.4f}")
            print(f"  Mean Classifier Continuity : {df['clf_continuity'].mean():.4f}")
            print(f"  Mean HMM Continuity        : {df['hmm_continuity'].mean():.4f}")
    else:
        print("\n[SKIP] HMM + Viterbi Evaluation")

    # ── Step 3: Folium Maps ───────────────────────────────────────
    if not args.skip_maps:
        from Phase_2.viz.folium_maps import generate_all_trip_maps

        model_path = tuned_model_path if os.path.exists(tuned_model_path) else fallback_model
        maps_out   = "artifacts/plots/folium_maps"

        paths = run_step(
            "Folium Map Generation (All Student-Test Trips)",
            generate_all_trip_maps,
            args.partition,
            maps_out,
            model_path=model_path,
            max_trips=args.max_trips,
        )
        print(f"\nGenerated {len(paths)} interactive maps in '{maps_out}/'")
    else:
        print("\n[SKIP] Folium Map Generation")

    # ── Final Summary ─────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  Phase 2 Complete! Outputs:")
    print("  • artifacts/bayesian_opt/     — Optuna study + plots")
    print("  • artifacts/models/*_tuned.joblib — Tuned model weights")
    print("  • artifacts/results/viterbi_evaluation.csv — HMM comparison")
    print("  • artifacts/plots/folium_maps/ — Interactive HTML maps")
    print(f"{'='*60}")
    print("\nNext steps:")
    print("  1. Start the FastAPI server (Laptop 1):")
    print("     uvicorn Phase_2.backend.app:app --host 0.0.0.0 --port 8000")
    print("  2. Run the Streamlit UI (Laptop 2):")
    print("     streamlit run Phase_2/frontend/app.py")


if __name__ == "__main__":
    main()
