"""
Reproduction test for Malaiwong & Oglu 2026 locomotion analysis.

Runs two tiers:
  1. Demo test   — always runs; uses the 4-genotype subset in demo_data/.
  2. Full test   — runs when --data-dir is supplied; rescans raw tracking
                   data and checks that key metrics match published values.

Usage:
    python test_reproduce.py                              # demo only
    python test_reproduce.py --data-dir /path/to/dataset # demo + full rescan
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile

import pandas as pd

PYTHON = sys.executable

EXPECTED_FULL = {
    "n_recordings":  117,
    "n_genotypes":   32,
    "n_particles":   13709,
    "n2_speed_mean": 0.163,   # mm/s, ± 0.005 tolerance
    # LME fold-changes (± 0.05 tolerance)
    "lme_fc": {
        "T10C6.6": 0.587,
        "ugt-64":  1.415,
        "ser-2":   1.028,
        "tph-1":   0.766,
    },
}

EXPECTED_DEMO = {
    "n_genotypes":   4,
    "n2_speed_mean": 0.163,
}


def run(cmd, cwd=None):
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        print("STDOUT:", result.stdout[-2000:])
        print("STDERR:", result.stderr[-2000:])
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")
    return result.stdout


def check(label, actual, expected, tol=0.005):
    ok = abs(actual - expected) <= tol
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {label}: {actual:.4f}  (expected {expected:.4f} ± {tol})")
    return ok


def run_demo_test():
    print("\n=== Demo test (demo_data/, no NAS required) ===")
    passed = True

    with tempfile.TemporaryDirectory() as tmp:
        demo_tmp = os.path.join(tmp, "demo_data")
        shutil.copytree("demo_data", demo_tmp)
        shutil.copy("batch_postural_comparison.py", tmp)
        shutil.copy("exclude.csv", tmp)

        run([PYTHON, "batch_postural_comparison.py",
             "--out-dir", "demo_data/",
             "--exclude", "exclude.csv",
             "--title",   "Demo test"],
            cwd=tmp)

        rec  = pd.read_parquet(os.path.join(demo_tmp, "recording_data.parquet"))

        n_geno = rec["genotype"].nunique()
        ok = n_geno == EXPECTED_DEMO["n_genotypes"]
        print(f"  [{'PASS' if ok else 'FAIL'}] n_genotypes: {n_geno}  (expected {EXPECTED_DEMO['n_genotypes']})")
        passed &= ok

        n2_speed = rec[rec["genotype"] == "N2"]["speed"].mean()
        passed &= check("N2 mean speed (mm/s)", n2_speed, EXPECTED_DEMO["n2_speed_mean"], tol=0.005)

        fig = os.path.join(demo_tmp, "postural_comparison.png")
        ok  = os.path.exists(fig)
        print(f"  [{'PASS' if ok else 'FAIL'}] figure generated")
        passed &= ok

    return passed


def run_full_test(data_dir):
    print(f"\n=== Full test (rescan from {data_dir}) ===")
    passed = True

    with tempfile.TemporaryDirectory() as tmp:
        shutil.copy("batch_postural_comparison.py", tmp)
        shutil.copy("exclude.csv", tmp)

        run([PYTHON, "batch_postural_comparison.py",
             "--data-dir", data_dir,
             "--out-dir",  "results/",
             "--exclude",  "exclude.csv",
             "--refresh",
             "--title",    "Full reproduction test"],
            cwd=tmp)

        out  = os.path.join(tmp, "results")
        part = pd.read_parquet(os.path.join(out, "particle_data.parquet"))
        rec  = pd.read_parquet(os.path.join(out, "recording_data.parquet"))
        stat = pd.read_csv(os.path.join(out, "postural_comparison_stats.csv"))

        exp = EXPECTED_FULL
        for label, actual, expected in [
            ("n_recordings", len(rec),                exp["n_recordings"]),
            ("n_genotypes",  rec["genotype"].nunique(), exp["n_genotypes"]),
            ("n_particles",  len(part),               exp["n_particles"]),
        ]:
            ok = int(actual) == int(expected)
            print(f"  [{'PASS' if ok else 'FAIL'}] {label}: {int(actual)}  (expected {int(expected)})")
            passed &= ok

        n2_speed = rec[rec["genotype"] == "N2"]["speed"].mean()
        passed &= check("N2 mean speed (mm/s)", n2_speed, exp["n2_speed_mean"], tol=0.005)

        speed_stat = stat[stat["metric"] == "speed"].set_index("genotype")
        for geno, fc_expected in exp["lme_fc"].items():
            if geno in speed_stat.index:
                fc_actual = speed_stat.loc[geno, "fold_change"]
                passed &= check(f"LME fold-change {geno}", fc_actual, fc_expected, tol=0.05)
            else:
                print(f"  [FAIL] {geno} not found in LME results")
                passed = False

    return passed


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default=None,
                        help="Path to dataset root containing genotype subdirectories. "
                             "When supplied, runs the full NAS rescan test in addition to the demo test.")
    args = parser.parse_args()

    results = []
    results.append(("Demo", run_demo_test()))
    if args.data_dir:
        results.append(("Full", run_full_test(args.data_dir)))

    print("\n=== Summary ===")
    all_passed = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
        all_passed &= passed

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
