"""Phase 0.8 R6.2.1: Leakage Regression Verification.

Runs inference three times under different data conditions and
hashes the outputs. If the three canonical JSON hashes are identical,
the model does NOT leak scoring_points or record.csv content.

Exit 0 = PASS (no leakage detected)
Exit 1 = FAIL (leakage detected, or verification could not run)
"""

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import h5py
import jax
import jax.numpy as jnp
from flax.training import train_state as flax_train_state
import orbax.checkpoint as ocp

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(SCRIPT_DIR))

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"

from phaseA_run_inference import (
    run_inference, SYSTEM_CONFIGS, MAX_D, DET_DIM,
)
from foundation.models import RCAWorldFoundation

DEFAULT_CHECKPOINT = str(PROJECT_ROOT / "checkpoints" / "phaseA" / "best")
DEFAULT_DATA_PATH = str(PROJECT_ROOT / "data" / "processed" / "hipster_dataset.h5")


class LeakageVerificationError(RuntimeError):
    """Raised when the verification cannot produce a meaningful result.

    This includes: missing checkpoint, missing data files, or
    inference returning zero predictions (all of which would
    produce identical empty-hash false PASSes).
    """


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def canonical_hash(obj):
    """SHA-256 of canonical JSON representation (sorted keys, compact)."""
    raw = json.dumps(obj, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()


def hash_json_file(path):
    """Read a JSON file and return its canonical SHA-256."""
    with open(path, "r") as f:
        data = json.load(f)
    return canonical_hash(data)


def mutate_scoring_points(query_csv_path):
    """Replace every scoring_points cell with a synthetic non-GT value.

    Preserves all other columns (task_index, instruction, etc.).
    """
    rows = []
    with open(query_csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames)
        for row in reader:
            row["scoring_points"] = (
                "MUTATED: The only root cause is completely fake data "
                "9999-99-99 00:00:00 and the component is nonexistent_xyz "
                "and the reason is fabrication_test")
            rows.append(row)

    with open(query_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def delete_record_csv(data_dir):
    """Remove record.csv if it exists in the data directory."""
    record_path = Path(data_dir) / "record.csv"
    if record_path.exists():
        record_path.unlink()


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------

def _load_model_and_state(checkpoint_path, data_path):
    """Build the model, load the checkpoint and stats.

    Raises:
        LeakageVerificationError: If checkpoint or data file is missing.
    """
    if not os.path.exists(checkpoint_path):
        raise LeakageVerificationError(
            f"Checkpoint not found: {checkpoint_path}")
    if not os.path.exists(data_path):
        raise LeakageVerificationError(
            f"Data file not found: {data_path}")

    model = RCAWorldFoundation(
        common_dim=128, max_obs_dim=MAX_D, num_entity_types=3,
        det_dim=DET_DIM, stoch_dim=32, stoch_classes=32,
        use_onset_head=True, use_component_head=True,
    )
    ex_x = jnp.zeros((1, 24, 10, MAX_D))
    rng = jax.random.PRNGKey(0)
    init_rng, _ = jax.random.split(rng)
    v = model.init(init_rng, ex_x, jnp.zeros(10, dtype=jnp.int32),
                   jax.random.PRNGKey(1))
    tx = __import__("optax").adam(1e-3)
    state = flax_train_state.TrainState.create(
        apply_fn=model.apply, params=v["params"], tx=tx)

    state = state.replace(
        params=ocp.PyTreeCheckpointer().restore(checkpoint_path)["params"])
    print(f"Loaded checkpoint: {checkpoint_path}")

    with h5py.File(data_path, "r") as f:
        ob_mean = f["stats/node_mean"][:].mean(axis=1).reshape(1, 1, 8)
        ob_std = f["stats/node_std"][:].mean(axis=1).reshape(1, 1, 8)

    return model, state, ob_mean, ob_std


# ---------------------------------------------------------------------------
# Single-run wrapper
# ---------------------------------------------------------------------------

def _run_and_hash(system, temp_data_dir, model, state, ob_mean, ob_std,
                  method, all_zero_type, burn_in_min, resample_sec,
                  output_path):
    """Patch SYSTEM_CONFIGS data_dir, run inference, hash output.

    Raises:
        LeakageVerificationError: If inference returns zero predictions.
    """
    orig_cfg = deepcopy(SYSTEM_CONFIGS[system])
    SYSTEM_CONFIGS[system] = {
        **orig_cfg, "data_dir": str(temp_data_dir),
    }
    try:
        t0 = time.time()
        preds = run_inference(
            sys_name=system, model=model, state=state,
            ob_mean=ob_mean, ob_std=ob_std,
            use_posterior=False, scoring_method=method,
            all_zero_type=all_zero_type,
            burn_in_min=burn_in_min, resample_sec=resample_sec,
        )
        elapsed = time.time() - t0
        print(f"  Inference completed in {elapsed:.1f}s, "
              f"{len(preds)} predictions")

        if not preds:
            raise LeakageVerificationError(
                "Inference returned zero predictions; "
                "refusing to report PASS")

        with open(output_path, "w") as f:
            json.dump(preds, f, indent=2, default=str)

        return canonical_hash(preds)
    finally:
        SYSTEM_CONFIGS[system] = orig_cfg


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Leakage regression: verify inference does NOT leak "
                    "scoring_points or record.csv",
    )
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Path to original OpenRCA system data directory "
                             "(e.g. .../OpenRCA/Bank/Bank)")
    parser.add_argument("--system", type=str, required=True,
                        help="Canonical system name (Bank, Telecom, "
                             "Market/cloudbed-1, Market/cloudbed-2)")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT,
                        help="Path to model checkpoint directory")
    parser.add_argument("--data-path", type=str, default=DEFAULT_DATA_PATH,
                        help="Path to hipster_dataset.h5")
    parser.add_argument("--output-dir", type=str, default="leakage_output",
                        help="Directory for three prediction JSON files")
    parser.add_argument("--method", type=str, default="calibrated",
                        choices=["residual_only", "calibrated",
                                 "legacy_per_entity_max", "onset_head"])
    parser.add_argument("--all-zero-type", action="store_true")
    parser.add_argument("--burn-in-min", type=int, default=60)
    parser.add_argument("--resample-sec", type=int, default=120)
    args = parser.parse_args()

    if args.system not in SYSTEM_CONFIGS:
        print(f"ERROR: unknown system '{args.system}'. "
              f"Known: {list(SYSTEM_CONFIGS.keys())}", file=sys.stderr)
        raise SystemExit(1)

    orig_data_dir = Path(args.data_dir)
    if not orig_data_dir.exists():
        print(f"ERROR: data directory not found: {orig_data_dir}",
              file=sys.stderr)
        raise SystemExit(1)

    query_csv = orig_data_dir / "query.csv"
    if not query_csv.exists():
        print(f"ERROR: query.csv not found in {orig_data_dir}",
              file=sys.stderr)
        raise SystemExit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Phase 0.8 R6.2.1: Leakage Regression Verification")
    print(f"System:     {args.system}")
    print(f"Data dir:   {orig_data_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Method:     {args.method} | Burn-in: {args.burn_in_min}min | "
          f"Resample: {args.resample_sec}s")
    print("=" * 60)

    print("\nLoading model...")
    try:
        model, state, ob_mean, ob_std = _load_model_and_state(
            args.checkpoint, args.data_path)
    except LeakageVerificationError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        raise SystemExit(1)
    print("Model ready.\n")

    paths = {
        "original": output_dir / "predictions_original.json",
        "mutated_scoring": output_dir / "predictions_mutated_scoring.json",
        "without_record": output_dir / "predictions_without_record.json",
    }
    hashes = {}

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        try:
            # ---- Run A: original data ----
            print(f"[{'A':>2}/3] Run with ORIGINAL data")
            data_a = tmp / "original"
            shutil.copytree(orig_data_dir, data_a, symlinks=True)
            hashes["original"] = _run_and_hash(
                args.system, data_a, model, state, ob_mean, ob_std,
                args.method, args.all_zero_type,
                args.burn_in_min, args.resample_sec,
                str(paths["original"]),
            )
            print(f"  Hash: {hashes['original']}\n")

            # ---- Run B: mutated scoring_points ----
            print(f"[{'B':>2}/3] Run with MUTATED scoring_points")
            data_b = tmp / "mutated_scoring"
            shutil.copytree(orig_data_dir, data_b, symlinks=True)
            mutate_scoring_points(str(data_b / "query.csv"))
            hashes["mutated_scoring"] = _run_and_hash(
                args.system, data_b, model, state, ob_mean, ob_std,
                args.method, args.all_zero_type,
                args.burn_in_min, args.resample_sec,
                str(paths["mutated_scoring"]),
            )
            print(f"  Hash: {hashes['mutated_scoring']}\n")

            # ---- Run C: no record.csv ----
            print(f"[{'C':>2}/3] Run WITHOUT record.csv")
            data_c = tmp / "without_record"
            shutil.copytree(orig_data_dir, data_c, symlinks=True)
            delete_record_csv(data_c)
            hashes["without_record"] = _run_and_hash(
                args.system, data_c, model, state, ob_mean, ob_std,
                args.method, args.all_zero_type,
                args.burn_in_min, args.resample_sec,
                str(paths["without_record"]),
            )
            print(f"  Hash: {hashes['without_record']}\n")

        except LeakageVerificationError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            raise SystemExit(1)

    # ---- Verdict ----
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    for label, h in hashes.items():
        print(f"  {label:<20s}  {h}")

    unique = set(hashes.values())
    if len(unique) == 1:
        print("\nPASS — no leakage detected (all three hashes identical)")
        raise SystemExit(0)
    else:
        print("\nFAIL — leakage detected (hashes differ)")
        # Show which differ
        for i, (l1, h1) in enumerate(hashes.items()):
            for l2, h2 in list(hashes.items())[i + 1:]:
                if h1 != h2:
                    print(f"  DIFF: {l1} vs {l2}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
