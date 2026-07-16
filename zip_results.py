# Run this as a new cell in the Kaggle notebook AFTER training has finished
# (it needs `history`, `G`, and the samples/ folder from the training cells above).

import json
import zipfile
from pathlib import Path

# 1. Dump the loss/timing history so exact numbers (not just eyeballed from a plot)
#    are available for analysis.
with open("training_history.json", "w") as f:
    json.dump(history, f, indent=2)

# 2. Record final evaluation metrics alongside it, if you've already run the
#    `evaluate(val_loader)` cell earlier in the notebook — safe to skip if not.
summary = {
    "num_epochs_run": len(history["G_loss"]),
    "final_G_loss": history["G_loss"][-1] if history["G_loss"] else None,
    "final_D_loss": history["D_loss"][-1] if history["D_loss"] else None,
    "avg_epoch_seconds": (
        sum(history["epoch_seconds"]) / len(history["epoch_seconds"])
        if history["epoch_seconds"] else None
    ),
    "train_pairs": len(train_pairs) if "train_pairs" in globals() else None,
    "val_pairs": len(val_pairs) if "val_pairs" in globals() else None,
}
with open("run_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

# 3. Zip the per-epoch sample grids (small PNGs) + the two JSON files above.
#    Checkpoints are intentionally excluded — they're large binary weights, not
#    something worth uploading for a code/results review.
output_zip = "results_bundle.zip"
with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zf:
    samples_dir = Path("samples")
    if samples_dir.is_dir():
        for f in sorted(samples_dir.glob("*.png")):
            zf.write(f, f"samples/{f.name}")
    zf.write("training_history.json")
    zf.write("run_summary.json")

size_mb = Path(output_zip).stat().st_size / 1e6
print(f"Wrote {output_zip} ({size_mb:.1f} MB)")
print("Download it from the Kaggle file browser (left sidebar -> Output),")
print("then upload it here.")
