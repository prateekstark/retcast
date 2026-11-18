"""Download and cache ETT benchmark datasets and Google TimesFM-1.0-200M weights."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from huggingface_hub import hf_hub_download

from retcast.data import ETT_CONFIGS, ETTDataset, download_ett_datasets
from retcast.pretrain_data import download_non_ett_corpus


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/ett")
    parser.add_argument("--non_ett_dir", type=str, default="data/non_ett")
    parser.add_argument("--skip_timesfm", action="store_true")
    parser.add_argument("--skip_non_ett", action="store_true")
    args = parser.parse_args()

    paths = download_ett_datasets(args.data_dir)
    for name, path in paths.items():
        ds = ETTDataset(name, csv_path=path)
        b = ds.boundaries
        print(
            f"[data] {name:6s} -> {path} "
            f"(rows={ds.raw_data.shape[0]}, cols={ds.num_series}, "
            f"splits=train:{b[0]}/val:{b[1]-b[0]}/test:{b[2]-b[1]})"
        )

    if not args.skip_non_ett:
        non_ett = download_non_ett_corpus(args.non_ett_dir)
        print(
            f"[non_ett] cached {len(non_ett)} non-ETT datasets: "
            f"{', '.join(sorted(non_ett.keys()))}"
        )

    if not args.skip_timesfm:
        ckpt = hf_hub_download(
            repo_id="google/timesfm-1.0-200m-pytorch",
            filename="torch_model.ckpt",
        )
        size_mb = Path(ckpt).stat().st_size / (1024 * 1024)
        print(f"[timesfm] cached at {ckpt} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
