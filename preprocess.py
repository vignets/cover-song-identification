import itertools
import sys
import warnings
from pathlib import Path

import deepdish as dd
import numpy as np
import pandas as pd

import cv2
from tqdm import tqdm

SPECT_LEN = 500
ROOT = Path.cwd()


def load(song, feature_type):
    feature_spec = dd.io.load(song, f"/{feature_type}")
    feature_spec = feature_spec.astype(np.float32)
    if feature_spec.shape[1] == 12:
        feature_spec = feature_spec.T

    # convert to fixed length
    desired_spect_len = 15500
    if feature_spec.shape[1] >= desired_spect_len:
        feature_spec = feature_spec[:, :desired_spect_len]
    else:
        feature_spec = np.pad(
            feature_spec,
            ((0, 0), (0, desired_spect_len - feature_spec.shape[1])),
            "wrap",
        )

    # resize
    feature_spec = cv2.resize(
        feature_spec, (SPECT_LEN, 12), interpolation=cv2.INTER_AREA
    )
    return feature_spec


if __name__ == "__main__":
    # path to features data
    feature_types = ["crema", "chroma_cens", "hpcp"]

    for dset, folder in [("trainset", "benchmark"), ("valset", "coveranalysis")]:
        feature_dir = ROOT / "datasets" / f"da-tacos_{folder}_subset_single_files"
        store = pd.HDFStore(ROOT / "datasets" / f"{dset}.h5", mode="w")

        data = list()
        for track in feature_dir.glob("*/*"):
            track_name = track.stem
            work_name = track.parent.stem
            data.append([work_name, track_name])

        dataset_csv = pd.DataFrame(data, columns=["work_id", "track_id"])
        work_id_max_len = dataset_csv.work_id.map(len).max()
        track_id_max_len = dataset_csv.track_id.map(len).max()
        rec = dataset_csv.to_records(index=False)
        index = pd.MultiIndex.from_tuples(
            [(*r, b) for r, b in itertools.product(rec, range(12))],
            names=["work_id", "track_id", "chroma_bins"],
        )

        for feature_type in feature_types:

            feature_spec = np.empty(
                (dataset_csv.shape[0], 12, SPECT_LEN), dtype=np.float32
            )

            for row in tqdm(
                dataset_csv.itertuples(),
                total=dataset_csv.shape[0],
                desc=f"{dset} - {feature_type}",
            ):
                track_file = feature_dir / row.work_id / row.track_id
                feature_spec[row.Index, :, :] = load(
                    track_file.with_suffix(".h5"), feature_type
                )

            df = pd.DataFrame(
                feature_spec.reshape(-1, feature_spec.shape[-1]), index=index
            )

            store.append(
                feature_type,
                df,
                min_itemsize={"work_id": work_id_max_len, "track_id": track_id_max_len},
            )

        store.close()
