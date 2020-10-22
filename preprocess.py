import itertools
from pathlib import Path

import click
import cv2
import deepdish as dd
import numpy as np
import pandas as pd
from tqdm import tqdm


def load(song, feature_type, spect_len):
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
        feature_spec, (spect_len, 12), interpolation=cv2.INTER_AREA
    )
    return feature_spec


@click.command(
    help="script to downsize the da-tacos dataset, "
    "generates a single h5 file for each subset"
)
@click.option(
    "--data-dir",
    "-d",
    type=click.Path(exists=True),
    help="path to datasets folder",
    default="datasets",
    show_default=True,
)
@click.option(
    "--pcp-features",
    "-p",
    type=click.Choice(["crema", "chroma_cens", "hpcp"], case_sensitive=False),
    multiple=True,
    help="select pcp feature (use this option multiple times for more than one selection)",
    default=["crema"],
    show_default=True,
)
@click.option(
    "--spect-len", help="Resized Spectral Length", default=500, show_default=True,
)
def preprocess(data_dir, pcp_features, spect_len):

    for dset_folder in ["benchmark", "coveranalysis"]:
        feature_dir = Path(data_dir) / f"da-tacos_{dset_folder}_subset_single_files"
        store = pd.HDFStore(Path(data_dir) / f"{dset_folder}.h5", mode="w")

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

        for feature_type in pcp_features:

            feature_spec = np.empty(
                (dataset_csv.shape[0], 12, spect_len), dtype=np.float32
            )

            for row in tqdm(
                dataset_csv.itertuples(),
                total=dataset_csv.shape[0],
                desc=f"{dset_folder} - {feature_type}",
            ):
                track_file = feature_dir / row.work_id / row.track_id
                feature_spec[row.Index, :, :] = load(
                    track_file.with_suffix(".h5"), feature_type, spect_len
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


if __name__ == "__main__":
    preprocess()
