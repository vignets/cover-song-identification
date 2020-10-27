import itertools
from pathlib import Path

import click
import cv2
import deepdish as dd
import numpy as np
import pandas as pd
from acoss.extractors import batch_feature_extractor
from tqdm import tqdm


def dir_to_df(data_dir):
    """Converts folder structure to pandas dataframe

    Args:
        data_dir ([Path]): Path to folder

    Returns:
        [pd.DataFrame]: dataframe
    """
    data_track = list()
    for track in data_dir.glob("*/*"):
        track_name = track.stem
        work_name = track.parent.stem
        data_track.append([work_name, track_name])

    data_track_df = pd.DataFrame(data_track, columns=["work_id", "track_id"])
    return data_track_df


def downsize(feature_spec, spect_len):
    """converts input array to a fixed length and downsizes it to desired spect_len.

    Args:
        feature_spec (np.ndarray): input array
        spect_len (int): resized output spectral length

    Returns:
        [np.ndarray]: output array
    """

    feature_spec = feature_spec.astype(np.float32)
    if feature_spec.shape[1] == 12:
        feature_spec = feature_spec.T

    # convert to fixed length of 180 sec
    # hop leangth is 512 and 44100 sampling rate
    # (44100 / 512) * 180 = approx. 15500 time frames
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


def dir_to_h5(data_dir, output_file, pcp_features, spect_len):

    """Packs all feature .h5 files inside folders and subfolders into a single h5 file.

    Args:
        data_dir (Path): path to feature dir containing h5 files
        output_file (Path): full path name to output h5 file
        pcp_features (tuple): sequence containing all pcp features to include
        spect_len (int): resized output spectral length
    """
    store = pd.HDFStore(output_file, mode="w")

    for feature_type in pcp_features:
        data_track_df = dir_to_df(data_dir)
        work_id_max_len = data_track_df.work_id.map(len).max()
        track_id_max_len = data_track_df.track_id.map(len).max()
        rec = data_track_df.to_records(index=False)
        index = pd.MultiIndex.from_tuples(
            [(*r, b) for r, b in itertools.product(rec, range(12))],
            names=["work_id", "track_id", "chroma_bins"],
        )

        feature_specs = np.empty(
            (data_track_df.shape[0], 12, spect_len), dtype=np.float32
        )

        for row in tqdm(data_track_df.itertuples(), total=data_track_df.shape[0]):
            track_file = data_dir / row.work_id / row.track_id
            feature_spec = dd.io.load(track_file.with_suffix(".h5"), f"/{feature_type}")
            feature_specs[row.Index, :, :] = downsize(feature_spec, spect_len)

        df = pd.DataFrame(
            feature_specs.reshape(-1, feature_specs.shape[-1]), index=index
        )

        store.append(
            feature_type,
            df,
            min_itemsize={"work_id": work_id_max_len, "track_id": track_id_max_len},
        )

    store.close()


@click.group(
    help="script contains two subcommands to either generates a single h5 file from audio directory or da-tacos dataset"
)
def preprocess():
    pass


@preprocess.command(
    help="command to downsize the da-tacos dataset, "
    "generates a single h5 file for each subset"
)
@click.option(
    "--data-dir",
    "-d",
    type=click.Path(exists=True),
    help="path to datasets folder which contains da-tacos_benchmark_subset_single_files/ and da-tacos_coveranalysis_subset_single_files/ folders",
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
    "--spect-len",
    "-s",
    help="Resized output spectral length",
    default=500,
    show_default=True,
)
def da_tacos(data_dir, pcp_features, spect_len):

    for dset_folder in ("benchmark", "coveranalysis"):
        feature_dir = Path(data_dir) / f"da-tacos_{dset_folder}_subset_single_files"
        output_file = Path(data_dir) / f"{dset_folder}.h5"
        dir_to_h5(feature_dir, output_file, pcp_features, spect_len)


@preprocess.command(
    help="command to convert audio files directly to a single h5 file for testing and evaluation."
)
@click.option(
    "--audio-dir",
    "-a",
    type=click.Path(exists=True),
    help="path to audio folder",
    required=True,
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(exists=True),
    help="path to output folder, where the script will generate an intermediate features folder and single h5 file from that folder.",
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
    "--run-mode",
    "-r",
    type=click.Choice(["parallel", "single"], case_sensitive=False),
    help="Whether to run the extractor in single or parallel mode.",
    default="parallel",
    show_default=True,
)
@click.option(
    "--workers",
    "-n",
    help="No of workers in parallel mode",
    default=-1,
    show_default=True,
)
@click.option(
    "--spect-len",
    "-s",
    help="Resized output spectral length",
    default=500,
    show_default=True,
)
def from_audio_dir(audio_dir, output_dir, pcp_features, run_mode, workers, spect_len):
    audio_dir = Path(audio_dir)
    feature_dir = Path(output_dir) / f"{audio_dir.stem}_features"
    output_file = Path(output_dir) / f"{audio_dir.stem}.h5"

    data_track_df = dir_to_df(audio_dir)
    data_track_df.to_csv(
        Path(output_dir) / f"{audio_dir.stem}_annotations.csv", index=False
    )

    extractor_profile = {
        "sample_rate": 44100,
        "input_audio_format": ".mp3",
        "downsample_audio": False,
        "downsample_factor": 2,
        "endtime": 180,
        "features": list(pcp_features),
    }

    batch_feature_extractor(
        dataset_csv=Path(output_dir) / f"{audio_dir.stem}_annotations.csv",
        audio_dir=str(audio_dir) + "/",
        feature_dir=str(feature_dir) + "/",
        n_workers=workers,
        mode=run_mode,
        params=extractor_profile,
    )

    dir_to_h5(feature_dir, output_file, pcp_features, spect_len)


if __name__ == "__main__":
    preprocess()
