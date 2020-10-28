# Cover Song Identification
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/vignejs/cover-song-identification/master) [![MIT license](https://img.shields.io/badge/License-MIT-blue.svg)](https://lbesson.mit-license.org/)

[<img src="https://deepnote.com/buttons/launch-in-deepnote.svg">](https://deepnote.com/project/34044948-e4b5-4552-8a22-f468666deb3c)

Cover Song Identification with Siamese Network. Embeddings can be generated with CNN encoder for faster retrieval of similar (cover) songs for a given query song.

## Dataset
Dataset taken from [Da-Tacos](https://github.com/MTG/da-tacos), which consists of subsets i.e benchmark and coveranalysis. Benchmark subset has 15000 songs with 1000 unique works each with 13 performances (2000 songs were added as noise), whereas Coveranalysis subset has 10000 songs with 5000 unique works each with 2 performances. You can find more in their official github [repo](https://github.com/MTG/da-tacos).

## Preprocess

```
➜ python preprocess.py --help
Usage: preprocess.py [OPTIONS] COMMAND [ARGS]...

  script contains two subcommands to either generates a single h5 file from
  audio directory or da-tacos dataset

Options:
  --help  Show this message and exit.

Commands:
  da-tacos         command to downsize the da-tacos dataset, generates a...
  from-audio-dir  command to convert audio files directly to a single h5...

```
### Generating from da-tacos dataset
```
➜ python preprocess.py da-tacos --help
Usage: preprocess.py da-tacos [OPTIONS]

  command to downsize the da-tacos dataset, generates a single h5 file for
  each subset

Options:
  -d, --data-dir PATH             path to datasets folder which contains da-
                                  tacos_benchmark_subset_single_files/ and da-
                                  tacos_coveranalysis_subset_single_files/
                                  folders  [default: datasets]

  -p, --pcp-features [crema|chroma_cens|hpcp]
                                  select pcp feature (use this option multiple
                                  times for more than one selection)
                                  [default: crema]

  -s, --spect-len INTEGER         Resized output spectral length  [default:
                                  500]

  --help                          Show this message and exit.

```
### Generating from an audio directory
```
➜ python preprocess.py from-audio-dir --help
Usage: preprocess.py from-audio-dir [OPTIONS]

  command to convert audio files directly to a single h5 file for testing
  and evaluation.

Options:
  -a, --audio-dir PATH            path to audio folder  [required]
  -o, --output-dir PATH           path to output folder, where the script will
                                  generate an intermediate features folder and
                                  single h5 file from that folder.  [default:
                                  datasets]

  -p, --pcp-features [crema|chroma_cens|hpcp]
                                  select pcp feature (use this option multiple
                                  times for more than one selection)
                                  [default: crema]

  -r, --run-mode [parallel|single]
                                  Whether to run the extractor in single or
                                  parallel mode.  [default: parallel]

  -n, --workers INTEGER           No of workers in parallel mode  [default:
                                  -1]

  -s, --spect-len INTEGER         Resized output spectral length  [default:
                                  500]

  --help                          Show this message and exit.

```

## Network
![network](nn.png)