# Cover Song Identification
Cover Song Identification with Siamese Network. Embeddings can be generated with CNN encoder for faster retrieval of similar (cover) songs for a given query song.

## Dataset
Dataset taken from [Da-Tacos](https://github.com/MTG/da-tacos), which consists of subsets i.e benchmark and coveranalysis. Benchmark subset has 15000 songs with 1000 unique works each with 13 performances (2000 songs were added as noise), whereas Coveranalysis subset has 10000 songs with 5000 unique works each with 2 performances. You can find more in their official github [repo](https://github.com/MTG/da-tacos).

## Preprocess
```
➜ python preprocess.py --help                                                                                     (coversong) 
Usage: preprocess.py [OPTIONS]

  script to downsize the da-tacos dataset, generates a single h5 file for
  each subset

Options:
  -d, --data-dir PATH             path to datasets folder  [default: datasets]
  -p, --pcp-features [crema|chroma_cens|hpcp]
                                  select pcp feature (use this option multiple
                                  times for more than one selection)
                                  [default: crema]

  --spect-len INTEGER             Resized Spectral Length  [default: 500]
  --help                          Show this message and exit.

```

## Network
![network](nn.png)