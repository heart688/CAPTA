# Data preparation

[Back to README](../README.md)

## Official sources

No traffic arrays are included. Obtain the data from the original providers and follow their access and usage terms:

- **METR-LA / PEMS-BAY:** [DCRNN dataset instructions](https://github.com/liyaguang/DCRNN#data-preparation), including the authors' links to `metr-la.h5` and `pems-bay.h5`.
- **DiDi-Chengdu / DiDi-Shenzhen:** the historical traffic-index datasets from the DiDi GAIA Initiative. Use [DiDi's official data portal](https://outreach.didichuxing.com/opendata/datasets) to check availability or contact the provider at the address published there. The [pFedCTP paper](https://www.ijcai.org/proceedings/2024/0611.pdf) identifies GAIA as their source.

As checked on 2026-10-15, the DiDi portal lists a synthetic ride-hailing dataset, not direct downloads of the historical 524-road Chengdu and 627-road Shenzhen benchmarks. The old `/app-outreach/CTIE` URL now redirects to an education page. Request the historical data from the provider if needed; the synthetic dataset is not a substitute for these benchmarks. This repository does not mirror either dataset or guarantee access.

## Local input format

Create the following directories locally after obtaining and preparing the data:

```text
data/
  metr-la/dataset.npy
  pems-bay/dataset.npy
  chengdu/dataset.npy
  shenzhen/dataset.npy
```

Arrays are `[time, node, channel]`, with channel 0 = speed in the original unit and channel 1 = normalized time-of-day in `[0,1)`. The loader generates day-of-week from chronological steps when `time_index_channel` is null. The calendar channels are retained for input-format compatibility; CAPTA's calendar embedding is zero.

The experimental arrays have shapes `[34272,207,2]`, `[52116,325,2]`, `[17280,524,2]`, and `[17280,627,2]` respectively. Their exact processed-series hashes are in [configs/dataset.json](../configs/dataset.json). The shape/channel/hash manifest refers to these processed arrays, not to arbitrary provider downloads. An array with another preprocessing convention is a new data setting, not an exact reproduction.

## Preprocessing and checksums

The repository does not include a provider-specific raw-to-NumPy converter: raw schemas and redistribution permissions differ, and the original processed files are not being published here. Keep chronological ordering, original speed units, and a consistent node ordering. The two input channels must be speed and normalized time-of-day, not pre-windowed train/test samples.

The loader checks the full file's shape and SHA256 before training. A new conversion, including different dtype or serialization, may produce a different checksum even if values are similar. For a separately prepared legal dataset, copy the manifest, record its actual shape and SHA256, and pass `--dataset /path/to/manifest.json`; disclose the preprocessing difference when reporting results. Do not copy an old hash onto new data or claim exact reproduction without matching inputs.

Use `shasum -a 256 /path/to/dataset.npy` to obtain a file checksum. A manifest's `root` is resolved relative to that manifest; `--data-root /path/to/data` overrides it.

CAPTA is graph-free and neither loads nor requires `matrix.npy`. Dataset hashes are verified before use. All arrays and generated outputs are excluded by `.gitignore`.
