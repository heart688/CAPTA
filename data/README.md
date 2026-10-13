# Data layout

No traffic arrays are included. Use datasets you are permitted to access and redistribute.

```text
data/
  metr-la/dataset.npy
  pems-bay/dataset.npy
  chengdu/dataset.npy
  shenzhen/dataset.npy
```

Arrays are `[time, node, channel]`, with channel 0 = speed in the original unit and channel 1 = normalized time-of-day in `[0,1)`. The loader generates day-of-week from chronological steps when `time_index_channel` is null. The calendar channels are retained for input-format compatibility; CAPTA's calendar embedding is zero.

The experimental arrays have shapes `[34272,207,2]`, `[52116,325,2]`, `[17280,524,2]`, and `[17280,627,2]` respectively. Their exact processed-series hashes are in `configs/dataset.json`. The shape/channel/hash manifest refers to these processed arrays, not to arbitrary provider downloads. An array with another preprocessing convention is a new data setting, not an exact reproduction.

The repository does not include a provider-specific raw-to-NumPy converter: raw schemas and redistribution permissions differ, and the original processed files are not being published here. For a different legal dataset, prepare the two channels and edit the manifest with its actual shape and SHA256; disclose that difference when reporting results. Do not copy an old hash onto new data.

CAPTA is graph-free and neither loads nor requires `matrix.npy`. Dataset hashes are verified before use. All arrays and generated outputs are excluded by `.gitignore`.
