# Z-Image-Turbo worker

Text to image with [Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) (Apache 2.0): an 8-step
distilled 6B model, a few seconds a 1024 × 1024 picture. SolisVox draws its concept pictures with it (sheets of four
objects, cut apart and sent to SF3D) and can draw game images. Built by `.github/workflows/zimage.yml` into
`ghcr.io/<repo>:zimage-<sha>`.

The weights (about 20 GB) download at a worker's first start at a pinned revision and stay loaded; on a card under
30 GB the text encoder and the transformer take turns on the GPU (`enable_model_cpu_offload`).

```json
{"input": {"prompt": "a weathered wooden crate, plain light grey background", "width": 1024, "height": 1024}}
{"input": {"prompts": ["sheet one …", "sheet two …"]}}
{"input": {"probe": true}}
```

Output: `images` (`image_b64`, `format` jpeg or png, `seed`, `seconds`), the GPU, whether it offloads, the first
load's seconds and the commit. Endpoint: 24 GB or 48 GB cards on CUDA 12.8 hosts, 40 GB container disk.
