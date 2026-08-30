# trellis-voxel

A [RunPod serverless](https://runpod.io) worker that turns an image into a
coloured voxel grid using [TRELLIS](https://github.com/microsoft/TRELLIS).

Send it a picture, get back occupied voxel coordinates and an RGB colour per
voxel. No mesh, no renderer.

## Why it returns voxels rather than a mesh

TRELLIS is voxel-based internally. Its first stage — `ss_flow_img_dit_L_16l8`
into `ss_dec_conv3d_16l8` — decodes to a **64³ occupancy grid** before any mesh,
gaussian or radiance-field decoder runs. So reading voxels out is reading the
pipeline's own intermediate representation, not post-processing a mesh.

That also means the fragile CUDA extensions used only for mesh export
(`nvdiffrast`, `kaolin`, `diffoctreerast`) are optional here, and the Dockerfile
treats them as such.

Colour comes from the gaussian decoder: every gaussian carries a position and a
spherical-harmonic DC term, which is its base colour. Those are splatted onto the
occupancy grid — no rendering involved.

## API

Request:

```json
{ "input": {
    "image_b64": "<base64 png>",
    "seed": 0,
    "resolution": 64,
    "sparse_steps": 12,
    "slat_steps": 12,
    "want_colour": true,
    "remove_background": true
} }
```

`image_url` works instead of `image_b64`.

Response:

```json
{ "output": {
    "resolution": 64,
    "coords": [[31, 20, 33]],
    "colours": [[194, 142, 90]],
    "colour_source": "gaussian",
    "counts": { "voxels": 18342 },
    "timings": { "generate": 6.1, "total": 7.4 }
} }
```

`{"input": {"probe": true}}` returns which optional CUDA extensions actually
built — useful because you cannot check that without a GPU.

## Deploy

Push this repo, let the included workflow build and publish to GHCR, then create
a RunPod serverless endpoint from `ghcr.io/<you>/trellis-voxel:latest`.

| setting | value |
|---|---|
| GPU | RTX A4000 16 GB (enough for `TRELLIS-image-large`) |
| Active workers | 0 — nothing is billed while idle |
| Max workers | 1 |
| Container disk | 25 GB |

Weights are **not** baked into the image. Attach them host-side instead:

```
--model-reference microsoft/TRELLIS-image-large
```

Smaller image, faster cold starts, and no monthly storage cost. Build with
`--build-arg BAKE_WEIGHTS=1` if you would rather embed them.

## Model choice

| | weights | est. peak VRAM |
|---|---|---|
| `microsoft/TRELLIS-image-large` | 3.3 GB | ~6–8 GB |
| `microsoft/TRELLIS.2-4B` | 16.2 GB | ~10–16 GB |

TRELLIS.2 depends on TRELLIS-1 — its `pipeline.json` points
`sparse_structure_decoder` at `microsoft/TRELLIS-image-large`. Both therefore
produce the same 64³ grid, so this worker handles either; only colour extraction
differs. For TRELLIS.2 use a 24 GB GPU and reference both repos.

## Licence

The worker glue here is MIT. TRELLIS and its weights are under Microsoft's own
licence — check that repo before commercial use.
