# trellis-voxel

RunPod serverless workers for the BlockGen project. The repo is named for the
first one; it now holds three, built by two independent workflows.

| worker | image tag | what it does |
|---|---|---|
| TRELLIS / TRELLIS.2 | `:v1` `:v2` | image to coloured voxel grid — root `Dockerfile*`, `handler.py` |
| Hunyuan3D-2.1 | `:hunyuan3d` | image to textured glb — [`hunyuan3d/`](hunyuan3d/) |
| SkinTokens | `:rig` | glb to auto-rigged glb — [`rig/`](rig/) |

The last two are the avatar pipeline: a prompt becomes a rigged Luanti player
model. See [PIPELINE.md](PIPELINE.md) for how the stages fit together, the GPU
each endpoint needs, and why the handoff between them goes over a network
volume rather than through the response body.

## TRELLIS: image to voxels

A [RunPod serverless](https://runpod.io) worker that turns an image into a
coloured voxel grid using [TRELLIS](https://github.com/microsoft/TRELLIS) or
[TRELLIS.2](https://github.com/microsoft/TRELLIS.2).

Send it a picture, get back occupied voxel coordinates with a colour — and, on
TRELLIS.2, real PBR materials — per voxel. No mesh, no renderer.

## Why it returns voxels rather than a mesh

Both generations are voxel-based internally.

**TRELLIS-1's** first stage — `ss_flow_img_dit_L_16l8` into `ss_dec_conv3d_16l8`
— decodes to a **64³ occupancy grid** before any mesh, gaussian or
radiance-field decoder runs. Reading voxels out is reading the pipeline's own
intermediate representation, not post-processing a mesh. That also means the
fragile CUDA extensions used only for mesh export (`nvdiffrast`, `kaolin`,
`diffoctreerast`) are optional, and the v1 Dockerfile treats them as such.

**TRELLIS.2's** O-Voxel output *is* a sparse voxel grid: `mesh.coords` with a
per-voxel `mesh.attrs`. This is the better fit and the reason to move.

## The two generations

They cannot share an image — TRELLIS-1 links against torch 2.4/cu121 and
TRELLIS.2 against torch 2.6/cu124, and every compiled extension is bound to one
exact torch build. So there are two Dockerfiles and two tags, and **the switch is
which tag the endpoint runs**:

| | tag | Dockerfile | GPU | materials |
|---|---|---|---|---|
| TRELLIS-1 | `:v1` | `Dockerfile` | 16 GB is enough | colour + alpha only |
| TRELLIS.2 | `:v2` | `Dockerfile.v2` | **24 GB required** | base colour, metallic, roughness, alpha |

`handler.py` is shared and detects which generation is installed by import,
rather than trusting `TRELLIS_VERSION` — an endpoint can be pointed at the wrong
image, and importability is the fact.

To roll back, point the endpoint at `:v1` (or the `trellis-1-working` tag in git
history) and recycle the workers.

### What the material channel is worth

TRELLIS-1 has **no material channel**. It reconstructs radiance, not materials.
The closest thing it carries is each gaussian's learned opacity — a surface the
model rebuilt as see-through is one it thinks you can see through — so the v1
path exports that as `opacity`.

Inferring glass from brightness instead does not work, and that is measured, not
assumed: on a generated stone cottage the wall band's luma spans 84–102 against a
median of 95, under a tenth of a stop, and the only genuinely dark clusters are
the eave shadow and the ground shadow.

TRELLIS.2 predicts `base_color`, `metallic`, `roughness` and `alpha` per voxel
directly, so glass and metal stop being guesses.

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
    "trellis_version": "2",
    "source_resolution": 256,
    "coords": [[31, 20, 33]],
    "colours": [[194, 142, 90]],
    "opacity": [255],
    "metallic": [12],
    "roughness": [180],
    "colour_source": "pbr",
    "counts": { "voxels": 18342 },
    "timings": { "generate": 6.1, "total": 7.4 }
} }
```

`metallic` and `roughness` are `null` on TRELLIS-1. `colour_source` is `"pbr"`,
`"gaussian"` or `"none"`.

`resolution` is the grid you asked for; `source_resolution` is what the model
decoded at. TRELLIS.2 can decode at up to 1024³, which as JSON would be hundreds
of megabytes and would blow past RunPod's response limit — so the worker
aggregates down to `resolution` before replying, averaging attributes within
each cell.

`{"input": {"probe": true}}` returns which optional CUDA extensions actually
built, plus the GPU and its VRAM — useful because you cannot check that without
a GPU. On a v2 worker with under 23 GB it also returns a `vram_warning`, which
turns a baffling mid-job OOM into one readable line.

## Deploy

Push this repo and the included workflow builds **both** images and publishes
them to GHCR. Then create a RunPod serverless endpoint from the tag you want.

There is deliberately **no `:latest`**. RunPod does not roll workers when a
moving tag changes, so an ambiguous tag turns "did my fix deploy?" into a guess.
Pin the immutable tag instead:

```
ghcr.io/<you>/trellis-voxel:v2-<commit sha>
```

| setting | v1 | v2 |
|---|---|---|
| GPU | RTX A4000 16 GB | 24 GB (A5000 / L4 / 4090 / A100) |
| Active workers | 0 — nothing is billed while idle | 0 |
| Max workers | 1 | 1 |
| Container disk | 25 GB | 40 GB |

Weights are **not** baked into either image. Attach them host-side instead:

```
--model-reference microsoft/TRELLIS-image-large     # v1
--model-reference microsoft/TRELLIS.2-4B            # v2
```

Smaller image, faster cold starts, and no monthly storage cost. Build with
`--build-arg BAKE_WEIGHTS=1` if you would rather embed them.

| | weights | GPU |
|---|---|---|
| `microsoft/TRELLIS-image-large` | 3.3 GB | 16 GB is comfortable |
| `microsoft/TRELLIS.2-4B` | 16.2 GB | 24 GB, per Microsoft; verified on A100/H100 |

`TORCH_CUDA_ARCH_LIST` targets sm_80/86/89/90 in both images, which covers the
usual RunPod serverless GPUs. It does **not** include sm_120 (Blackwell) — for a
5090-class worker, add `12.0` and rebuild.

## Licence

The worker glue here is MIT. TRELLIS and its weights are under Microsoft's own
licence — check that repo before commercial use.
