# Pixal3D RunPod deployment

The worker is a third image-to-3D provider, alongside TRELLIS 2 and Hunyuan3D. The site defaults to Pixal3D after this deployment is validated. The admin Storefront model picker changes subsequent generations and can switch back to either old model. Existing assets and running jobs are unaffected. The older Automatic selection remains available for rollback.

## Image build

Published from `https://github.com/maximilianco/trellis-voxel`, with these files under `pixal3d/` and the supplied workflow at `.github/workflows/pixal3d.yml`. The workflow builds on pushes to `main` or `pixal3d-runpod` and publishes an immutable image:

```
ghcr.io/maximilianco/trellis-voxel:pixal3d-<full worker commit SHA>
```

The base is the successfully built TRELLIS.2 CUDA image `v2-01e1b90f4d897f0db04289ae1fa20977b8bd784e`. TRELLIS and Hunyuan tags are unchanged. Pixal source and model revisions, MoGe code and weights, DINO weights and NAF source are pinned. PyTorch remains 2.6/cu124 so inherited compiled extensions retain their ABI. NATTEN 0.21.0 builds CUDA kernels for Ampere/Ada; do not choose a Blackwell GPU for this image. Build failures are fatal. Weights are downloaded by the worker, rather than embedded in this image.

## New endpoint

RunPod Console → Serverless → New endpoint → Custom container. Select the immutable image above; use the repository's existing GHCR registry credentials if private. Alternatively deploy the GitHub repo with Dockerfile `pixal3d/Dockerfile` and build context `pixal3d`.

Initial configuration for validation:

| Setting | Value |
| --- | --- |
| Endpoint type | Queue based |
| GPU | L40S or A6000, 48 GB (Ampere/Ada) |
| GPUs per worker | 1 |
| Minimum workers | 0 |
| Maximum workers | 1 |
| Idle timeout | 30 seconds; temporarily longer during a comparison series |
| Execution timeout | 2,400 seconds |
| Container disk | 80 GB |
| Environment | `PIXAL3D_LOW_VRAM=true`, `PIXAL3D_MODEL_RESOLUTION=1024`, `ATTN_BACKEND=sdpa` |
| Start command | Leave blank; image CMD starts the handler |

48 GB is a conservative initial test allocation, not a measured minimum for our workload. Upstream estimates approximately 10–12 GB in low-VRAM mode and about 18 GB normally. Test 24 GB GPUs after recording our actual peak usage and times. Low-VRAM mode offloads models between stages. Multiple workers are independent processes and need no public HTTP port.

Cache these repositories using RunPod's model cache if available, or mount persistent storage and set `HF_HOME` and `TORCH_HOME` there:

* `TencentARC/Pixal3D` (single-view checkpoints about 24 GB; exclude `*_mv.*`)
* `camenduru/dinov3-vitl16-pretrain-lvd1689m`
* `Ruicheng/moge-2-vitl`

`TORCH_HOME` additionally caches NAF's released weights and source. A slow first download should be distinguished from inference. Probe does not load weights. Background cutout uses u2net already baked into the base image; the worker does not instantiate `briaai/RMBG-2.0`. Pixal3D is MIT; upstream dependency licences still apply, including the DINO mirror's original DINOv3 terms.

## Validate before switching

1. Submit `probe_input.json` to `/run`, then poll `/status/<job-id>`. All listed core modules must be present, GPU CUDA available, worker revision equal the image tag and NATTEN compiled kernels usable. Probe alone does not validate model inference.
2. Submit a known reference with `{"image_b64":"...","seed":2090003113,"resolution":512,"model_resolution":1024,"want_colour":true}`. `resolution` is the OUTPUT voxel grid; `model_resolution` is the 1024/1536 Pixal cascade. Do not pass TRELLIS's `pipeline_type:"512"`.
3. Compare the castle's saved prepared reference (`1410c0d8257ab5d14a55ee2b`) against raw colours and exports at identical game size. Check dark/pink regional fidelity, front/back orientation, geometry, inference time and actual cost. No user credits need to be charged for this standalone test. Do not overwrite its existing assets until reviewed.
4. Return native sparse PBR arrays without a GLB/remeshing/texture-baking pass. The game compiler uses the existing archive, interiors and all-format export pipeline. A 180-degree horizontal rotation converts Pixal's -Y/Z-up canonical front to the game convention; validate it with an asymmetric reference. `PIXAL3D_FRONT_ROTATION=false` disables this transform for comparison. Archives carry the marker, and the old TRELLIS-only mirror does not run on Pixal outputs.

## Site configuration and rollback

Once GPU validation passes, set `RUNPOD_PIXAL3D_ENDPOINT=<new id>` in `/etc/frombeyond/schematic-generator.env`, or `runpod_pixal3d_endpoint = <new id>` in the existing AI config. The existing `runpod_api_key` is reused. Install `pixal3d_provider.py`, the updated worker and server, and admin UI. Never point the old TRELLIS endpoint setting at a Pixal endpoint: automatic building selection would still send some jobs to Hunyuan.

The new server initializes the admin model selection to Pixal3D. Set `SCHEMATIC_RECONSTRUCTOR=auto` when staging these changes before the endpoint is ready. Admin selection is persisted in the database and overrides the environment after initialization. Select TRELLIS 2, Hunyuan3D or Automatic in Storefront settings to roll back. Each generation receives an explicit `--reconstructor` argument; a change cannot alter a worker already running. Replays use saved payloads without buying another 3D job.

Pixal failure raises a job error, rather than automatically purchasing a second reconstruction. Missing PBR channels also fail instead of publishing a monochrome build. Existing TRELLIS/Hunyuan fallback behaviour is unchanged when those routes are selected.

The currently configured production RunPod key returned HTTP 403 for endpoint management. Creating the new endpoint requires the Console or a key with management access; normal job submission can reuse the existing key after the endpoint is created.

Sources: [Pixal3D](https://github.com/TencentARC/Pixal3D), [inference code](https://github.com/TencentARC/Pixal3D/blob/master/inference.py), [NATTEN installation](https://natten.org/install/), [RunPod GitHub deployment](https://docs.runpod.io/serverless/workers/github-integration).
