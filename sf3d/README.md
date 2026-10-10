# Stable Fast 3D RunPod worker

Isolated additional reconstruction option. Does not replace TRELLIS, Hunyuan
or Pixal3D, and never triggers a second model job on rejection or failure.

Source: Stability-AI/stable-fast-3d at ff21fc491b4dc5314bf6734c7c0dabd86b5f5bb2.
Weights: https://huggingface.co/stabilityai/stable-fast-3d at
f0c9a8ffd62cb1bbc8a7a53c9f87a0be1b6be778 (4,024,289,892 bytes).
Accept the gated model conditions with the Hugging Face account whose token is
used for the worker. The Stability AI Community License applies; check its
commercial revenue threshold before use.

## RunPod settings

- CUDA 12.4-compatible Ampere/Ada GPU; initial comparison uses 24 GB.
- No low-VRAM setting, no model offloading, no remeshing.
- 30 GB container disk for CUDA/PyTorch, public auxiliary caches and gated weights.
- Min workers 0, max 1 during validation; 5 s idle timeout; 600 s execution timeout.
- Secret environment variable `HF_TOKEN`: read token with access to this model.
- Default `SF3D_TEXTURE_RESOLUTION=1024`.
- Cached model field: `https://huggingface.co/stabilityai/stable-fast-3d`.
  Supply the gated model token in RunPod's model-cache settings too.
  The worker reads the exact pinned snapshot from
  `/runpod-volume/huggingface-cache/hub/models--stabilityai--stable-fast-3d/snapshots/f0c9a8ffd62cb1bbc8a7a53c9f87a0be1b6be778`.
  Runtime HF_HOME remains `/workspace/model-cache`, containing baked auxiliary
  models. Without that RunPod model cache, the gated weights download once per
  container. To use a prepopulated snapshot, set `SF3D_MODEL_PATH` to the
  directory containing config.yaml and model.safetensors.

## Request

```json
{"input":{"image_b64":"BASE64_PNG_OR_JPEG","texture_resolution":1024,"foreground_ratio":0.85}}
```

RGBA input keeps its existing transparency; otherwise u2net removes background
on CPU. Square padding preserves aspect ratio and the standard SF3D 85%
foreground framing. One textured GLB is returned in `glb_b64`; output also
contains the actual prepared image, mesh statistics, peak VRAM, GPU name and
timings split into model loading, image preparation, mesh/texture and export.
Model speed claims exclude cold container/model downloads and subsequent block
conversion. The site can voxelize the GLB on CPU after releasing the GPU worker.

Native output is Y-up with the conditioning camera on -Z. The site adapter
keeps this front direction and does not reuse Pixal's +Z-camera rotation.
Textures are sampled from the embedded GLB atlas; no flat material fallback.

## Builds

The full Dockerfile compiles the pinned upstream dependencies. Handler-only
changes use Dockerfile.runtime with the first published dependency image.
GitHub builds detect changes to Dockerfile/constraints.txt, and a manual
`full_build=true` also recompiles dependencies. After a dependency update,
update Dockerfile.runtime's pinned base to the newly verified release.
Huge optional GitHub Actions build-cache exports are disabled; dependency
reuse happens through the pinned worker image instead.
