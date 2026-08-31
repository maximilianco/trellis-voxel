"""RunPod serverless handler: image in, coloured voxel grid out.

Returns TRELLIS' own sparse-structure occupancy rather than a mesh. That stage
decodes to a 64^3 grid before any renderer runs, so asking for voxels is asking
for less work, not more -- and it avoids the mesh export path, which is where
the fragile CUDA extensions live.

Colour comes from the gaussian decoder: each gaussian has a position and a
spherical-harmonic DC term, which is its base colour. Splatting those onto the
occupancy grid gives a colour per filled voxel without rendering anything.

Input:
    {"image_b64": "...", "seed": 0, "resolution": 64,
     "sparse_steps": 12, "slat_steps": 12, "cfg": 7.5,
     "want_colour": true, "remove_background": true}
  or {"image_url": "https://..."}

Output:
    {"resolution": 64,
     "coords": [[x,y,z], ...],          # occupied voxels, grid coordinates
     "colours": [[r,g,b], ...],         # 0-255, aligned with coords (if available)
     "counts": {...}, "timings": {...}, "colour_source": "gaussian"|"none"}
"""
from __future__ import annotations

import base64
import io
import os
import sys
import time
import traceback

import numpy as np

os.environ.setdefault("ATTN_BACKEND", "xformers")
os.environ.setdefault("SPCONV_ALGO", "native")

# Runpod's cached-model feature puts weights at /runpod-volume/huggingface-cache
# in the standard HF layout. If HF_HOME points anywhere else, from_pretrained
# silently re-downloads several GB on every cold start instead of using the
# cache we asked for. Detect the mount at runtime so the same image works with
# or without a model attached.
_RUNPOD_CACHE = "/runpod-volume/huggingface-cache"
if os.path.isdir(os.path.join(_RUNPOD_CACHE, "hub")):
    os.environ["HF_HOME"] = _RUNPOD_CACHE
    print(f"using Runpod model cache at {_RUNPOD_CACHE}", flush=True)
else:
    os.environ.setdefault("HF_HOME", "/workspace/hf")
    print("no Runpod model cache mounted; weights will download on first use", flush=True)

if os.path.isdir("/workspace/TRELLIS") and "/workspace/TRELLIS" not in sys.path:
    sys.path.insert(0, "/workspace/TRELLIS")

import runpod  # noqa: E402
from PIL import Image  # noqa: E402

# A repo id resolves through the HF cache, which is what Runpod's
# --model-reference populates host-side. A local path also works if the
# image was built with BAKE_WEIGHTS=1.
MODEL_PATH = os.environ.get("TRELLIS_MODEL", "microsoft/TRELLIS-image-large")
_PIPELINE = None


def pipeline():
    """Load once per worker; serverless keeps the process warm between jobs."""
    global _PIPELINE
    if _PIPELINE is None:
        import json
        import torch
        import trellis.pipelines as pipelines

        # TRELLIS-1 and TRELLIS.2 use different pipeline classes but both decode
        # to the same 64^3 sparse structure, so the voxel path is shared. Read
        # the name out of pipeline.json rather than guessing from the path.
        name = "TrellisImageTo3DPipeline"
        config = os.path.join(MODEL_PATH, "pipeline.json")
        if os.path.exists(config):
            with open(config, encoding="utf-8") as handle:
                name = json.load(handle).get("name", name)
        cls = getattr(pipelines, name, None) or pipelines.TrellisImageTo3DPipeline
        print(f"loading {name} from {MODEL_PATH}", flush=True)
        _PIPELINE = cls.from_pretrained(MODEL_PATH)
        if torch.cuda.is_available():
            _PIPELINE.cuda()
    return _PIPELINE


def _load_image(payload: dict) -> Image.Image:
    if payload.get("image_b64"):
        raw = base64.b64decode(payload["image_b64"])
    elif payload.get("image_url"):
        import urllib.request

        with urllib.request.urlopen(payload["image_url"], timeout=60) as response:
            raw = response.read()
    else:
        raise ValueError("provide image_b64 or image_url")
    return Image.open(io.BytesIO(raw)).convert("RGBA")


def generate_voxels(image, seed, resolution, want_colour, sparse_steps, slat_steps, cfg):
    """Drive the pipeline stages by hand to keep the occupancy grid.

    `TrellisImageTo3DPipeline.run()` computes the sparse structure and then
    discards it -- it returns only decoded formats (mesh/gaussian/radiance
    field). The 64^3 occupancy grid we actually want is the *output of
    sample_sparse_structure*, so the stages are called directly:

        cond   = get_cond([image])
        coords = sample_sparse_structure(cond, 1, params)   # (N, 4) [batch,x,y,z]
        slat   = sample_slat(cond, coords, params)          # only if colour wanted
        out    = decode_slat(slat, ["gaussian"])            # colour source

    Skipping the slat stage when colour is not requested also roughly halves the
    runtime, because that is the expensive half.
    """
    import torch

    pipe = pipeline()
    cond = pipe.get_cond([image])
    torch.manual_seed(seed)
    coords = pipe.sample_sparse_structure(
        cond, 1, {"steps": sparse_steps, "cfg_strength": cfg},
    )
    # argwhere gives [batch, x, y, z]; drop the batch column.
    voxels = coords[:, 1:].detach().cpu().to(torch.int32).numpy()

    outputs = None
    if want_colour:
        slat = pipe.sample_slat(cond, coords, {"steps": slat_steps, "cfg_strength": 3.0})
        outputs = pipe.decode_slat(slat, ["gaussian"])
    return voxels, outputs


def _colours_for(coords: np.ndarray, outputs, resolution: int):
    """Colour each occupied voxel, and record how opaque it is.

    Opacity is the one material signal TRELLIS-1 actually has. It carries no
    semantic or PBR channel -- it models radiance, not materials -- but every
    gaussian has a learned alpha, and a surface the model reconstructed as
    see-through is a surface it believes you can see through. That is the
    closest thing to "this is glass" the representation contains, and we were
    throwing it away by reading only position and the SH DC term.

    (TRELLIS.2 does emit real PBR including per-surface transparency, but only
    through its mesh/O-Voxel path, which is a different and much heavier route
    than the sparse-structure grid this worker returns.)
    """
    import torch

    gaussian = (outputs or {}).get("gaussian")
    if not gaussian:
        return None, None, "none"
    item = gaussian[0] if isinstance(gaussian, (list, tuple)) else gaussian
    positions = getattr(item, "get_xyz", None)
    if positions is None:
        positions = getattr(item, "_xyz", None)
    features = getattr(item, "get_features", None)
    if features is None:
        features = getattr(item, "_features_dc", None)
    if positions is None or features is None:
        return None, None, "none"
    alpha = getattr(item, "get_opacity", None)
    if alpha is None:
        alpha = getattr(item, "_opacity", None)

    positions = positions.detach().float().cpu()
    features = features.detach().float().cpu().reshape(positions.shape[0], -1)[:, :3]
    # SH DC -> linear RGB, the standard 3DGS convention.
    rgb = (features * 0.28209479177387814 + 0.5).clamp(0, 1)

    # TRELLIS works in a [-0.5, 0.5] cube; map gaussians into grid coordinates.
    grid = ((positions + 0.5) * resolution).round().to(torch.int64).clamp(0, resolution - 1)
    flat = (grid[:, 0] * resolution + grid[:, 1]) * resolution + grid[:, 2]

    total = resolution ** 3
    accumulator = torch.zeros((total, 3), dtype=torch.float32)
    weights = torch.zeros(total, dtype=torch.float32)
    alphas = torch.zeros(total, dtype=torch.float32)
    accumulator.index_add_(0, flat, rgb)
    weights.index_add_(0, flat, torch.ones(flat.shape[0]))
    if alpha is not None:
        alphas.index_add_(0, flat, alpha.detach().float().cpu().reshape(-1).clamp(0, 1))

    voxel = torch.from_numpy(coords.astype(np.int64))
    keys = (voxel[:, 0] * resolution + voxel[:, 1]) * resolution + voxel[:, 2]
    counted = weights[keys].clamp_min(1e-6)
    colours = (accumulator[keys] / counted[:, None])
    missing = weights[keys] < 0.5
    if bool(missing.any()):
        # A voxel with no gaussian in it takes the mean colour rather than black.
        present = ~missing
        fallback = colours[present].mean(0) if bool(present.any()) else torch.tensor([0.6, 0.6, 0.6])
        colours[missing] = fallback
    opacity = None
    if alpha is not None:
        # A voxel with no gaussian is assumed solid rather than transparent.
        opacity = (alphas[keys] / counted).clamp(0, 1)
        opacity[missing] = 1.0
        opacity = (opacity * 255).round().to(torch.uint8).numpy()
    return (colours * 255).round().clamp(0, 255).to(torch.uint8).numpy(), opacity, "gaussian"


def handler(event):
    payload = (event or {}).get("input") or {}
    started = time.time()
    timings = {}
    # Probing over the API rather than `docker run --gpus` matters: the machine
    # driving this may not have a CUDA GPU, and it does not need one.
    if payload.get("probe"):
        return {"probe": probe(), "model_path": MODEL_PATH}
    try:
        resolution = int(payload.get("resolution", 64))
        seed = int(payload.get("seed", 0))
        want_colour = bool(payload.get("want_colour", True))

        image = _load_image(payload)
        if payload.get("remove_background", True):
            try:
                image = pipeline().preprocess_image(image)
            except Exception:
                pass  # an already-cut image with alpha is fine as-is
        timings["load"] = round(time.time() - started, 2)

        mark = time.time()
        coords, outputs = generate_voxels(
            image, seed, resolution, want_colour,
            int(payload.get("sparse_steps", 12)),
            int(payload.get("slat_steps", 12)),
            float(payload.get("cfg", 7.5)),
        )
        timings["generate"] = round(time.time() - mark, 2)

        if coords is None or not len(coords):
            return {"error": "sparse structure decoded to zero occupied voxels",
                    "hint": "the subject may have been removed by background cutout"}

        mark = time.time()
        colours, opacity, colour_source = (None, None, "none")
        if want_colour:
            try:
                colours, opacity, colour_source = _colours_for(coords, outputs, resolution)
            except Exception:
                colours, opacity, colour_source = None, None, "failed"
        timings["colour"] = round(time.time() - mark, 2)
        timings["total"] = round(time.time() - started, 2)

        return {
            "resolution": resolution,
            "coords": coords.astype(int).tolist(),
            "colours": colours.astype(int).tolist() if colours is not None else None,
            "opacity": opacity.astype(int).tolist() if opacity is not None else None,
            "colour_source": colour_source,
            "counts": {"voxels": int(len(coords))},
            "timings": timings,
        }
    except Exception as exc:  # noqa: BLE001 - a failed job must report, not vanish
        return {"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(limit=6)}


def probe(event=None):
    """Report which optional extensions built, so a failure is diagnosable."""
    import importlib

    status = {}
    for module in ("torch", "xformers", "flash_attn", "spconv", "kaolin",
                   "nvdiffrast", "diffoctreerast",
                   "diff_gaussian_rasterization", "trellis"):
        try:
            loaded = importlib.import_module(module)
            status[module] = getattr(loaded, "__version__", "present")
        except Exception as exc:
            # Report *what* was missing. Recording only the exception type once
            # made a missing `easydict` read as "trellis is not installed",
            # which cost a full build cycle to work out.
            status[module] = f"MISSING ({type(exc).__name__}: {exc})"
    try:
        import torch

        status["cuda"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu only"
    except Exception:
        pass
    # If the rasterizer is absent, surface the build log rather than leaving the
    # cause on a machine nobody can log into.
    if "MISSING" in str(status.get("diff_gaussian_rasterization", "")):
        try:
            with open("/workspace/rasterizer_build.log", encoding="utf-8", errors="replace") as handle:
                tail = [line.rstrip() for line in handle if line.strip()][-12:]
            status["rasterizer_build_log"] = tail
        except OSError:
            status["rasterizer_build_log"] = "no log (layer predates logging)"
    status["hf_home"] = os.environ.get("HF_HOME")
    status["runpod_model_cache"] = os.path.isdir(os.path.join(_RUNPOD_CACHE, "hub"))
    hub = os.path.join(os.environ.get("HF_HOME", ""), "hub")
    status["cached_models"] = sorted(os.listdir(hub))[:8] if os.path.isdir(hub) else []
    return status


if __name__ == "__main__":
    if os.environ.get("TRELLIS_PROBE"):
        import json

        print(json.dumps(probe(), indent=2))
    else:
        runpod.serverless.start({"handler": handler})
