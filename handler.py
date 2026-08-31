"""RunPod serverless handler: image in, coloured voxel grid out.

Supports both generations, selected by TRELLIS_VERSION (default: whichever is
installed). They are NOT interchangeable at runtime -- TRELLIS-1 wants torch
2.4/cu121 and TRELLIS.2 wants torch 2.6/cu124 -- so the version is fixed when
the image is built and the switch is which image the endpoint runs. The handler
auto-detects rather than trusting the variable, so a mismatched env var cannot
silently produce a broken worker.

TRELLIS-1 returns the sparse-structure occupancy grid: that stage decodes to
64^3 before any renderer runs, so asking for voxels is asking for less work than
asking for a mesh. Colour comes from the gaussian decoder's spherical-harmonic
DC term, and transparency from each gaussian's learned alpha. There is no
material channel -- the model reconstructs radiance, not materials.

TRELLIS.2 is the better fit for voxels and the reason to move: its O-Voxel
output IS a sparse voxel grid carrying real PBR, so `attrs` gives base colour,
metallic, roughness AND alpha per voxel with no splatting and no inference from
brightness. Glass stops being a guess.

Input:
    {"image_b64": "...", "seed": 0, "resolution": 64,
     "sparse_steps": 12, "slat_steps": 12, "cfg": 7.5,
     "want_colour": true, "remove_background": true}
  or {"image_url": "https://..."}

Output:
    {"resolution": 64, "trellis_version": "1"|"2",
     "coords": [[x,y,z], ...],          # occupied voxels, grid coordinates
     "colours": [[r,g,b], ...],         # 0-255, aligned with coords
     "opacity": [a, ...],               # 0-255; low means see-through
     "metallic": [...], "roughness": [...],   # 0-255, TRELLIS.2 only
     "counts": {...}, "timings": {...},
     "colour_source": "pbr"|"gaussian"|"none"}
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
# TRELLIS.2 allocates in large, uneven blocks and the OOM we hit reported
# 3.87 GiB reserved but unallocated -- fragmentation, not genuine exhaustion.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

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

for _checkout in ("/workspace/TRELLIS", "/workspace/TRELLIS.2"):
    if os.path.isdir(_checkout) and _checkout not in sys.path:
        sys.path.insert(0, _checkout)

import runpod  # noqa: E402
from PIL import Image  # noqa: E402

_DEFAULT_MODEL = {"1": "microsoft/TRELLIS-image-large", "2": "microsoft/TRELLIS.2-4B"}
_PIPELINE = None
_VERSION = None


def version() -> str:
    """Which generation this image actually contains.

    Detected, not declared: TRELLIS_VERSION is a build-time intention and an
    endpoint can be pointed at the wrong image. Importability is the fact.
    """
    global _VERSION
    if _VERSION is None:
        import importlib.util

        wanted = str(os.environ.get("TRELLIS_VERSION", "")).strip()
        has2 = importlib.util.find_spec("trellis2") is not None
        has1 = importlib.util.find_spec("trellis") is not None
        if wanted in ("1", "2") and (has2 if wanted == "2" else has1):
            _VERSION = wanted
        else:
            _VERSION = "2" if has2 else "1"
            if wanted and wanted != _VERSION:
                print(f"TRELLIS_VERSION={wanted} requested but only "
                      f"{'trellis2' if has2 else 'trellis'} is installed; "
                      f"using {_VERSION}", flush=True)
    return _VERSION


def model_path() -> str:
    return os.environ.get("TRELLIS_MODEL") or _DEFAULT_MODEL[version()]


def pipeline():
    """Load once per worker; serverless keeps the process warm between jobs."""
    global _PIPELINE
    if _PIPELINE is None:
        import json
        import torch

        path = model_path()
        if version() == "2":
            from trellis2.pipelines import Trellis2ImageTo3DPipeline

            # from_pretrained builds the background remover unconditionally:
            #     pipeline.rembg_model = getattr(rembg, args['rembg_model']['name'])(...)
            # and that name is BiRefNet, i.e. briaai/RMBG-2.0 -- gated, and
            # CC BY-NC, non-commercial only. So the pipeline cannot even LOAD
            # without agreeing to a licence we may not want, however the images
            # are cut out later.
            #
            # __init__ already accepts rembg_model=None, so the model is optional
            # to the object; only the loader insists. Stubbing the class keeps
            # that download from ever happening, and _cutout() supplies the alpha
            # channel with u2net instead, which sends preprocess_image down its
            # has_alpha branch where rembg_model is never touched.
            #
            # Set TRELLIS_ALLOW_RMBG=1 to use the real one, having accepted BRIA's
            # terms and checked they suit your use.
            if os.environ.get("TRELLIS_ALLOW_RMBG", "0") != "1":
                from trellis2.pipelines import rembg as _rembg

                class _UnusedRembg:
                    """Stands in for BiRefNet so from_pretrained does not fetch it."""

                    def __init__(self, *args, **kwargs):
                        pass

                    def __call__(self, *args, **kwargs):
                        raise RuntimeError(
                            "background removal reached the stubbed RMBG-2.0: the "
                            "image arrived without an alpha channel. Check that "
                            "rembg/u2net is installed, or set TRELLIS_ALLOW_RMBG=1 "
                            "after accepting briaai/RMBG-2.0's non-commercial terms")

                _rembg.BiRefNet = _UnusedRembg
                print("RMBG-2.0 stubbed; using u2net for cutout "
                      "(set TRELLIS_ALLOW_RMBG=1 to use BiRefNet)", flush=True)

            # from_pretrained loads every model in pipeline.json, and on a cold
            # worker that read is 62% of the whole job (130s of 210s, measured).
            # We run the 512 pipeline, so the two 1024 DiTs -- 1.3B parameters
            # each, about 5 GB of the 16 GB -- are loaded and never used. The
            # base loader honours a `model_names_to_load` whitelist.
            if (os.environ.get("TRELLIS_PIPELINE_TYPE") or "512") == "512":
                Trellis2ImageTo3DPipeline.model_names_to_load = [
                    "sparse_structure_decoder", "sparse_structure_flow_model",
                    "shape_slat_decoder", "shape_slat_flow_model_512",
                    "tex_slat_decoder", "tex_slat_flow_model_512",
                ]
                print("loading 512 models only (skipping the 1024 DiTs)", flush=True)
            print(f"loading Trellis2ImageTo3DPipeline from {path}", flush=True)
            _PIPELINE = Trellis2ImageTo3DPipeline.from_pretrained(path)
        else:
            import trellis.pipelines as pipelines

            # Read the class name out of pipeline.json rather than guessing it
            # from the path.
            name = "TrellisImageTo3DPipeline"
            config = os.path.join(path, "pipeline.json")
            if os.path.exists(config):
                with open(config, encoding="utf-8") as handle:
                    name = json.load(handle).get("name", name)
            cls = getattr(pipelines, name, None) or pipelines.TrellisImageTo3DPipeline
            print(f"loading {name} from {path}", flush=True)
            _PIPELINE = cls.from_pretrained(path)
        if torch.cuda.is_available():
            # Pipeline.cuda() does `for model in self.models.values(): model.to(device)`
            # -- every model resident at once. TRELLIS.2 ships low_vram=True and
            # moves each model to self.device for its stage and back to cpu after,
            # so making them all resident defeats that and is how a 4B model OOMed
            # a 24 GB card with 22.4 GiB in use. The device property honours an
            # explicit _device, so point the pipeline at CUDA and let it do the
            # staging itself.
            if version() == "2" and getattr(_PIPELINE, "low_vram", False):
                _PIPELINE._device = torch.device("cuda")
                print("v2 low_vram: models stay on CPU and move per stage", flush=True)
            else:
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


def _cutout(image: "Image.Image") -> "Image.Image":
    """Give the image a real alpha channel using the ungated u2net.

    TRELLIS.2's own preprocess_image reaches for BiRefNet (briaai/RMBG-2.0),
    which is BOTH gated on HuggingFace and CC BY-NC -- non-commercial only.
    But it only does so when the input has no meaningful alpha:

        if has_alpha: output = input
        else:         output = self.rembg_model(input)

    So cutting the background out first means that model is never called. rembg's
    u2net is already in the image and carries no such restriction, and the
    geometry work we still want -- square crop about the subject, alpha
    premultiply -- happens in preprocess_image either way.
    """
    import numpy as np

    if image.mode == "RGBA" and not np.all(np.array(image)[:, :, 3] == 255):
        return image                      # already cut out; nothing to do
    from rembg import remove

    return remove(image.convert("RGBA"))


def reduce_to(coords: np.ndarray, attrs, source: int, target: int):
    """Aggregate a fine voxel grid down to the resolution the caller asked for.

    TRELLIS.2 decodes at up to 1024^3. Returning that as JSON would be hundreds
    of megabytes and blow past Runpod's response limit, and the schematic is a
    few dozen blocks across regardless -- so the reduction belongs on the GPU
    box, not on the far side of the wire. Attributes are averaged within a cell,
    which is right for colour and for alpha alike.
    """
    if source <= target:
        return coords, attrs, np.ones(len(coords), dtype=np.int32)
    binned = np.floor(coords.astype(np.float64) * (target / source)).astype(np.int64)
    binned = np.clip(binned, 0, target - 1)
    key = (binned[:, 0] * target + binned[:, 1]) * target + binned[:, 2]
    unique, inverse = np.unique(key, return_inverse=True)
    inverse = inverse.reshape(-1)
    counts = np.bincount(inverse).astype(np.float64)
    out = np.stack([unique // (target * target),
                    (unique // target) % target,
                    unique % target], axis=1).astype(np.int32)
    # How many source voxels fell in each output cell. Occupancy alone is
    # "any voxel present", which inflates thin structure: a ship's rigging is
    # one or two voxels thick at 512 and becomes a solid block at 64, turning a
    # 7.7%-fill subject into a 19% lump. The count lets the caller tell a wall
    # from a rope.
    density = counts.astype(np.int32)
    if attrs is None:
        return out, None, density
    averaged = np.stack(
        [np.bincount(inverse, weights=attrs[:, channel].astype(np.float64)) / counts
         for channel in range(attrs.shape[1])], axis=1)
    return out, averaged, density


def _to_numpy(value, dtype=None):
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu()
        if dtype is not None:
            value = value.to(dtype)
        return value.numpy()
    return np.asarray(value)


def _generate_v1(image, seed, resolution, want_colour, sparse_steps, slat_steps, cfg):
    """Drive the stages by hand to keep the occupancy grid.

    `TrellisImageTo3DPipeline.run()` computes the sparse structure and then
    discards it -- it returns only decoded formats. The 64^3 grid we want is the
    *output of sample_sparse_structure*, so the stages are called directly.
    Skipping the slat stage when colour is not wanted roughly halves the runtime.
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
    colours, opacity, source = (None, None, "none")
    if outputs is not None:
        try:
            colours, opacity, source = _gaussian_appearance(voxels, outputs, resolution)
        except Exception:
            colours, opacity, source = None, None, "failed"
    return {"coords": voxels, "colours": colours, "opacity": opacity,
            "metallic": None, "roughness": None,
            "density": np.ones(len(voxels), dtype=np.int32),
            "source_resolution": resolution, "colour_source": source}


def _generate_v2(image, seed, resolution, want_colour, sparse_steps, slat_steps, cfg,
                 pipeline_type=None):
    """TRELLIS.2: the O-Voxel output is already a voxel grid with materials.

    `run()` returns MeshWithVoxel, whose `coords`/`attrs` are the sparse voxel
    grid and its per-voxel PBR. `layout` gives the channel slices --
    base_color 0:3, metallic 3:4, roughness 4:5, alpha 5:6 -- all in 0..1. No
    gaussian splatting, and transparency is a property the model actually
    predicted rather than something inferred from how dark a pane looks.
    """
    import torch

    pipe = pipeline()
    # TRELLIS.2's samplers are FlowEulerGuidanceIntervalSampler and take
    # `guidance_strength`; `cfg_strength` is TRELLIS-1's name for it. An unknown
    # key is not rejected here -- it is forwarded into the flow model, so the
    # wrong name surfaces three minutes in as
    #     SparseStructureFlowModel.forward() got an unexpected keyword argument
    # rather than as a bad argument at the call site.
    # Default to the 512 pipeline rather than the config's 1024_cascade. Not only
    # to fit the card: the result is reduced to a 64-cube for the schematic, so a
    # 512 decode is already eight times finer than anything that survives, and
    # the cascade was spending a second 1024 pass on detail we discard.
    kind = pipeline_type or os.environ.get("TRELLIS_PIPELINE_TYPE") or "512"
    mesh = pipe.run(
        image,
        seed=seed,
        sparse_structure_sampler_params={"steps": sparse_steps,
                                         "guidance_strength": cfg},
        preprocess_image=False,     # already cut out by the caller
        pipeline_type=kind,
    )[0]

    coords = _to_numpy(mesh.coords, torch.int32).astype(np.int32)
    voxel_size = float(getattr(mesh, "voxel_size", 0) or 0)
    source_resolution = int(round(1.0 / voxel_size)) if voxel_size > 0 else resolution

    attrs = _to_numpy(getattr(mesh, "attrs", None), torch.float32)
    if attrs is None or not want_colour:
        coords, _, density = reduce_to(coords, None, source_resolution, resolution)
        return {"coords": coords, "colours": None, "opacity": None,
                "metallic": None, "roughness": None, "density": density,
                "source_resolution": source_resolution, "colour_source": "none"}

    layout = getattr(mesh, "layout", None) or {
        "base_color": slice(0, 3), "metallic": slice(3, 4),
        "roughness": slice(4, 5), "alpha": slice(5, 6),
    }
    coords, attrs, density = reduce_to(coords, attrs, source_resolution, resolution)

    def channel(name):
        span = layout.get(name)
        if span is None or attrs.shape[1] < (span.stop or 0):
            return None
        return np.clip(attrs[:, span] * 255.0, 0, 255).round().astype(np.uint8)

    colours, alpha = channel("base_color"), channel("alpha")
    metallic, roughness = channel("metallic"), channel("roughness")
    return {
        "coords": coords,
        "colours": colours,
        "opacity": None if alpha is None else alpha[:, 0],
        "metallic": None if metallic is None else metallic[:, 0],
        "roughness": None if roughness is None else roughness[:, 0],
        "density": density,
        "source_resolution": source_resolution,
        "colour_source": "pbr" if colours is not None else "none",
    }


def _gaussian_appearance(coords: np.ndarray, outputs, resolution: int):
    """Colour each occupied voxel, and record how opaque it is.

    Opacity is the only material signal TRELLIS-1 carries. It has no semantic or
    PBR channel -- it models radiance, not materials -- but every gaussian has a
    learned alpha, and a surface the model reconstructed as see-through is one it
    believes you can see through. That is the closest thing to "this is glass"
    the representation contains, and reading only position and the SH DC term
    threw it away.
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


def _listify(value):
    return None if value is None else np.asarray(value).astype(int).tolist()


# Above this many voxels the JSON form is both enormous and slow to build, and
# Runpod's response limit is a hard wall. A 384-block castle needs a 256 grid,
# which is roughly 700k voxels -- about 90 MB as JSON lists, ~8 MB packed.
PACK_THRESHOLD = 150_000


def _pack(arrays: dict) -> str:
    """Compress the voxel arrays into one base64 blob.

    JSON lists of integers cost roughly ten bytes per value. The same data as
    little-endian binary under zlib is an order of magnitude smaller, which is
    the difference between a 256-grid fitting in a response and not.
    """
    import base64
    import io
    import zlib

    buffer = io.BytesIO()
    np.savez(buffer, **{k: v for k, v in arrays.items() if v is not None})
    return base64.b64encode(zlib.compress(buffer.getvalue(), 6)).decode("ascii")


def handler(event):
    payload = (event or {}).get("input") or {}
    started = time.time()
    timings = {}
    # Probing over the API rather than `docker run --gpus` matters: the machine
    # driving this may not have a CUDA GPU, and it does not need one.
    if payload.get("probe"):
        return {"probe": probe(), "model_path": model_path(), "trellis_version": version()}
    try:
        resolution = int(payload.get("resolution", 64))
        seed = int(payload.get("seed", 0))
        want_colour = bool(payload.get("want_colour", True))

        image = _load_image(payload)
        if payload.get("remove_background", True):
            # Cut out first, so preprocess_image takes its has_alpha branch and
            # never loads the gated, non-commercial RMBG-2.0.
            try:
                image = _cutout(image)
            except Exception as exc:
                print(f"cutout unavailable ({type(exc).__name__}: {exc}); "
                      f"falling back to the pipeline's own remover", flush=True)
            try:
                image = pipeline().preprocess_image(image)
            except Exception as exc:
                # Do not swallow this. Skipping it entirely leaves a white
                # background in the frame, and white background becomes geometry.
                print(f"WARNING: preprocess_image failed ({type(exc).__name__}: "
                      f"{exc}); the subject may not be centred or cut out", flush=True)
                timings["preprocess_error"] = f"{type(exc).__name__}: {exc}"
        timings["load"] = round(time.time() - started, 2)

        mark = time.time()
        extra = {}
        if version() == "2":
            extra["pipeline_type"] = payload.get("pipeline_type")
        generate = _generate_v2 if version() == "2" else _generate_v1
        result = generate(
            image, seed, resolution, want_colour,
            int(payload.get("sparse_steps", 12)),
            int(payload.get("slat_steps", 12)),
            float(payload.get("cfg", 7.5)),
            **extra,
        )
        timings["generate"] = round(time.time() - mark, 2)

        coords = result["coords"]
        if coords is None or not len(coords):
            return {"error": "decoded to zero occupied voxels",
                    "hint": "the subject may have been removed by background cutout"}
        timings["total"] = round(time.time() - started, 2)

        payload_out = {
            "resolution": resolution,
            "trellis_version": version(),
            "source_resolution": result.get("source_resolution"),
            "colour_source": result.get("colour_source"),
            "counts": {"voxels": int(len(coords))},
            "timings": timings,
        }
        fields = {"coords": coords.astype(np.int32),
                  "colours": result.get("colours"), "opacity": result.get("opacity"),
                  "metallic": result.get("metallic"), "roughness": result.get("roughness"),
                  "density": result.get("density")}
        if len(coords) > PACK_THRESHOLD:
            payload_out["packed"] = _pack(fields)
            payload_out["encoding"] = "npz+zlib+base64"
        else:
            payload_out["encoding"] = "json"
            for name, value in fields.items():
                payload_out[name] = (value.astype(int).tolist()
                                     if value is not None else None)
        timings["pack"] = round(time.time() - mark, 2)
        return payload_out
    except Exception as exc:  # noqa: BLE001 - a failed job must report, not vanish
        return {"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(limit=6)}


def probe(event=None):
    """Report which optional extensions built, so a failure is diagnosable."""
    import importlib

    modules = ["torch", "xformers", "flash_attn", "spconv", "kaolin",
               "nvdiffrast", "diffoctreerast", "diff_gaussian_rasterization", "trellis",
               # FlexGEMM installs as `flex_gemm`, with an underscore. Probing
               # for "flexgemm" reported a MISSING extension that had built
               # perfectly well, which is exactly the kind of false alarm this
               # probe exists to prevent.
               "trellis2", "o_voxel", "flex_gemm", "cumesh"]
    status = {}
    for module in modules:
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
        if torch.cuda.is_available():
            total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            status["vram_gb"] = round(total, 1)
            # TRELLIS.2 is documented as needing 24 GB. Saying so here turns a
            # baffling mid-job OOM into a line in the probe.
            if version() == "2" and total < 23:
                status["vram_warning"] = (
                    f"TRELLIS.2 wants >=24 GB; this worker has {total:.1f} GB")
    except Exception:
        pass
    # If the rasterizer is absent, surface the build log rather than leaving the
    # cause on a machine nobody can log into.
    if version() == "1" and "MISSING" in str(status.get("diff_gaussian_rasterization", "")):
        try:
            with open("/workspace/rasterizer_build.log", encoding="utf-8", errors="replace") as handle:
                tail = [line.rstrip() for line in handle if line.strip()][-12:]
            status["rasterizer_build_log"] = tail
        except OSError:
            status["rasterizer_build_log"] = "no log (layer predates logging)"
    for module, log in (("flex_gemm", "flexgemm"), ("cumesh", "cumesh"), ("o_voxel", "ovoxel")):
        if "MISSING" in str(status.get(module, "")):
            try:
                with open(f"/workspace/{log}_build.log", encoding="utf-8", errors="replace") as handle:
                    status[f"{log}_build_log"] = [line.rstrip() for line in handle if line.strip()][-15:]
            except OSError:
                status[f"{log}_build_log"] = "no log"
    # TRELLIS.2 conditions on DINOv3, which is a GATED repo: without an accepted
    # licence and an HF_TOKEN on the endpoint, the pipeline loads fine and then
    # dies at the first job with a 401. Checking it here costs nothing and turns
    # a wasted GPU job into one line of probe output.
    if version() == "2":
        status["hf_token_set"] = bool(os.environ.get("HF_TOKEN")
                                      or os.environ.get("HUGGING_FACE_HUB_TOKEN"))
        try:
            from huggingface_hub import model_info

            model_info("facebook/dinov3-vitl16-pretrain-lvd1689m")
            status["dinov3_access"] = "ok"
        except Exception as exc:
            status["dinov3_access"] = (
                f"BLOCKED ({type(exc).__name__}: {str(exc)[:160]}) -- accept the "
                f"licence at huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m "
                f"and set HF_TOKEN on the endpoint")
    status["trellis_version"] = version()
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
