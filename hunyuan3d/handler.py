"""Hunyuan3D-2.1 as a RunPod serverless worker: image in, textured glb out.

Stage one of the avatar pipeline. Stage two is the SkinTokens rigger
(deploy/rig), stage three is the local Luanti conversion (lab/luanti_rig.py).

The reason to run this at all instead of TRELLIS is the texture. TRELLIS bakes
its albedo out of the multiview diffusion and the result measures flat: a 2048
map off TRELLIS.2 has a mean gradient of 1.16/255, and rebuilding it from a
128px thumbnail costs 4.2/255 of error. Asking that pipeline for more pixels
gets more pixels and no more detail. Hunyuan3D runs a separate 2B paint model
over the finished shape, which is a different mechanism rather than a larger
setting.

Two things bite on this endpoint and both are handled below.

  * VRAM. Shape is 10 GB and paint is 21 GB, but they are 29 GB together, so a
    24 GB card can do either and not both. probe() says so explicitly rather
    than letting a job die three minutes in.
  * Response size. A textured glb does not fit in a RunPod /runsync response --
    this is the same ceiling that made TRELLIS' 4096 texture come back empty.
    Anything past the inline limit is written to the network volume and
    returned as a path instead of a blob.
"""
from __future__ import annotations

import base64
import io
import os
import time
import traceback

_RUNPOD_CACHE = "/runpod-volume/huggingface-cache"
_OUTPUT_DIR = "/runpod-volume/outputs"

# RunPod's synchronous response ceiling is not published as a hard number and
# behaves as "large payloads silently arrive empty", so this is deliberately
# conservative. Past it, the glb goes to the volume.
_INLINE_LIMIT = int(os.environ.get("INLINE_LIMIT_BYTES", 8 * 1024 * 1024))

_SHAPE = None
_PAINT = None


def _persist_hy3d_cache():
    """Point Hunyuan3D's own cache at the volume, if there is one.

    Setting HF_HOME is not enough. hy3dshape does not fetch through the plain
    HuggingFace cache -- the worker log says so directly:

        Try to load model from local path:
          /root/.cache/hy3dgen/tencent/Hunyuan3D-2.1/hunyuan3d-dit-v2-1
        Model path not exists, try to download from huggingface

    That path is fixed under the home directory, so without this the 15 GB of
    weights land on container disk and every cold worker downloads them again,
    however large a network volume is attached.
    """
    if not os.path.isdir("/runpod-volume"):
        return
    target = "/runpod-volume/hy3dgen"
    link = os.path.expanduser("~/.cache/hy3dgen")
    try:
        os.makedirs(target, exist_ok=True)
        os.makedirs(os.path.dirname(link), exist_ok=True)
        if os.path.islink(link):
            return
        if os.path.isdir(link):
            # Real directory already holding weights: leave it be rather than
            # delete something a running job may be reading.
            return
        os.symlink(target, link)
    except OSError as exc:
        print(f"could not persist the hy3dgen cache: {exc}", flush=True)


def model_path() -> str:
    """Prefer the host-side model cache when the endpoint has one attached."""
    if os.path.isdir(_RUNPOD_CACHE):
        os.environ.setdefault("HF_HOME", _RUNPOD_CACHE)
        os.environ.setdefault("HUGGINGFACE_HUB_CACHE", _RUNPOD_CACHE)
    _persist_hy3d_cache()
    baked = "/workspace/weights"
    if os.path.isdir(baked) and os.listdir(baked):
        return baked
    return os.environ.get("HY3D_MODEL", "tencent/Hunyuan3D-2.1")


def shape_pipeline():
    global _SHAPE
    if _SHAPE is None:
        from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
        _SHAPE = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            model_path(), subfolder="hunyuan3d-dit-v2-1")
    return _SHAPE


def _resolve_config_paths(config, verbose: bool = True) -> dict:
    """Rewrite the paint config's relative paths to absolute ones.

    Upstream disagrees with itself about what its relative paths are relative
    to. The multiview config is spelled

        hy3dpaint/cfgs/hunyuan-paint-pbr.yaml     (from the repo root)

    and the upscaler is spelled

        ckpt/RealESRGAN_x4plus.pth                (from inside hy3dpaint)

    No working directory satisfies both, so chdir cannot fix this -- it only
    chooses which of the two fails. Each relative path is instead tried against
    both bases and set to whichever actually exists, which leaves correct paths
    untouched and does not depend on where the worker runs.
    """
    root = os.environ.get("HY3D_ROOT", "/workspace/Hunyuan3D-2.1")
    bases = (root, os.path.join(root, "hy3dpaint"))
    resolved = {}
    fields = vars(config) if hasattr(config, "__dict__") else {}
    for attr, value in list(fields.items()):
        if not isinstance(value, str) or not value or os.path.isabs(value):
            continue
        for base in bases:
            candidate = os.path.join(base, value)
            if os.path.exists(candidate):
                setattr(config, attr, candidate)
                resolved[attr] = candidate
                if verbose:
                    print(f"  paint config: {attr} -> {candidate}", flush=True)
                break
        else:
            # Existence is the filter, so anything left here is either not a
            # path at all ("cuda") or genuinely missing. Report the ones that
            # look like files so a missing weight is visible in the log rather
            # than two minutes into the next job.
            if "/" in value or value.endswith((".yaml", ".yml", ".pth", ".ckpt")):
                resolved[attr] = f"UNRESOLVED: {value}"
                if verbose:
                    print(f"  paint config: {attr} UNRESOLVED {value}", flush=True)
    return resolved


def paint_pipeline(view_size: int = 512, max_views: int = 6):
    global _PAINT
    if _PAINT is None:
        from textureGenPipeline import Hunyuan3DPaintPipeline, Hunyuan3DPaintConfig
        config = Hunyuan3DPaintConfig(max_num_view=max_views, resolution=view_size)
        _resolve_config_paths(config)
        _PAINT = Hunyuan3DPaintPipeline(config)
    return _PAINT


def probe(event=None, deep: bool = False) -> dict:
    """Report what the worker actually has, over the API.

    The machine driving this endpoint has a 4 GB GPU and no Docker, so
    `docker run --gpus` is not an available way to find out whether the image
    works. It has to be askable remotely.
    """
    report = {"model_path": model_path(),
              "runpod_cache": os.path.isdir(_RUNPOD_CACHE),
              "inline_limit_bytes": _INLINE_LIMIT}
    try:
        import torch
        report["torch"] = torch.__version__
        report["cuda"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            total = torch.cuda.get_device_properties(0).total_memory / 1e9
            report["gpu"] = name
            report["vram_gb"] = round(total, 1)
            if total < 23:
                report["warning"] = (f"{total:.0f} GB card: too small for shape "
                                     f"(10 GB) plus paint (21 GB). Send "
                                     f"want_texture=false, or move to a 48 GB GPU.")
            elif total < 31:
                report["warning"] = (f"{total:.0f} GB card: shape and paint do not "
                                     f"fit together (29 GB). Texture is run in a "
                                     f"second pass with the shape model unloaded.")
    except Exception as exc:  # noqa: BLE001
        report["torch_error"] = f"{type(exc).__name__}: {exc}"
    for name, module in (("hy3dshape", "hy3dshape.pipelines"),
                         ("hy3dpaint", "textureGenPipeline")):
        try:
            __import__(module)
            report[name] = "importable"
        except Exception as exc:  # noqa: BLE001
            report[name] = f"{type(exc).__name__}: {exc}"

    # "importable" turned out not to mean "will run": the paint stage failed
    # twice on data files it opens by relative path, each time only after two
    # minutes of shape generation. Building the config here is cheap -- it
    # loads no weights -- and says exactly which paths resolve, so that class
    # of failure is answerable from a probe instead of a job.
    try:
        from textureGenPipeline import Hunyuan3DPaintConfig
        config = Hunyuan3DPaintConfig(max_num_view=6, resolution=512)
        report["paint_config"] = _resolve_config_paths(config, verbose=False)
    except Exception as exc:  # noqa: BLE001
        report["paint_config"] = f"{type(exc).__name__}: {exc}"

    # Constructing the config was still not enough: the next failure was a
    # missing module imported while the *pipeline* was being built, which
    # again cost a shape generation to reach. deep=True builds the pipeline
    # itself. That loads weights and takes minutes, but it is the only check
    # that exercises what a real job exercises, and it costs no generation.
    if deep:
        started = time.time()
        try:
            paint_pipeline()
            report["paint_pipeline"] = f"constructed in {time.time()-started:.0f}s"
        except Exception as exc:  # noqa: BLE001
            report["paint_pipeline"] = (f"{type(exc).__name__}: {exc}
"
                                        + traceback.format_exc()[-1500:])
    return report


def _load_image(payload: dict):
    from PIL import Image
    if payload.get("image_b64"):
        data = base64.b64decode(payload["image_b64"])
    elif payload.get("image_url"):
        import urllib.request
        with urllib.request.urlopen(payload["image_url"], timeout=60) as response:
            data = response.read()
    else:
        raise ValueError("no image_b64 or image_url in input")
    return Image.open(io.BytesIO(data)).convert("RGBA")


def _cutout(image):
    """A white background becomes geometry, so it has to go before generation."""
    import rembg
    return rembg.remove(image)


def _deliver(blob: bytes, job_id: str, prefer_volume: bool) -> dict:
    """Inline the glb when it fits, otherwise leave it on the volume.

    Returning a path is not a fallback so much as the normal case for anything
    textured: a painted 2k glb is comfortably past the inline ceiling.
    """
    if not prefer_volume and len(blob) <= _INLINE_LIMIT:
        return {"glb_b64": base64.b64encode(blob).decode("ascii"),
                "encoding": "glb", "bytes": len(blob), "delivery": "inline"}
    if os.path.isdir(os.path.dirname(_OUTPUT_DIR)) or os.path.isdir("/runpod-volume"):
        os.makedirs(_OUTPUT_DIR, exist_ok=True)
        path = os.path.join(_OUTPUT_DIR, f"{job_id}.glb")
        with open(path, "wb") as handle:
            handle.write(blob)
        return {"glb_path": path, "bytes": len(blob), "delivery": "volume",
                "hint": "attach the same network volume to read this, or "
                        "re-run with want_texture=false for an inline result"}
    return {"error": f"glb is {len(blob)} bytes, over the {_INLINE_LIMIT} inline "
                     f"limit, and no /runpod-volume is attached to write it to"}


def handler(event):
    payload = (event or {}).get("input") or {}
    job_id = (event or {}).get("id") or str(int(time.time()))
    started = time.time()
    timings = {}

    if payload.get("probe"):
        return {"probe": probe(deep=bool(payload.get("deep")))}

    try:
        import torch
        import trimesh

        image = _load_image(payload)
        if payload.get("remove_background", True):
            try:
                image = _cutout(image)
            except Exception as exc:  # noqa: BLE001
                print(f"cutout unavailable ({type(exc).__name__}: {exc}); "
                      f"a flat background will become geometry", flush=True)
        timings["load"] = round(time.time() - started, 2)

        mark = time.time()
        mesh = shape_pipeline()(
            image=image,
            num_inference_steps=int(payload.get("steps", 30)),
            guidance_scale=float(payload.get("guidance_scale", 5.0)),
            octree_resolution=int(payload.get("octree_resolution", 256)),
            generator=torch.manual_seed(int(payload.get("seed", 0))),
        )[0]
        timings["shape"] = round(time.time() - mark, 2)

        # Clean up before decimating: floaters and degenerate faces otherwise
        # eat budget that should go to the body, and they are exactly the loose
        # shells that show up as specks in-game.
        mark = time.time()
        for name in ("FloaterRemover", "DegenerateFaceRemover"):
            try:
                module = __import__("hy3dshape.postprocessors", fromlist=[name])
                mesh = getattr(module, name)()(mesh)
            except Exception as exc:  # noqa: BLE001
                print(f"{name} unavailable: {type(exc).__name__}: {exc}", flush=True)

        # Luanti player models live around 8k triangles, not the million a
        # generator will happily emit.
        target = int(payload.get("decimation_target", 0))
        if target:
            try:
                from hy3dshape.postprocessors import FaceReducer
                mesh = FaceReducer()(mesh, max_facenum=target)
            except Exception as exc:  # noqa: BLE001
                print(f"FaceReducer unavailable: {type(exc).__name__}: {exc}", flush=True)
        timings["postprocess"] = round(time.time() - mark, 2)

        counts = {"vertices": int(len(mesh.vertices)), "faces": int(len(mesh.faces))}

        if payload.get("want_texture", True):
            mark = time.time()
            # Shape and paint together need 29 GB. Dropping the shape model
            # first is what lets a 24 GB card get through both stages.
            if payload.get("unload_shape", True):
                global _SHAPE
                _SHAPE = None
                torch.cuda.empty_cache()
            import tempfile
            handle, untextured = tempfile.mkstemp(suffix=".obj")
            os.close(handle)
            mesh.export(untextured)
            painted = paint_pipeline(
                view_size=int(payload.get("view_size", 512)),
                max_views=int(payload.get("max_views", 6)),
            )(mesh_path=untextured, image_path=image)
            timings["paint"] = round(time.time() - mark, 2)
            mesh = trimesh.load(painted, force="scene") if isinstance(painted, str) else painted

        blob = mesh.export(file_type="glb")
        if isinstance(blob, str):
            blob = blob.encode("utf-8")
        timings["total"] = round(time.time() - started, 2)

        result = {"counts": counts, "timings": timings,
                  "textured": bool(payload.get("want_texture", True))}
        result.update(_deliver(blob, job_id, bool(payload.get("force_volume", False))))
        return result
    except Exception as exc:  # noqa: BLE001 - a failed job must report, not vanish
        return {"error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(limit=8)}


if __name__ == "__main__":
    import runpod
    runpod.serverless.start({"handler": handler})
