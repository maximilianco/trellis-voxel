"""SkinTokens auto-rigger as a RunPod serverless worker: glb in, rigged glb out.

Stage two of the avatar pipeline (Hunyuan3D -> here -> Luanti conversion).

Upstream's supported entry point is the CLI:

    python demo.py --input in.glb --output out.glb --use_transfer

and this shells out to it rather than importing the pipeline. That is a
deliberate choice: the repo's Python surface is a Gradio demo, not a stable
API, so calling the documented command is the thing least likely to break on an
upstream commit. The cost is process startup per job, which is noise next to
autoregressive generation.

--use_transfer is on by default here. Without it the rigged mesh comes back
without the original texture and at the model's own normalised scale, and this
pipeline cares about both -- the whole reason for running Hunyuan3D upstream is
the texture, and throwing it away at the rigging stage would be perverse.
"""
from __future__ import annotations

import base64
import glob
import json
import os
import subprocess
import tempfile
import time
import traceback

_ROOT = os.environ.get("SKINTOKENS_ROOT", "/workspace/SkinTokens")
_OUTPUT_DIR = "/runpod-volume/outputs"
_INLINE_LIMIT = int(os.environ.get("INLINE_LIMIT_BYTES", 8 * 1024 * 1024))


def probe(event=None) -> dict:
    report = {"root": _ROOT, "inline_limit_bytes": _INLINE_LIMIT}
    try:
        import torch
        report["torch"] = torch.__version__
        report["cuda"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            total = torch.cuda.get_device_properties(0).total_memory / 1e9
            report["gpu"] = torch.cuda.get_device_name(0)
            report["vram_gb"] = round(total, 1)
            if total < 14:
                report["warning"] = (f"{total:.0f} GB card is under the 14 GB "
                                     f"SkinTokens needs")
    except Exception as exc:  # noqa: BLE001
        report["torch_error"] = f"{type(exc).__name__}: {exc}"
    report["demo_py"] = os.path.isfile(os.path.join(_ROOT, "demo.py"))
    # Weights land in experiments/ via download.py; say what is actually there
    # rather than trusting the build log.
    experiments = os.path.join(_ROOT, "experiments")
    report["weights"] = sorted(os.path.basename(p) for p in glob.glob(experiments + "/*")) \
        if os.path.isdir(experiments) else "missing"
    return report


def _deliver(blob: bytes, job_id: str, prefer_volume: bool) -> dict:
    if not prefer_volume and len(blob) <= _INLINE_LIMIT:
        return {"glb_b64": base64.b64encode(blob).decode("ascii"),
                "encoding": "glb", "bytes": len(blob), "delivery": "inline"}
    if os.path.isdir("/runpod-volume"):
        os.makedirs(_OUTPUT_DIR, exist_ok=True)
        path = os.path.join(_OUTPUT_DIR, f"{job_id}_rigged.glb")
        with open(path, "wb") as handle:
            handle.write(blob)
        return {"glb_path": path, "bytes": len(blob), "delivery": "volume"}
    return {"error": f"rigged glb is {len(blob)} bytes, over the {_INLINE_LIMIT} "
                     f"inline limit, and no /runpod-volume is attached"}


def _summarise(blob: bytes) -> dict:
    """Report what the rig actually contains, not just that a file came back.

    A glb with no skin is a silent failure -- it loads, it renders, and it will
    not animate -- so the caller gets the joint and animation counts to check.
    """
    import struct
    try:
        length, _kind = struct.unpack_from("<II", blob, 12)
        document = json.loads(blob[20:20 + length])
        skins = document.get("skins", [])
        return {"nodes": len(document.get("nodes", [])),
                "skins": len(skins),
                "joints": len(skins[0]["joints"]) if skins else 0,
                "animations": len(document.get("animations", [])),
                "bone_names": [document["nodes"][j].get("name")
                               for j in skins[0]["joints"][:64]] if skins else []}
    except Exception as exc:  # noqa: BLE001
        return {"summary_error": f"{type(exc).__name__}: {exc}"}


def handler(event):
    payload = (event or {}).get("input") or {}
    job_id = (event or {}).get("id") or str(int(time.time()))
    started = time.time()

    if payload.get("probe"):
        return {"probe": probe()}

    try:
        if payload.get("glb_b64"):
            source_bytes = base64.b64decode(payload["glb_b64"])
        elif payload.get("glb_path"):
            # Lets stage one hand over a file on the shared volume instead of
            # round-tripping a textured mesh through two base64 payloads.
            with open(payload["glb_path"], "rb") as handle:
                source_bytes = handle.read()
        else:
            return {"error": "no glb_b64 or glb_path in input"}

        work = tempfile.mkdtemp(prefix="rig_")
        source = os.path.join(work, "input.glb")
        destination = os.path.join(work, "output.glb")
        with open(source, "wb") as handle:
            handle.write(source_bytes)

        command = ["python", "demo.py", "--input", source, "--output", destination,
                   "--top_k", str(int(payload.get("top_k", 5))),
                   "--temperature", str(float(payload.get("temperature", 1.0)))]
        if payload.get("use_transfer", True):
            command.append("--use_transfer")
        if payload.get("use_skeleton"):
            command.append("--use_skeleton")

        mark = time.time()
        completed = subprocess.run(command, cwd=_ROOT, capture_output=True,
                                   text=True, timeout=int(payload.get("timeout", 1500)))
        elapsed = round(time.time() - mark, 2)

        if not os.path.isfile(destination):
            # The CLI can exit 0 having written nothing; the file is the truth.
            return {"error": "SkinTokens produced no output",
                    "returncode": completed.returncode,
                    "stdout": completed.stdout[-2000:],
                    "stderr": completed.stderr[-2000:]}

        with open(destination, "rb") as handle:
            blob = handle.read()

        result = {"rig": _summarise(blob),
                  "source_bytes": len(source_bytes),
                  "timings": {"rig": elapsed, "total": round(time.time() - started, 2)}}
        result.update(_deliver(blob, job_id, bool(payload.get("force_volume", False))))
        if not result["rig"].get("skins"):
            result["warning"] = ("the returned glb has no skin: it will render "
                                 "but not animate")
        return result
    except subprocess.TimeoutExpired:
        return {"error": "SkinTokens timed out", "seconds": payload.get("timeout", 1500)}
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(limit=8)}


if __name__ == "__main__":
    import runpod
    runpod.serverless.start({"handler": handler})
