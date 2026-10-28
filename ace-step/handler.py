"""RunPod serverless worker: ACE-Step 1.5 (MIT), text to music.

ACE-Step's own API server runs inside the container (models loaded once, at start), and each job becomes one
release_task -> query_result round on it; the audio comes back base64-encoded. Using the project's server rather
than its Python internals keeps the worker on the code path ACE-Step tests and documents.

Input:
  {"prompt": "calm island adventure, marimba and flute, 100 bpm", "duration": 30}
  optional: "instrumental" (true), "lyrics", "seed" (-1: random), "steps" (8), "format" ("mp3"), "bpm",
            "key_scale" ("C major"), "time_signature" ("4"), "thinking" (true: the 5 Hz LM plans the piece)
  {"probe": true}  -> versions, GPU and whether the models are loaded; no generation
Output:
  {"audio_base64", "format", "seed", "duration", "seconds", "metas", "dit_model", "lm_model", "commit", "acestep"}
"""

import base64
import json
import os
import subprocess
import sys
import time
import urllib.parse

import requests
import runpod

PORT = 8001
BASE = f"http://127.0.0.1:{PORT}"
COMMIT = os.environ.get("WORKER_COMMIT", "unknown")
ACESTEP = os.environ.get("ACESTEP_COMMIT", "unknown")
FORMATS = {"mp3", "flac", "opus", "aac", "wav"}


def start_server():
    """ACE-Step's API server on localhost, models loaded eagerly: /health answers once they are."""
    env = dict(os.environ, ACESTEP_API_HOST="127.0.0.1", ACESTEP_API_PORT=str(PORT), ACESTEP_NO_INIT="false")
    server = subprocess.Popen([sys.executable, "-m", "acestep.api_server", "--host", "127.0.0.1", "--port", str(PORT)], cwd="/app", env=env)
    started = time.time()
    while time.time() - started < 900:
        if server.poll() is not None:
            raise RuntimeError(f"ACE-Step's server exited with {server.returncode} while starting")
        try:
            if requests.get(f"{BASE}/health", timeout=5).ok:
                print(f"[worker] ACE-Step ready in {time.time() - started:.1f} s", flush=True)
                return server
        except requests.RequestException:
            pass
        time.sleep(2)
    raise RuntimeError("ACE-Step's server didn't answer /health within 15 minutes")


def unwrap(body):
    """The server wraps its answers ({"data": ..., "code": ...}); take what's inside."""
    return body.get("data", body) if isinstance(body, dict) else body


def gpu_name():
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception as error:  # noqa: BLE001 - a probe reports, never fails
        return f"unknown ({error})"


def generate(job_input):
    prompt = str(job_input.get("prompt") or job_input.get("caption") or "").strip()
    if not prompt:
        return {"error": "prompt is required: the music to make (style, instruments, tempo, mood)"}
    duration = max(10.0, min(240.0, float(job_input.get("duration", 30))))
    audio_format = str(job_input.get("format", "mp3")).lower()
    if audio_format not in FORMATS:
        return {"error": f"format is one of {sorted(FORMATS)}"}
    lyrics = str(job_input.get("lyrics") or "")
    if job_input.get("instrumental", True) and not lyrics:
        lyrics = "[Instrumental]"
    seed = int(job_input.get("seed", -1))
    request = {
        "prompt": prompt,
        "lyrics": lyrics,
        "audio_duration": duration,
        "inference_steps": max(1, min(20, int(job_input.get("steps", 8)))),
        "audio_format": audio_format,
        "batch_size": 1,
        "thinking": bool(job_input.get("thinking", True)),
        "lm_backend": "pt",
        "use_random_seed": seed < 0,
        "seed": seed,
    }
    for key in ("bpm", "key_scale", "time_signature", "vocal_language"):
        if job_input.get(key) not in (None, ""):
            request[key] = job_input[key]
    started = time.time()
    submitted = unwrap(requests.post(f"{BASE}/release_task", json=request, timeout=60).json())
    task_id = submitted.get("task_id") if isinstance(submitted, dict) else None
    if not task_id:
        return {"error": f"ACE-Step refused the task: {json.dumps(submitted)[:500]}"}
    while time.time() - started < 600:
        time.sleep(1)
        items = unwrap(requests.post(f"{BASE}/query_result", json={"task_id_list": [task_id]}, timeout=30).json())
        item = items[0] if isinstance(items, list) and items else {}
        status = item.get("status", 0)
        if status == 0:
            continue
        result = item.get("result")
        if isinstance(result, str):
            try:
                result = json.loads(result)
            except json.JSONDecodeError:
                pass
        if status != 1:
            return {"error": f"ACE-Step failed: {json.dumps(result)[:800]}"}
        made = result[0] if isinstance(result, list) and result else result
        if not isinstance(made, dict) or not made.get("file"):
            return {"error": f"ACE-Step gave no file: {json.dumps(result)[:800]}"}
        # The file is in this container: read it where the server saved it (its URL names the path).
        query = urllib.parse.urlparse(made["file"]).query
        path = urllib.parse.parse_qs(query).get("path", [""])[0]
        if path and os.path.isfile(path):
            with open(path, "rb") as handle:
                audio = handle.read()
            os.remove(path)
        else:
            audio = requests.get(f"{BASE}{made['file']}", timeout=60).content
        return {
            "audio_base64": base64.b64encode(audio).decode("ascii"),
            "format": audio_format,
            "bytes": len(audio),
            "seed": made.get("seed_value"),
            "duration": duration,
            "seconds": round(time.time() - started, 2),
            "metas": made.get("metas"),
            "dit_model": made.get("dit_model"),
            "lm_model": made.get("lm_model"),
            "commit": COMMIT,
            "acestep": ACESTEP,
        }
    return {"error": "ACE-Step took longer than 10 minutes"}


def handler(job):
    job_input = job.get("input") or {}
    if job_input.get("probe"):
        health = requests.get(f"{BASE}/health", timeout=10)
        return {"ready": health.ok, "health": unwrap(health.json()) if health.ok else health.text[:300], "gpu": gpu_name(), "commit": COMMIT, "acestep": ACESTEP}
    try:
        return generate(job_input)
    except Exception as error:  # noqa: BLE001 - the caller gets the reason
        return {"error": f"{type(error).__name__}: {error}"}


if __name__ == "__main__":
    start_server()
    runpod.serverless.start({"handler": handler})
