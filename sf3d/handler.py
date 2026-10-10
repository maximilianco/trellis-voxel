"""Single-image SF3D RunPod worker; returns the original textured GLB."""
import base64
import io
import os
import time
import threading

MODEL = None
REMBG = None
LOCK = threading.Lock()
MAX_IMAGE_BYTES = 20 * 1024 * 1024
MAX_PIXELS = 16 * 1024 * 1024


def decode_image(payload):
    from PIL import Image
    value = payload.get('image_b64')
    if not isinstance(value, str) or len(value) > (MAX_IMAGE_BYTES * 4 // 3 + 128):
        raise ValueError('Provide image_b64, at most 20 MiB')
    if value.startswith('data:'):
        if ',' not in value:
            raise ValueError('Invalid image data URI')
        value = value.split(',', 1)[1]
    raw = base64.b64decode(value, validate=True)
    if not raw or len(raw) > MAX_IMAGE_BYTES:
        raise ValueError('Image is empty or too large')
    image = Image.open(io.BytesIO(raw))
    if image.width * image.height > MAX_PIXELS or min(image.size) < 16:
        raise ValueError('Image dimensions must be at least 16 and at most 16 megapixels')
    image.load()
    from PIL import ImageOps
    return ImageOps.exif_transpose(image).convert('RGBA')


def load_model():
    global MODEL
    if MODEL is not None:
        return MODEL
    import torch
    from huggingface_hub import snapshot_download
    from sf3d.system import SF3D
    if not torch.cuda.is_available():
        raise RuntimeError('SF3D worker requires CUDA')
    local = os.environ.get('SF3D_MODEL_PATH')
    if not local:
        local = snapshot_download(
            repo_id=os.environ.get('SF3D_MODEL_ID', 'stabilityai/stable-fast-3d'),
            revision=os.environ.get('SF3D_MODEL_REVISION', 'f0c9a8ffd62cb1bbc8a7a53c9f87a0be1b6be778'),
            allow_patterns=['config.yaml', 'model.safetensors'],
            token=os.environ.get('HF_TOKEN') or None,
        )
    candidate = SF3D.from_pretrained(local, config_name='config.yaml', weight_name='model.safetensors')
    candidate.to('cuda').eval()
    MODEL = candidate
    return MODEL


def handler(job):
    global REMBG
    payload = job.get('input') or {}
    if not isinstance(payload, dict):
        raise ValueError('input must be an object')
    # Validate before model loading, so malformed requests incur no inference.
    image = decode_image(payload)
    texture_resolution = int(payload.get('texture_resolution', os.environ.get('SF3D_TEXTURE_RESOLUTION', '1024')))
    if texture_resolution not in {512, 1024, 2048}:
        raise ValueError('texture_resolution must be 512, 1024 or 2048')
    foreground_ratio = float(payload.get('foreground_ratio', 0.85))
    if not 0.5 <= foreground_ratio <= 0.95:
        raise ValueError('foreground_ratio must be between 0.5 and 0.95')
    import torch
    import rembg
    from sf3d.utils import remove_background, resize_foreground
    started = time.perf_counter()
    with LOCK:
        mark = time.perf_counter()
        model = load_model()
        torch.cuda.synchronize()
        load_seconds = time.perf_counter() - mark
        mark = time.perf_counter()
        if image.getextrema()[3][0] == 255:
            if REMBG is None:
                # CPU removal avoids reserving CUDA VRAM in ONNX Runtime.
                REMBG = rembg.new_session('u2net', providers=['CPUExecutionProvider'])
            image = remove_background(image, REMBG)
        image = resize_foreground(image, foreground_ratio, out_size=(512, 512))
        prepared = io.BytesIO()
        image.save(prepared, format='PNG')
        prepare_seconds = time.perf_counter() - mark
        torch.cuda.reset_peak_memory_stats()
        mark = time.perf_counter()
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            mesh, _ = model.run_image(image, bake_resolution=texture_resolution, remesh='none', vertex_count=-1)
        torch.cuda.synchronize()
        mesh_seconds = time.perf_counter() - mark
        if not len(mesh.vertices) or not len(mesh.faces):
            raise RuntimeError('SF3D produced an empty mesh')
        if getattr(mesh.visual.material, 'baseColorTexture', None) is None:
            raise RuntimeError('SF3D mesh is missing its color texture')
        mark = time.perf_counter()
        glb = mesh.export(file_type='glb', include_normals=True)
        export_seconds = time.perf_counter() - mark
        result = {
            'generator': 'sf3d', 'colour_source': 'texture',
            'coordinate_transform': 'sf3d-native-y-up-front-minus-z-v1',
            'glb_b64': base64.b64encode(glb).decode('ascii'),
            'prepared_image_b64': base64.b64encode(prepared.getvalue()).decode('ascii'),
            'vertices': len(mesh.vertices), 'triangles': len(mesh.faces),
            'texture_resolution': texture_resolution, 'glb_bytes': len(glb),
            'model_revision': os.environ.get('SF3D_MODEL_REVISION'),
            'worker_revision': os.environ.get('SF3D_WORKER_REVISION', 'development'),
            'gpu': torch.cuda.get_device_name(0),
            'peak_vram_gb': round(torch.cuda.max_memory_allocated() / 2**30, 3),
            'timings': { 'load': round(load_seconds, 3), 'prepare': round(prepare_seconds, 3),
                'mesh_and_texture': round(mesh_seconds, 3), 'glb_export': round(export_seconds, 3),
                'total': round(time.perf_counter() - started, 3) },
        }
        print('sf3d: mesh_and_texture=%.3fs total=%.3fs triangles=%d peak_vram=%.3fGiB' %
              (mesh_seconds, result['timings']['total'], len(mesh.faces), result['peak_vram_gb']), flush=True)
        return result


if __name__ == '__main__':
    import runpod
    runpod.serverless.start({'handler': handler})
