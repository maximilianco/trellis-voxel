"""Pixal3D RunPod worker: image -> native PBR voxels, without GLB export.

Compatible with the site's npz+zlib+base64 transport. One inference at a time.
Importing this module never loads weights or starts a RunPod worker.
"""
from __future__ import annotations

import base64
import importlib
import io
import json
import math
import os
import tempfile
import threading
import time
import traceback
import zlib
from pathlib import Path

import numpy as np
from PIL import Image

os.environ.setdefault('ATTN_BACKEND', 'sdpa')
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
cache = Path('/runpod-volume/huggingface-cache')
if (cache/'hub').is_dir():
    os.environ.setdefault('HF_HOME', str(cache))
os.environ.setdefault('HF_HOME', '/workspace/hf')
os.environ.setdefault('TORCH_HOME', str(Path(os.environ['HF_HOME'])/'torch'))

MODEL = 'TencentARC/Pixal3D'
MODEL_REVISION = 'b0cb2e1b794cab9aa0ac38a95d794a4d9337437f'
TRANSFORM = 'pixal3d-front-yaw180-v1'
_PIPELINE = None
_LOCK = threading.Lock()


def setting_bool(name, default=False):
    return os.environ.get(name, str(default)).lower() not in {'0', 'false', 'no', 'off'}


class U2NetRembg:
    """Use the same u2net cutout as the previous worker; do not load RMBG-2.0."""
    def __init__(self, *args, **kwargs):
        self.session = None
    def to(self, *args, **kwargs):
        return self
    def cpu(self):
        return self
    def __call__(self, image):
        from rembg import new_session, remove
        if self.session is None:
            self.session = new_session('u2net')
        return remove(image.convert('RGBA'), session=self.session)


def pipeline():
    global _PIPELINE
    if _PIPELINE is None:
        from huggingface_hub import snapshot_download
        from pixal3d.pipelines import rembg
        import inference
        # from_pretrained otherwise instantiates the gated background model,
        # even if input images already have alpha. Replace it before loading.
        rembg.BiRefNet = U2NetRembg
        dino_path = snapshot_download('camenduru/dinov3-vitl16-pretrain-lvd1689m',
                                     revision='3c276edd87d6f6e569ff0c4400e086807d0f3881')
        for config in inference.IMAGE_COND_CONFIGS.values():
            config['model_name'] = dino_path
        model_path = os.environ.get('PIXAL3D_MODEL_PATH') or snapshot_download(
            MODEL, revision=os.environ.get('PIXAL3D_MODEL_REVISION', MODEL_REVISION),
            allow_patterns=['pipeline.json', 'ckpts/*.json', 'ckpts/*.safetensors', 'LICENSE', 'NOTICE'],
            ignore_patterns=['*_mv.*'])
        _PIPELINE = inference.init_pipeline(model_path=model_path,
                                           low_vram=setting_bool('PIXAL3D_LOW_VRAM', True))
    return _PIPELINE


def load_image(payload):
    encoded = payload.get('image_b64')
    if not isinstance(encoded, str) or not encoded:
        raise ValueError('image_b64 is required')
    if len(encoded) > 32_000_000:
        raise ValueError('Input image is too large')
    raw = base64.b64decode(encoded, validate=True)
    with Image.open(io.BytesIO(raw)) as image:
        if image.width*image.height > 50_000_000:
            raise ValueError('Input image dimensions are too large')
        return image.convert('RGBA' if image.mode == 'RGBA' else 'RGB')


def camera_params(image, manual_fov=None):
    import torch
    import inference
    if manual_fov is not None:
        angle = float(manual_fov)
        if not math.isfinite(angle) or not 0.05 <= angle <= 2.8:
            raise ValueError('fov must be radians between 0.05 and 2.8')
        distance = inference.distance_from_fov(angle, torch.tensor([-1., 0., 0.]),
                    torch.tensor([0, 511]), 1., 512)['distance_from_x']
        return {'camera_angle_x': angle, 'distance': distance, 'mesh_scale': 1.}
    # Load MoGe only for this stage and offload before the 3D cascade.
    from huggingface_hub import hf_hub_download
    weights = hf_hub_download('Ruicheng/moge-2-vitl','model.pt',
                             revision='39c4d5e957afe587e04eec59dc2bcc3be5ecd968')
    model = inference.load_moge_model(device='cuda',model_name=weights)
    try:
        with tempfile.TemporaryDirectory(prefix='pixal-camera-') as temporary:
            path = Path(temporary)/'input.png'
            image.save(path)
            return inference.get_camera_params_wild_moge(str(path), model, device='cuda',
                                                        mesh_scale=1., image_resolution=512)
    finally:
        model.cpu()
        del model
        torch.cuda.empty_cache()


def as_numpy(value):
    if hasattr(value, 'detach'):
        value = value.detach()
        # NumPy has no bfloat16 dtype; decoder outputs may use BF16.
        if getattr(value.dtype, 'is_floating_point', False):
            value = value.float()
        value = value.cpu().numpy()
    return np.asarray(value)


def voxel_arrays(mesh, target):
    """Reduce raw PBR attributes with aligned coordinates and honest density."""
    coords = as_numpy(mesh.coords).astype(np.int32)
    attrs = as_numpy(mesh.attrs).astype(np.float32)
    if coords.ndim != 2 or coords.shape[1] != 3 or not len(coords):
        raise ValueError('Pixal3D returned no valid voxel coordinates')
    if attrs.ndim != 2 or len(attrs) != len(coords) or not np.isfinite(attrs).all():
        raise ValueError('Pixal3D returned invalid voxel attributes')
    source = int(round(1./float(mesh.voxel_size)))
    if source <= 0 or (coords<0).any() or (coords>=source).any():
        raise ValueError('Pixal3D coordinates do not fit the decoded grid')
    actual = min(int(target), source)
    if source > actual:
        binned = np.floor(coords.astype(np.float64)*actual/source).astype(np.int32)
        keys = (binned[:,0].astype(np.int64)*actual+binned[:,1])*actual+binned[:,2]
        keys, inverse = np.unique(keys, return_inverse=True)
        inverse = inverse.reshape(-1)
        counts = np.bincount(inverse)
        attrs = np.stack([np.bincount(inverse, weights=attrs[:,i])/counts
                          for i in range(attrs.shape[1])],axis=1)
        coords = np.stack([keys//(actual*actual), keys//actual%actual, keys%actual],axis=1).astype(np.int32)
        density = counts.astype(np.int32)
    else:
        density = np.ones(len(coords),np.int32)
    # Pixal's canonical camera is -Y with +Z up. Rotate horizontal axes by
    # 180 degrees to the game front (+Z after the Z-up -> Y-up axis swap).
    # This preserves handedness, vertical direction and row/attribute alignment.
    if setting_bool('PIXAL3D_FRONT_ROTATION', True):
        coords = coords.copy()
        coords[:,:2] = coords[:,:2].min(0)+coords[:,:2].max(0)-coords[:,:2]
    layout = mesh.layout
    def channel(name, rgb=False):
        span = layout.get(name)
        if span is None or not isinstance(span,slice) or (span.stop or 0)>attrs.shape[1]:
            raise ValueError('Missing PBR channel: '+name)
        value = np.clip(attrs[:,span]*255.,0,255).round().astype(np.uint8)
        if value.shape[1] != (3 if rgb else 1):
            raise ValueError('Invalid PBR channel: '+name)
        return value if rgb else value[:,0]
    return {'coords':coords, 'colours':channel('base_color',True), 'opacity':channel('alpha'),
            'metallic':channel('metallic'),'roughness':channel('roughness'),'density':density}, source, actual


def pack(fields):
    buffer = io.BytesIO()
    np.savez(buffer, **fields)
    return base64.b64encode(zlib.compress(buffer.getvalue(),6)).decode('ascii')


def probe():
    modules = {}
    for name in ['torch','pixal3d','inference','flex_gemm','cumesh','o_voxel','natten','moge','rembg','transformers']:
        try:
            module = importlib.import_module(name)
            modules[name] = getattr(module,'__version__','present')
        except Exception as error:
            modules[name] = f'MISSING ({type(error).__name__}: {error})'
    gpu = None
    kernel = {'ok':False,'error':'CUDA unavailable'}
    try:
        import torch
        if torch.cuda.is_available():
            device = torch.cuda.get_device_properties(0)
            gpu = {'name':device.name,'vram_gib':round(device.total_memory/1024**3,1),
                   'compute_capability':list(torch.cuda.get_device_capability(0))}
            from natten.functional import na2d_qk, na2d_av
            with torch.inference_mode():
                q = torch.randn((1,1,8,8,32),device='cuda',dtype=torch.float32)
                scores = na2d_qk(q,q,kernel_size=(3,3))
                result = na2d_av(scores.softmax(-1),q,kernel_size=(3,3))
                assert result.shape==q.shape and bool(torch.isfinite(result).all())
                # Exercise NAF's real legacy path, without downloading weights.
                naf = torch.hub.load('/workspace/NAF','naf',source='local',pretrained=False,device='cuda')
                result = naf(torch.randn((1,3,18,18),device='cuda'),
                             torch.randn((1,64,9,9),device='cuda'),(18,18))
                torch.cuda.synchronize()
                assert result.shape==(1,64,18,18) and bool(torch.isfinite(result).all())
                del naf
            kernel = {'ok':True,'backend':'legacy-qk-av','naf_tested':True,'pretrained':False}
    except Exception as error:
        kernel = {'ok':False,'error':f'{type(error).__name__}: {error}'}
    return {'modules':modules,'gpu':gpu,'model':MODEL,'model_revision':os.environ.get('PIXAL3D_MODEL_REVISION',MODEL_REVISION),
            'worker_revision':os.environ.get('PIXAL3D_WORKER_REVISION','development'),
            'low_vram':setting_bool('PIXAL3D_LOW_VRAM',True),'weights_loaded':_PIPELINE is not None,
            'attention_backend':os.environ.get('ATTN_BACKEND'),'natten_kernel':kernel,'background_remover':'u2net'}


def handler(event):
    payload = (event or {}).get('input') or {}
    if payload.get('probe'):
        return {'probe':probe()}
    started = time.time()
    try:
        target = int(payload.get('resolution',512))
        if target not in (64,128,256,512):
            raise ValueError('resolution must be 64, 128, 256 or 512 (voxel output grid)')
        model_resolution = int(payload.get('model_resolution',os.environ.get('PIXAL3D_MODEL_RESOLUTION','1024')))
        if model_resolution not in (1024,1536):
            raise ValueError('model_resolution must be 1024 or 1536')
        kind = payload.get('pipeline_type')
        if kind and kind != f'{model_resolution}_cascade':
            raise ValueError('pipeline_type must match model_resolution; TRELLIS pipeline names are incompatible')
        if payload.get('want_colour') is False:
            raise ValueError('Colour is required for schematic generation')
        # Validate the input before fetching weights or occupying a GPU.
        image = load_image(payload)
        seed = int(payload.get('seed',0))
        with _LOCK:
            import torch
            with torch.inference_mode():
                pipe = pipeline()
                image = pipe.preprocess_image(image)
                camera = camera_params(image,payload.get('fov'))
                timings = {'preparing':round(time.time()-started,2)}
                mark = time.time()
                meshes = pipe.run(image, camera_params=camera, seed=seed, preprocess_image=False,
                                  pipeline_type=f'{model_resolution}_cascade', max_num_tokens=49152)
                timings['generate'] = round(time.time()-mark,2)
                fields, source, actual = voxel_arrays(meshes[0],target)
                mark = time.time()
                packed = pack(fields)
                timings['pack'] = round(time.time()-mark,2)
                timings['total'] = round(time.time()-started,2)
        return {'generator':'pixal3d','model':MODEL,'model_revision':os.environ.get('PIXAL3D_MODEL_REVISION',MODEL_REVISION),
                'worker_revision':os.environ.get('PIXAL3D_WORKER_REVISION','development'),
                'colour_source':'pbr','encoding':'npz+zlib+base64','packed':packed,
                'resolution':actual,'source_resolution':source,'model_resolution':model_resolution,
                'coordinate_transform':TRANSFORM if setting_bool('PIXAL3D_FRONT_ROTATION',True) else 'none',
                'camera':camera,'counts':{'voxels':len(fields['coords'])},'timings':timings}
    except Exception as error:
        return {'error':f'{type(error).__name__}: {error}','traceback':traceback.format_exc(limit=6)}


if __name__ == '__main__':
    import runpod
    runpod.serverless.start({'handler':handler})
