"""Probe or test a new endpoint without creating storefront jobs or credits.

Set RUNPOD_API_KEY, or use --config pointing to the existing minetest-style
config. Credentials are never printed or written to the output archive.
"""
import argparse
import base64
import io
import json
import os
import time
import urllib.request
import zlib
from pathlib import Path

import numpy as np


def request(url,key,payload=None):
    request=urllib.request.Request(url,headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'},
        data=None if payload is None else json.dumps(payload).encode())
    with urllib.request.urlopen(request,timeout=60) as response:return json.load(response)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--endpoint',required=True)
    parser.add_argument('--config')
    parser.add_argument('--image',help='Omit for an import/GPU probe; setting this buys one inference job')
    parser.add_argument('--output',required=True)
    parser.add_argument('--seed',type=int,default=2090003113)
    parser.add_argument('--resolution',type=int,choices=[64,128,256,512],default=512)
    parser.add_argument('--model-resolution',type=int,choices=[1024,1536],default=1024)
    parser.add_argument('--timeout',type=int,default=2400)
    args=parser.parse_args()
    key=os.environ.get('RUNPOD_API_KEY')
    if not key and args.config:
        config=dict(row.split('=',1) for row in Path(args.config).read_text().splitlines()
                    if '=' in row and not row.lstrip().startswith('#'))
        key={k.strip():v.strip() for k,v in config.items()}.get('runpod_api_key')
    if not key:parser.error('Set RUNPOD_API_KEY or provide --config')
    if not args.endpoint.isalnum():parser.error('Endpoint must be an ID, not a URL')
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    raw=Path(args.image).read_bytes() if args.image else None
    payload={'image_b64':base64.b64encode(raw).decode(),'seed':args.seed,'resolution':args.resolution,
             'model_resolution':args.model_resolution,'want_colour':True} if raw else {'probe':True}
    base=f'https://api.runpod.ai/v2/{args.endpoint}'
    submitted=request(base+'/run',key,{'input':payload})
    job=submitted['id'];started=time.time();last=None
    try:
        while time.time()-started<args.timeout:
            result=request(base+'/status/'+job,key)
            status=result.get('status')
            (output/'job.json').write_text(json.dumps({'id':job,'endpoint':args.endpoint,'status':status,
                'elapsed_seconds':round(time.time()-started),'delayTime':result.get('delayTime'),
                'executionTime':result.get('executionTime')},indent=2))
            if status!=last:print(f'{job}: {status}',flush=True);last=status
            if status=='COMPLETED':break
            if status in ['FAILED','CANCELLED','TIMED_OUT']:raise RuntimeError('Job failed: '+str(result.get('error',''))[:500])
            time.sleep(5)
        else:raise TimeoutError('Endpoint test timed out')
    except BaseException:
        try:request(base+'/cancel/'+job,key,{})
        except Exception:pass
        raise
    value=result.get('output') or {}
    if value.get('error'):raise RuntimeError(value['error'])
    if raw:
        if value.get('generator')!='pixal3d' or value.get('colour_source')!='pbr':
            raise RuntimeError('Incorrect worker or missing PBR colours')
        packed=value.pop('packed')
        with np.load(io.BytesIO(zlib.decompress(base64.b64decode(packed)))) as arrays:
            if arrays['coords'].shape!=arrays['colours'].shape or not len(arrays['coords']):
                raise RuntimeError('Invalid voxel payload')
            np.savez_compressed(output/'voxels.npz',**{name:arrays[name] for name in arrays.files})
        (output/'payload_meta.json').write_text(json.dumps(value,indent=2))
        (output/'source.png').write_bytes(raw)
    else:
        (output/'probe.json').write_text(json.dumps(value,indent=2))
        probe=value.get('probe') or {}
        if not probe.get('gpu') or not probe.get('natten_kernel',{}).get('ok') or any(
            str(version).startswith('MISSING') for version in probe.get('modules',{}).values()):
            raise RuntimeError('Worker probe failed; see probe.json for import/GPU/kernel details')
    print(json.dumps(value,indent=2))


if __name__=='__main__':main()
