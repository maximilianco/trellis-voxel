# ACE-Step 1.5 worker

Text to music with [ACE-Step 1.5](https://github.com/ACE-Step/ACE-Step-1.5) (MIT): instrumental loops and songs, a
few seconds per track on a 24 GB GPU. Built by `.github/workflows/ace-step.yml` into
`ghcr.io/<repo>:ace-step-<sha>`; the main model (10.1 GB) is baked in, so the endpoint needs no volume.

The handler runs ACE-Step's own API server inside the container (models loaded once at start) and turns each job
into one generation on it.

```json
{"input": {"prompt": "calm island adventure, marimba and flute, 100 bpm", "duration": 30}}
{"input": {"probe": true}}
```

Optional input: `instrumental` (true), `lyrics`, `seed`, `steps` (8), `format` (`mp3`), `bpm`, `key_scale`,
`time_signature`, `thinking` (true: the 5 Hz LM plans the piece before the DiT renders it). The output carries
`audio_base64`, `format`, `seed`, `seconds` (generation time), the models and both commits.

Endpoint: a 24 GB GPU (A5000, 3090, 4090, L4) on hosts with CUDA 12.8, 40 GB container disk.
