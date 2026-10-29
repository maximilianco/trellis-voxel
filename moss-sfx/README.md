# MOSS-SoundEffect worker

Text to sound effect with [MOSS-SoundEffect v2.0](https://huggingface.co/OpenMOSS-Team/MOSS-SoundEffect-v2.0)
(OpenMOSS, Apache 2.0 code and weights): a 1.3B DiT with flow matching, a DAC VAE and a Qwen3 text encoder; 48 kHz,
up to 30 s. SolisVox makes the sound effects recordings don't cover with it (a creature's call, a spell, a machine).
Built by `.github/workflows/moss-sfx.yml` into `ghcr.io/<repo>:moss-sfx-<sha>`.

The weights (about 10 GB) download at a worker's first start at a pinned revision and stay loaded; `torch.compile`
stays off (`TORCHDYNAMO_DISABLE=1`), since its first call compiles for minutes.

```json
{"input": {"prompt": "a heavy wooden door creaking open slowly", "seconds": 3}}
{"input": {"prompts": ["a coin pickup chime", "a short magical sparkle"], "seconds": 1.5, "steps": 50}}
{"input": {"probe": true}}
```

Options: `seconds` (0.5–30, default 4), `steps` (10–150, default 50; the model card uses 100), `cfg_scale` (default
4), `seed` (−1 random), `format` (`mp3` at 160 kbit/s, or `wav`). Output: `sounds` (`audio_b64`, `format`,
`seconds`, `seed`, `took`), the GPU, the first load's seconds and the commit. Endpoint: a 24 GB card on CUDA 12.8
hosts, 30 GB container disk.
