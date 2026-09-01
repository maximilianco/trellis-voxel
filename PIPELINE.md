# Avatar pipeline: prompt to Luanti player model

```
prompt --[gemini]--> image --[hunyuan3d]--> textured glb
       --[skintokens]--> rigged glb --[local]--> luanti player model
```

Three stages. The first two are RunPod serverless endpoints built on GitHub;
the third is numpy-only geometry and runs on the workstation, because there is
no reason to rent a GPU to rescale a mesh.

| stage | where | in / out | code |
|---|---|---|---|
| 1 shape + PBR texture | RunPod, 24 GB GPU | png -> textured glb | `deploy/hunyuan3d` |
| 2 auto-rig | RunPod, 24 GB GPU | glb -> rigged glb | `deploy/rig` |
| 3 Luanti conversion | local | rigged glb -> player model | `lab/luanti_rig.py` |

Driver for all three: `lab/avatar_pipeline.py`.

## Why these two models

**Hunyuan3D-2.1 over TRELLIS**, for the texture specifically. TRELLIS bakes its
albedo out of the multiview diffusion and the output measures flat: a 2048 map
off TRELLIS.2 has a mean gradient of 1.16/255, and rebuilding it from a 128px
thumbnail costs only 4.2/255 of error. That is why generated avatars read as
low-resolution no matter what texture size is requested — the pixels are there
and the detail is not, so asking for 4096 buys nothing (and exceeds the RunPod
response limit besides). Hunyuan3D runs a separate 2B PBR paint model over the
finished shape, which is a different mechanism rather than a larger setting.

**SkinTokens over fitting a skeleton geometrically.** It predicts skeleton and
skin weights as one autoregressive token sequence. The alternative already in
this repo (`lab/rig.py`) fits the six Luanti bones to the mesh and weights by
distance-to-bone; that works only for a humanoid in a known pose, and it
creases at the shoulder. It stays as the fallback for when stage 2 is
unavailable or returns no skin.

## Deploying

Both images build on GitHub Actions and push to GHCR. This is not a preference:
the workstation has a 4 GB GPU, no Docker, single-digit GB free on C: and a
~0.5 MB/s uplink.

1. Push a repo containing `hunyuan3d/`, `rig/`, and
   `.github/workflows/avatars.yml` (copy `github-workflow-avatars.yml` there).
2. Run the workflow. It builds both in parallel with `fail-fast: false`, so one
   failing does not block the other, and publishes:

   ```
   ghcr.io/<repo>:hunyuan3d          ghcr.io/<repo>:hunyuan3d-<sha>
   ghcr.io/<repo>:rig                ghcr.io/<repo>:rig-<sha>
   ```

   Pin the `-<sha>` tag on the endpoint. There is deliberately no `:latest`:
   RunPod does not roll workers when a moving tag changes, which turns "did my
   fix deploy?" into a guess — that already cost several cycles chasing stale
   TRELLIS workers.
3. Create two endpoints:

   | | GPU | model attachment | volume |
   |---|---|---|---|
   | hunyuan3d | 24 GB **Ada or Ampere** (A5000/4090/L4/A10G) | `tencent/Hunyuan3D-2.1` | required |
   | rig | 24 GB (A5000 verified) | none (weights baked) | shared with stage 1 |

   **24 GB, not the 48 GB the upstream table implies.** Those figures are
   10 GB for shape, 21 GB for texture, and 29 GB for the two *co-resident* —
   and the handler never holds both, dropping the shape model and emptying the
   cache before painting. Peak is therefore the texture stage's 21 GB. The
   headroom on a 24 GB card is thin; if painting runs out of memory, lower
   `max_views` (6 → 4) and `view_size` (512 → 384) before renting a bigger
   card, or send `want_texture=false` for the 10 GB shape-only path.

   **Pick the GPU by architecture, not only by capacity.** RunPod's 48 GB tier
   now includes RTX PRO 6000 Blackwell, which is `sm_120`, and Hunyuan3D-2.1
   pins torch 2.5.1/cu124 — that build stops at `sm_90`. Put the endpoint on a
   Blackwell card and every worker fails its fitness check with

   ```
   CUDA error: no kernel image is available for execution on the device
   ... sm_120 is not compatible with the current PyTorch installation
   ```

   which surfaces as jobs sitting in `IN_QUEUE` forever, looking for all the
   world like a capacity shortage. It is not; no worker can start. Rebuilding
   does not help either, because the torch pin is what upstream's two native
   extensions compile against. Give that endpoint an Ada or Ampere card.

   The rigger is free of this: torch 2.7/cu128 supports Blackwell, and its
   flash-attn is compiled with `sm_120` in the arch list so the card works
   there.

   Ada or Ampere at 24 GB covers both endpoints, which is also where the
   supply is; 48 GB only buys headroom on the texture stage.

   If the GHCR package is private, add a RunPod container-registry auth entry
   with your GitHub username and a PAT carrying `read:packages`.
4. Add to `minetest.conf`, beside the existing TRELLIS keys:

   ```
   runpod_api_key              = ...
   runpod_trellis_endpoint     = ...
   runpod_hunyuan_endpoint     = ...
   runpod_skintokens_endpoint  = ...
   avatar_generator            = trellis | hunyuan3d   (optional default)
   ```

5. Check both are alive before spending a generation on them:

   ```
   python lab/avatar_pipeline.py --probe
   ```

   Each worker's `probe` reports torch, the GPU and its VRAM, whether the
   native extensions imported, and — for the rigger — which weight files are
   actually on disk rather than what the build log claimed.

## Running

```
python lab/avatar_pipeline.py "female dwarf blacksmith in a leather apron"
python lab/avatar_pipeline.py "..." --generator trellis    # switch generator
```

Which generator makes the mesh is a switch: `--generator`, else
`avatar_generator` in minetest.conf, else whichever endpoint is configured,
preferring hunyuan3d. Both return a glb and differ only in how the request is
spelled, so each has a small adapter and the rest of the pipeline is unaware.

Writes into `D:/blockgen-models/avatars/`: the source png, the Hunyuan3D glb,
the rigged glb, and `<slug>_luanti.glb` plus a JSON report of the fit.

Stage 3 alone, on any rigged (or unrigged) glb:

```
python lab/luanti_rig.py input.glb output_luanti.glb
```

## Handoff, and the response-size ceiling

A textured glb does not fit in a RunPod synchronous response — the same limit
that made TRELLIS' 4096 texture come back empty. So stages 1 and 2 hand off **by
path on a shared network volume**, not by payload, and only the finished model
returns inline. Both handlers take `glb_b64` or `glb_path` and return whichever
fits, reporting which in a `delivery` field.

If the rigged glb is too large to come back inline, `avatar_pipeline.py` stops
and says so rather than half-finishing: fetch it from the volume and run stage
3 locally.

## What stage 3 does, and why it is needed at all

SkinTokens returns a good rig and the wrong rig. Good, because the weights are
learned. Wrong, because it predicts its own skeleton — its own bone count,
names and hierarchy — and **no animation**. Luanti does not take an arbitrary
skeleton for a player: every game drives players through six fixed bones and one
fixed timeline, with the ranges hard-coded in mods.

```
stand 0-79   sit 81-160   lay 162-166   walk 168-187   mine 189-198   walk_mine 200-219
```

So stage 3 keeps the half worth keeping: it matches the predicted joints to the
six canonical bones **by where they sit on the body** (an auto-rigger's bone
names are its own convention), collapses the learned weights onto those six,
and attaches the stock animation lifted from
`games/minetest_game/mods/player_api/models/character.b3d`.

Three details decide whether the result works in-game, and all three are
measured rather than assumed:

- **Scale.** Luanti reads glb in node units — the installed moose is 1.31 units
  tall standing on y=0, not 13.1 — so the model is normalised to 1.75, the
  player collisionbox height, feet on the floor.
- **Pose.** The stock skeleton binds arms-down, and a T-posed generation has to
  be posed down to match. Stage 3 detects which it has from the width-to-height
  ratio (T-pose ≈ 0.92, arms-down ≈ 0.35) instead of assuming.
- **One timeline, not six clips.** `set_animation()` indexes frames into a
  single track, so the ranges above only mean anything if all 221 frames export
  as one animation.

The animation transfer is a **delta from the stock bind**, not a copy of its
local rotations: the stock bones carry a 180° bind rotation of their own, and
copying that would import the reference rig's convention along with the motion.

## Verified, and not

**Stage 2 + stage 3 are verified against a real rig.** A TRELLIS glb went to
the SkinTokens endpoint and came back with 52 joints named `bone_0`…`bone_51`
and no animations — the "good rig, wrong rig" this pipeline is built around.
After retargeting, the six bones are populated and left/right symmetric (arms
17 each, legs 3 each), and evaluating the skinning at stock frames raises the
arm for `mine` and folds the legs for `sit`, 1.75 down to 1.35. See
`lab/skintokens_anim.png`.

Two things that first run caught, both worth knowing:

- Joint positions must come from **composed node transforms**. Summing
  translations put 36 of 52 joints in the head and none in the legs: in a bone
  chain each translation is a length in its parent's *rotated* frame, so
  ignoring rotation stacks the whole skeleton along one axis.
- The rigger need not reach the fitted neck. SkinTokens stops at the base of
  the skull, a few centimetres below the neck measured off the mesh, so the
  Head bone gets nothing weighted to it unless the topmost centreline cluster
  is promoted — otherwise the head rides the torso with a dead animation track.

**Stage 1 is built but not yet run end-to-end.** The image builds and the
handler answers `probe`, but no Hunyuan3D generation has completed here yet.

**An inherent limit, not a defect:** Luanti's player skeleton has no elbow or
knee, so collapsing 17 arm joints onto one rigid bone loses articulation and
the shoulder deforms stiffly. The stock Minetest character has exactly the same
constraint; it is the price of dropping into any game with the standard ranges.
