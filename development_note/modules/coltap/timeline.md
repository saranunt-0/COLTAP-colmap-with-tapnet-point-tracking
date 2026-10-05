# Module: coltap
<!-- TAPNext++ point tracking as a drop-in replacement for COLMAP's
     correspondence search (feature extraction + matching). -->

---

## [2026-10-05] Stage 1: TAPNext++ -> COLMAP integration for static scenes

**Type**: `feature`
**Status**: `resolved` (stage 1); see "Next steps" for stage 2

### Context
Goal from the user:
1. Seamless integration: same input (image folder / video), same output
   (COLMAP database + sparse model) so every COLMAP module still works.
2. Demo video/GIF comparing COLMAP tracking with TAPNet tracking.
3. Feature selection weight that prefers confident, static points; first
   stage = static scenes only.

Clarifications made while researching:
- "TAPNet++" = **TAPNext++** (DeepMind, CVPR 2026 Findings, arXiv
  2604.10582), a fine-tuned TAPNext (TrecViT-B) checkpoint for long
  (1024-frame) tracking with re-detection. PyTorch-only checkpoint:
  `storage.googleapis.com/dm-tapnet/tapnextpp/tapnextpp_ckpt.pt` (256 px)
  and `.../gresearch/tapnextpp/tapnextpp_512.ckpt` (512 px).
- COLMAP has no "point tracking" module: tracks are the connected components
  of pairwise verified matches. TAPNext++ therefore replaces
  `feature_extractor` + `*_matcher`, not the mapper.
- TAP models need **temporally ordered** frames. Unordered photo collections
  remain a SIFT job (COLTAP falls back to `--tracker sift`).

### Work Done
1. Vendored the TAPNext PyTorch model (`tapnext_torch.py`,
   `tapnext_lru_modules.py`, `pscan.py`, certainty helper) from
   google-deepmind/tapnet@730cda1 (Apache-2.0); only imports changed. Avoids
   pulling `tapnet`'s JAX/TF dependencies.
2. Verified conventions empirically: model I/O is (y, x) in 256-space for
   both checkpoints; pixel-corner origin == COLMAP keypoint convention, so
   conversion is a pure per-axis scale. Query timestep layout (t, y, x).
3. Tracking scheme (`tracking.py`): one tracker instance per query batch
   (same as DeepMind's VOTS TAPNext++ tracker), coverage-driven re-seeding,
   instance cap (3) with handoff of confident points, optional backward pass.
   - Without the cap, instances piled up (9 active, ~9 s/frame on CPU).
     With cap + handoff: 3.6x faster, same accuracy.
4. Database writer (`database.py`) using pycolmap: COLMAP's image reader,
   keypoints, raw matches, COLMAP's own two-view verification. Track
   selection: observation confidence, min track length, static score.
5. CLI mirroring `colmap` + forwarding of all other commands to the binary.
6. Synthetic ray-cast scene with exact GT for quantitative evaluation.
7. Real static clips: Great Wall (drone, 96 frames), Colosseum (drone,
   114 frames), fox (handheld, 50 frames, COLMAP reference poses).
8. Fox investigation (debug, see below) -> seeding fix + SIFT gap bridge.
9. Evaluation metric fix: first rotation metric aligned cameras with a Sim3
   fitted on camera centers only, which is ill-conditioned for near-linear
   trajectories (a unit test showed 2 deg "error" on exact data). Replaced
   by the gauge-free pairwise relative rotation error (RRE). On the
   synthetic scene this flipped the rotation comparison from "COLTAP worse"
   (0.10 vs 0.06 deg, artifact) to "on par/slightly better" (0.022 vs
   0.024 deg).
10. CI compatibility: `scripts/format/python.sh --all` (ruff 0.15.20, root
    config) would reformat the vendored code and mis-sort `coltap` imports.
    Root `ruff.toml`: `extend-exclude` vendored dir + `force-exclude`, and
    `known-first-party = ["coltap"]`. Verified idempotent with 0.15.20.

### Checklist
- [x] TAPNext++ coordinate convention verified against exact GT (synthetic)
- [x] Tracking accuracy on synthetic GT: median 0.36 px (fp32), 0.41 px
      (bf16), 92-94 % < 1 px, visibility precision 99.8 %
- [x] Database readable by unmodified COLMAP mapper (pycolmap 4.2.1)
- [x] Static score rejects independently moving points in a controlled test
- [x] Unit tests (14) pass without network weights; 1 end-to-end test with
      real weights (CLI -> DB -> mapper, 12/12 images) passes, auto-skips
      when the checkpoint is not cached
- [x] CI python format check simulated with ruff 0.15.20: no diff
- [ ] GPU timing (no GPU in this container) — see Unverified Items
- [ ] Dense MVS via `--dense 1` (needs CUDA COLMAP binary)

### Unverified Items
- [ ] `coltap automatic_reconstructor --workspace_path ws --image_path imgs --dense 1`
      on a CUDA machine with the `colmap` binary — expected: `ws/dense/fused.ply`.
- [ ] `--TapTracker.device cuda` speed — expected: well under 1 s/frame.
- [ ] `--TapTracker.resolution 512` checkpoint (4x tokens; not run on CPU).

### Root Cause / Outcome
See `coltap/README.md` for the benchmark tables. Summary: on smooth static
videos COLTAP registers the same images as COLMAP-SIFT with ~2x longer
tracks and 5-20x fewer 3D points; pose accuracy is on par (synthetic GT:
same ATE, RRE 0.022 vs 0.024 deg). On the discontinuous fox photo set it is
complete (50/50) only thanks to the SIFT gap bridge and less accurate than
SIFT (RRE 0.61 vs 0.29 deg).

#### Debug report: fox split into two models (31/50 registered)
- Reproduce: `coltap automatic_reconstructor` on fox (50 frames) -> 2 models.
- Isolate: per-frame kept observations showed frame 31 (`0072.jpg`) with 0
  tracks; consecutive shared tracks 30|31 and 31|32 = 0.
- Root causes (confirmed):
  1. Bug: re-seeding was blocked by `min_keyframe_interval` although
     coverage was 0 -> a frame with no tracks. Fix: `force_seed_coverage`
     (0.4) always seeds below that coverage.
  2. Input: `0054.jpg -> 0072.jpg` is an 18-frame jump; no temporal tracker
     can follow it. Checked COLMAP-SIFT's database: it does not match across
     the jump directly either; the halves are connected only through the
     sequential matcher's quadratic long-range pairs (i <-> i+16, i+32).
- Fix iterations (each re-exported from saved tracks, mapper only):
  a. SIFT on +-3 frames around the break: still 2 models (no cross pairs).
  b. + schedule pairs crossing the break: 1 large model, 39/50 (bridge
     points seen by one image per side -> not triangulable).
  c. + match bridge frames among themselves per side: 1 model, 50/50.
- Verified the bridge never triggers on the smooth sequences (min shared
  consecutive tracks 313 / 316 / 384 vs threshold 30) and the forced seed
  never triggers there (min seeding coverage 0.73), so their results stand.
- Uncertainty: fox reference poses come from COLMAP-SIFT (full res), which
  biases that comparison toward SIFT.

### Fix / Implementation Detail
See `development_note/architecture.md`.

### Assumptions Made
- Input frames are temporally ordered (natural sort of file names).
- Single camera for the whole sequence by default (video); override with
  `--ImageReader.single_camera 0`.
- EXIF orientation is ignored when reading frames (matches COLMAP's raster
  size); a mismatch raises an explicit error.
- Static score uses the two-view epipolar test only: motion along epipolar
  lines or very slow motion is not detected (documented limitation).

### Next steps (stage 2: dynamic scenes)
- Semantic masks (person/car/cloth) via `--ImageReader.mask_path` (already
  honored by seeding and observation filtering).
- Motion segmentation of tracks (e.g. RoboTAP-style clustering on track
  residuals) feeding the same `static_score`.
- Per-observation weights in bundle adjustment (needs a custom BA loop in
  pycolmap; COLMAP's mapper treats all observations equally).

---

## [2026-10-05] Feature density investigation, adaptive sampling, hybrid mode

**Type**: `investigation` + `feature`
**Status**: `resolved` (one open sub-question, see Uncertainty)

### Context
User: TAP gives far fewer features than SIFT, especially on buildings and
windows. Asked (1) which parameters / sampling methods give more features,
(2) a tracking mode switch: sift | tapnet | both. Likes that TAP covers
sky / weak-texture regions.

### Work Done
1. **Funnel measurement** (Great Wall, per image): SIFT 1585 candidates ->
   1469 verified -> 1243 triangulated; TAP 428 -> 426 -> 412. TAP loses
   almost nothing after tracking: the gap is entirely query placement.
2. **Texture-stratified density** (3D-point observations per grid cell by
   Shi-Tomasi tercile): SIFT 0.25 / 2.26 / 8.60 (34x high/low), TAP
   0.89 / 1.37 / 1.39 (1.6x). Root cause: one query per grid cell by design.
3. **Adaptive sampling** (`queries.py`): per-cell quota counting live
   tracks; coverage pass (every cell, any texture) then texture pass (up to
   `max_queries_per_cell` corners >= `texture_threshold` x 95th-pct score);
   `query_detector` shi_tomasi | sift; raster NMS keeps `min_distance` from
   live tracks. Found and fixed while testing: the detector's candidate cap
   was filled by strong corners image-wide, starving weak cells; and the
   global quality floor (1e-3 of max) skipped 10 %-contrast texture ->
   `min_corner_quality` 1e-4 (Great Wall cell coverage 315 -> 336/336).
4. **Hybrid mode** (`sift.py`, `database.py`): COLMAP's own SIFT extraction
   + sequential matching into a temp DB, merged with TAP keypoints (SIFT
   keypoints keep their affine shape, TAP gets identity) and jointly verified
   per pair. Unit test caught passing Nx6 keypoints to verification.
5. **Bug found (mine, from the first round):** my emulation of COLMAP's
   sequential pairing took linear offsets 1..overlap UNION quadratic
   offsets; COLMAP's `quadratic_overlap` uses only 1, 2, 4, ... 2^(overlap-1)
   (src/colmap/controllers/pairing.cc). COLTAP had verified ~2x more pairs
   (1081 vs 545 on Great Wall), so the README claim "identical image pairs"
   was false. Fixed; the static score still votes over the denser union
   (more votes separate moving points better: 15-24 of 80 moving tracks
   survive depending on pair density), but the database gets exactly
   COLMAP's schedule. All COLTAP workspaces re-exported from saved tracks.

### Hypotheses tested for "adaptive TAP-only has worse rotation on GT"
Synthetic GT: adaptive 2D median error 0.33 px (better than uniform 0.40),
but RRE 0.081 deg vs 0.019 deg. Mapper seed spread < 0.002 deg -> systematic.
- Occlusion corners (T-junctions): 71-77 % of bad tracks lie on depth edges
  in both configs, but GT-oracle removal only moves adaptive 0.081 -> 0.063
  (uniform gets slightly worse). Minor contributor, not the cause.
- Correlated errors between neighbouring queries: residual cosine
  similarity within 25 px is 0.29 (uniform) vs 0.26 (adaptive). Not it.
- Gross-error tail: >3 px 0.77 % vs 0.70 %. Not it.
- Lens distortion k1 (error grows at both ends of the sequence = bend):
  k1 -0.0007 vs -0.0002, focal 501.1 vs 499.6. Not it.
- Open. Absolute size is small (0.08 deg); hybrid on uniform TAP is the
  most accurate configuration (0.013 deg vs SIFT 0.024 deg).

### Decisions
- Default TAP sampling stays uniform (accuracy first); adaptive is opt-in
  (`--TapTracker.max_queries_per_cell 4`), default per-instance cap raised
  800 -> 2000 so that one flag is enough.
- Hybrid = uniform TAP + COLMAP SIFT.
- Sky: tracks at infinity cannot be triangulated (COLMAP's min
  triangulation angle) and drifting clouds are not static, so sky tracks
  add no 3D structure; the useful part of TAP's coverage is weak-texture
  *surfaces* (haze-covered terrain, walls, ground).

### Root Cause / Outcome
- Fewer TAP features = query placement (1 per grid cell), not tracking loss.
- Adaptive sampling (`max_queries_per_cell 4`): 2.3-2.9x density, mostly on
  textured cells (Great Wall high-texture 1.36 -> 3.85 obs/cell, Colosseum
  1.25 -> 3.64), longer tracks, no slower (fewer instances); TAP-only pose
  accuracy on GT worse (open question above).
- Hybrid (uniform TAP + COLMAP SIFT): most accurate on GT (RRE 0.013 deg vs
  SIFT 0.024, TAP 0.019; ATE 0.030 % vs 0.043 / 0.060), SIFT-level density
  on texture plus TAP coverage, and fixes the fox case (RRE 0.28 deg vs TAP
  0.70, SIFT 0.29).

### Checklist
- [x] 18 unit tests + 2 end-to-end tests (tapnext, hybrid via the CLI) pass
- [x] CI ruff 0.15.20 format/check clean
- [x] All COLTAP workspaces re-exported with COLMAP's exact pair schedule
- [ ] Root cause of the adaptive-sampling rotation regression (open)
- [ ] GPU timing, 512 px checkpoint (no GPU here)
