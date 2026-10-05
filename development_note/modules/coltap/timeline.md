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
