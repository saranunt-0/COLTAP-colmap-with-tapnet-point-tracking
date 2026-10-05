# COLTAP — COLMAP with TAPNext++ point tracking

COLTAP replaces COLMAP's correspondence search (SIFT feature extraction +
pairwise matching) with **dense, long-term point tracks from DeepMind's
[TAPNext++](https://tap-next-plus-plus.github.io/)** for ordered image
sequences (videos, walk-throughs). The tracks are written into a **standard
COLMAP database**, so COLMAP's mapper and every downstream module (bundle
adjustment, undistortion, PatchMatch MVS, model converters, Gaussian-splatting
/ NeRF tools that read COLMAP models) run unchanged.

Three correspondence modes share one interface: `--tracker tapnext` (TAPNext++
tracks), `--tracker sift` (stock COLMAP) and `--tracker hybrid` (both, merged
into one database).

![COLMAP vs COLTAP vs hybrid tracking on a static drone shot](assets/great_wall_compare.gif)

*Tracks of the reconstructed 3D points. Left: COLMAP (SIFT + sequential
matching). Middle: COLTAP (TAPNext++). Right: hybrid, TAPNext++ tracks in color
and SIFT tracks in gray. Same frames, same COLMAP mapper. More in
[`assets/`](assets/): `colosseum_density.mp4` (sampling modes side by side),
synthetic scene with ground truth, fox.*

## How it plugs into COLMAP

COLMAP has no separate "tracking" step: a track is the connected component of
pairwise, geometrically verified feature matches. COLTAP produces the same
database tables directly from TAPNext++ tracks:

```
 stock COLMAP                                COLTAP
 feature_extractor  ─┐                        ┌─ coltap feature_tracker
 sequential_matcher ─┴─► database.db ◄────────┘   (TAPNext++ + COLMAP's own
                          │                         geometric verification)
                          ▼
     mapper / bundle_adjuster / image_undistorter / patch_match_stereo / ...
                          ▼
             sparse/0/{cameras,images,points3D}.bin   (unchanged format)
```

| stock COLMAP | COLTAP equivalent |
|---|---|
| `colmap feature_extractor` + `colmap sequential_matcher` | `coltap feature_tracker` (`--tracker tapnext \| sift \| hybrid`) |
| `colmap automatic_reconstructor` | `coltap automatic_reconstructor` (`--tracker tapnext \| sift \| hybrid`) |
| `colmap mapper` | `coltap mapper` (COLMAP's mapper via pycolmap) |
| any other command (`image_undistorter`, `patch_match_stereo`, `model_converter`, ...) | `coltap <command>` forwards verbatim to the `colmap` binary |

Options keep COLMAP's `--Section.option value` syntax
(`--ImageReader.camera_model`, `--ImageReader.mask_path`,
`--SequentialMatching.overlap`, `--TwoViewGeometry.max_error`,
`--Mapper.*`, ...). COLTAP adds `--TapTracker.*`, `--TrackSelection.*` and
`--GapBridge.*` (see `coltap feature_tracker --help`).

## Install

```bash
pip install -e ./coltap            # pulls pycolmap, torch, opencv, ...
# optional, for MP4/GIF rendering:
pip install -e "./coltap[viz]"
```

The TAPNext++ checkpoint (Apache-2.0) is downloaded on first use to
`~/.cache/coltap` (2.5 GB download, slimmed to 0.8 GB). Set
`COLTAP_CACHE_DIR` to change the location, or pass
`--TapTracker.checkpoint`.

## Quick start

```bash
# Video in, COLMAP model out (sparse/0 in COLMAP's binary format).
coltap automatic_reconstructor \
    --video_path clip.mp4 --VideoReader.stride 3 \
    --image_path ws/images --workspace_path ws

# Or step by step, exactly like COLMAP:
coltap feature_tracker --image_path ws/images --database_path ws/database.db \
    --tracks_path ws/tracks.npz
coltap mapper --image_path ws/images --database_path ws/database.db \
    --output_path ws/sparse
colmap image_undistorter ...      # unchanged COLMAP from here on

# Hybrid: TAPNext++ tracks + COLMAP SIFT in one database.
coltap automatic_reconstructor --tracker hybrid \
    --image_path ws/images --workspace_path ws_hybrid

# Side-by-side tracking video (optional third panel: hybrid).
coltap compare_tracking --image_path ws/images \
    --colmap_path ws_sift/sparse/0 --coltap_path ws/sparse/0 \
    --hybrid_path ws_hybrid/sparse/0 --output_path compare.mp4 compare.gif
```

On a CPU without a GPU use `--TapTracker.precision bf16` if the CPU supports
it (AVX512-BF16/AMX: ~2x faster, median 2D error 0.36 → 0.41 px on our
synthetic benchmark). On CUDA, fp16 is used automatically.

## Tracking modes

| `--tracker` | correspondences | use it when |
|---|---|---|
| `tapnext` (default) | TAPNext++ tracks only (+ SIFT only around breaks, see gap bridge) | smooth video; you want long tracks, a compact model, coverage of weakly textured surfaces |
| `sift` | stock COLMAP: SIFT + sequential matching | unordered photos; maximum density on strong texture |
| `hybrid` | both: SIFT keypoints/matches from COLMAP's own pipeline are appended to the TAPNext++ keypoints, and every image pair is verified jointly | best accuracy in our tests; jumpy or discontinuous captures; you want SIFT's density on buildings plus TAP's coverage elsewhere |

In hybrid mode SIFT and TAPNext++ keypoints are separate keypoints in the same
image, so a physical point can occasionally become two 3D points (one per
source). This did not hurt accuracy in our tests.

## Feature density: why TAPNext++ gives fewer points, and how to get more

**Diagnosis** (Great Wall, per image): SIFT produces 1585 candidates → 1469
verified → 1243 triangulated; TAPNext++ 428 → 426 → 412. TAP tracks are almost
never lost after tracking: the gap comes entirely from **where queries are
placed**. The default places one query per cell of a 24-cell-wide grid, which
spreads points evenly. SIFT instead concentrates on texture: per grid cell, its
3D-point observations go 0.25 / 2.26 / 8.60 from low- to high-texture cells
(34×), TAP's 0.92 / 1.32 / 1.36 (1.5×). So on buildings and windows SIFT has
~6× more points, while on weak texture TAP has ~4× more.

TAPNext++ itself has no detection threshold to lower: every query is a token,
so density is a choice of sampling, paid for in compute (cost per frame grows
with the number of queries, plus one image pass per active tracker instance).

**Knobs** (`--TapTracker.*`):

| option | default | effect |
|---|---|---|
| `max_queries_per_cell` | 1 | **texture-adaptive sampling**: cells are first filled to `queries_per_cell` everywhere (coverage), then textured cells up to this many |
| `texture_threshold` | 0.05 | corner score (× the image's 95th percentile) needed for the extra queries |
| `grid_cells` | 24 | cells along the long side; finer = more queries everywhere, including flat areas |
| `queries_per_cell` | 1 | guaranteed queries per cell (coverage) |
| `query_detector` | `shi_tomasi` | `sift` puts queries at SIFT keypoint locations (fewer, only on texture) |
| `min_corner_quality` | 1e-4 | Shi-Tomasi floor relative to the strongest corner; low so weakly textured cells still get queries |
| `query_min_distance` | cell/4 | spacing between queries and live tracks |
| `max_active_instances` | 3 | concurrent tracker instances (cost) |
| `bidirectional` | off | also track backwards from each query frame (2× cost, more observations per track) |
| `resolution` | 256 | 512 checkpoint: finer localization, ~4× tokens (not benchmarked here) |

**Measured** (Great Wall, 96 frames, same COLMAP pairing and mapper):

| configuration | obs./image | per-cell obs. low / mid / **high** texture | track length | reproj. (px) | agreement with SIFT model |
|---|---:|---|---:|---:|---:|
| COLMAP SIFT | 1243 | 0.25 / 2.26 / **8.60** | 20.8 | 0.40 | — |
| TAP uniform (default) | 403 | 0.92 / 1.32 / **1.36** | 42.6 | 0.54 | 0.35 % / 0.19° |
| TAP `grid_cells 40` | 1143 | 2.76 / 3.70 / **3.66** | 45.4 | 0.57 | 0.40 % / 0.41° |
| TAP `max_queries_per_cell 4` | 923 | 1.07 / 3.32 / **3.85** | **55.6** | 0.51 | 0.27 % / 0.41° |
| TAP `query_detector sift`, 4/cell | 384 | 0.05 / 0.45 / **2.92** | 45.8 | 0.40 | 0.32 % / 0.43° |
| hybrid (uniform TAP + SIFT) | 1652 | 1.14 / 3.63 / **10.0** | 23.6 | 0.42 | 0.11 % / 0.05° |
| hybrid (adaptive TAP + SIFT) | **2157** | 1.32 / 5.53 / **12.4** | 28.5 | 0.43 | 0.15 % / 0.24° |

On the Colosseum (arches and windows), `max_queries_per_cell 4` raises
high-texture density 2.9× (1.25 → 3.64) with tracks spanning 104 of 114
frames, and it was not slower (198 s vs 221 s): denser seeding keeps coverage
up, so fewer new tracker instances are needed.

**Accuracy check with ground truth** (synthetic scene, 48 frames):

| configuration | 3D points | ATE (% extent) | RRE mean / max (°) | 2D median error | correspondence time |
|---|---:|---:|---:|---:|---:|
| COLMAP SIFT | 9792 | 0.043 | 0.024 / 0.062 | — | 11 s |
| TAP uniform | 1114 | 0.060 | 0.019 / 0.054 | 0.40 px | 122 s |
| TAP adaptive (4/cell) | 2053 | 0.050 | 0.081 / 0.222 | **0.33 px** | 134 s |
| **hybrid (uniform TAP + SIFT)** | 10665 | **0.030** | **0.013 / 0.032** | — | 137 s |
| hybrid (adaptive TAP + SIFT) | 11715 | 0.034 | 0.034 / 0.094 | — | 149 s |

Adaptive sampling gives more and individually *more precise* tracks, yet a
larger rotation error on this scene. The mapper is deterministic here (spread
over 5 seeds < 0.002°), so the effect is systematic. Ruled out: occlusion
corners (removing all depth-edge tracks with ground truth only moves 0.081° →
0.063°), correlated errors between neighbouring queries, the gross-error
tail, and lens-distortion estimation. The cause is still open; all values
are small (< 0.1°).

**Recommendation.** For accuracy use `--tracker hybrid` (default uniform TAP
sampling): it was the most accurate configuration on ground truth (rotation
error about half of SIFT's) and has SIFT-level density on buildings. For a
denser TAP-only model, use `--TapTracker.max_queries_per_cell 4`.

**About the sky.** TAP can follow points in the sky and on almost textureless
regions, but sky points are effectively at infinity: COLMAP cannot
triangulate them (minimum triangulation angle 1.5°), so they add no 3D
points, and drifting clouds are not static anyway. The useful part of TAP's
coverage is weakly textured *surfaces* (hazy terrain, plain walls, ground). To
exclude the sky explicitly, pass sky masks via `--ImageReader.mask_path`.

## Which points are used: the selection weight

Every observation and track gets a score; only confident, static tracks
reach the database (`coltap/database.py`):

| stage | rule | option |
|---|---|---|
| observation | confidence = P(visible) × localization certainty (TAPNext++ heads) ≥ 0.5 | `--TrackSelection.min_confidence` |
| track | ≥ 3 confident observations | `--TrackSelection.min_track_length` |
| track | **static score** = fraction of verified image pairs in which the track is an inlier of COLMAP's two-view geometry ≥ 0.5 | `--TrackSelection.min_static_score` |
| ranking | weight = mean confidence × static score (used with a per-image cap) | `--TrackSelection.max_tracks_per_image` |

Points on independently moving objects violate the epipolar geometry of the
static background and get a low static score. Controlled test
(`tests/test_database.py`, default settings): 400 static + 80 independently
moving points over 16 frames → mean static score 0.99 vs 0.29; 396/400 static
tracks kept, 65/80 moving tracks rejected. The 15 that survive move slowly or
along their epipolar lines, which a two-view test cannot see (see
Limitations). The score is a vote over image pairs; it uses COLMAP's pair
schedule plus the dense window i+1 … i+overlap (more votes separate moving
points better), while the database gets exactly COLMAP's schedule.

Masks in COLMAP's format (`--ImageReader.mask_path`, black = ignore) are
honoured both for query placement and for every tracked
observation — the hook for stage-2 semantic masks (people, cars, cloth).

Query points are Shi-Tomasi corners placed per cell of a coarse grid (see
*Feature density*); a new tracker instance is started when live coverage
drops (new content, occlusions). At most 3 instances run at once; when the cap is hit
the oldest instance is retired and its confident points are handed off to the
new one under the same track id, which yields long tracks at bounded cost.

## Results (stage 1: static scenes)

All runs: CPU only (4 cores, bf16), pycolmap 4.2.1, single shared camera
(`SIMPLE_RADIAL`), COLMAP's sequential pair schedule (overlap 10, quadratic)
for every pipeline, unmodified COLMAP incremental mapper. TAP = default
uniform sampling. Reproduce with [`benchmark/compare.py`](benchmark/compare.py)
(runs all three modes and renders the comparison video).

### Synthetic static scene with exact ground truth (48 frames, 640×480)

Ray-traced textured room with occluding boxes (`benchmark/make_synthetic.py`).
See the accuracy table under *Feature density*: hybrid 0.013° / 0.030 %,
TAP 0.019° / 0.060 %, SIFT 0.024° / 0.043 % (mean relative rotation error /
ATE). TAPNext++ 2D accuracy: median error 0.40 px, 90.8 % of observations
within 1 px, 98.6 % within 2 px; 99.8 % of observations reported visible are
truly visible.

### Real static videos

| scene | method | registered | 3D points | obs./image | track length | reproj. (px) | time (s) corr. + mapping |
|---|---|---:|---:|---:|---:|---:|---:|
| Great Wall drone, 96 frames 640×360 | COLMAP | 96/96 | 5744 | 1243 | 20.8 | **0.40** | 13 + 46 |
| | COLTAP | 96/96 | 908 | 403 | **42.6** | 0.54 | ≈270 + 17 |
| | hybrid | 96/96 | 6724 | 1652 | 23.6 | 0.42 | ≈290 + 78 |
| Colosseum drone, 114 frames 640×360 | COLMAP | 114/114 | 8996 | 3510 | 44.5 | 0.38 | 40 + 177 |
| | COLTAP | 114/114 | 470 | 396 | **96.0** | **0.35** | 221 + 25 |
| | hybrid | 114/114 | 9219 | 3861 | 47.7 | 0.38 | 281 + 247 |

Great Wall correspondence times are from an earlier run without other jobs
on the machine (its re-run overlapped with other experiments); hybrid adds
19 s for SIFT and merging. No ground truth for these clips. COLTAP's camera
trajectory agrees with
COLMAP's within 0.35 % (Great Wall) / 0.52 % (Colosseum) of the trajectory
extent and 0.19° / 0.11° mean relative rotation; hybrid within 0.11 % / 0.10 %.

### Discontinuous captures: use hybrid

The instant-ngp *fox* set (50 hand-picked frames, 540×960, large jumps
including an 18-frame gap and a fast roll) is not a continuous video. Scored
against the instant-ngp reference poses (themselves produced by COLMAP-SIFT):

| method | registered | track length | reproj. error | ATE (% extent) | RRE mean / max (°) |
|---|---:|---:|---:|---:|---:|
| COLMAP | 50/50 | 6.2 | **0.54** | 0.26 | 0.29 / 0.83 |
| COLTAP | 50/50 | 6.0 | 0.77 | 1.00 | 0.70 / 1.76 |
| hybrid | 50/50 | 6.4 | 0.62 | **0.25** | **0.28 / 0.77** |

TAP-only needs the SIFT gap bridge to register all 50 images here (31/50
without) and is clearly less accurate; hybrid's poses match SIFT's (slightly
better against this SIFT-made reference), at a higher reprojection error.

## Limitations (read before using)

* **Ordered input only.** TAP models track through time. Unordered photo
  collections should use `--tracker sift` (stock COLMAP).
* **Gap bridge.** Where consecutive frames share < 30 tracks (cut, dropped
  frames), SIFT matches are added in a window around the break and on the
  long-range pairs crossing it (`--GapBridge.*`, on by default). It never
  triggered on the smooth sequences above.
* **Fewer, longer tracks (TAP-only).** ~400 points per frame by default
  versus 1–4k for SIFT; see *Feature density* for the knobs and for hybrid.
  TAP-only pose accuracy was on par on smooth video, worse on the jumpy fox
  set.
* **Speed.** TAPNext++ (ViT-B, ~1 s/frame/instance on 4 CPU cores in bf16) is
  much slower than SIFT on CPU; the mapper is 2–7× faster on COLTAP's compact
  tracks. Hybrid adds SIFT's extraction and its slower mapping. GPU timing has
  not been measured here.
* **No descriptors in the database.** The mapper and later stages do not need
  them; registering *new* images later with vocabulary-tree matching does.
* **Static score = two-view epipolar test.** Motion along epipolar lines or
  very slow motion is not detected; stage 2 adds masks / motion segmentation.

## Package layout

| module | role |
|---|---|
| `coltap/third_party/tapnext/` | vendored TAPNext/TAPNext++ PyTorch model (Apache-2.0, unmodified apart from imports) |
| `coltap/model.py` | checkpoint handling, preprocessing, one online step with visibility/certainty |
| `coltap/queries.py` | coverage grid, uniform / texture-adaptive query sampling |
| `coltap/tracking.py` | multi-instance online tracking with handoff, optional backward pass |
| `coltap/database.py` | selection weights, COLMAP database writer, verification |
| `coltap/bridge.py` | SIFT gap bridge for broken track graphs |
| `coltap/sift.py` | COLMAP's SIFT features + sequential matches for the hybrid mode |
| `coltap/pipeline.py`, `coltap/cli.py` | pipelines and the `coltap` command line |
| `coltap/visualize.py`, `coltap/evaluate.py`, `coltap/synthetic.py` | demos, metrics, synthetic ground truth |

Tests: `cd coltap && pytest` (no network weights needed).

## Credits and licenses

* TAPNext / TAPNext++: Google DeepMind, Apache-2.0
  ([tapnet](https://github.com/google-deepmind/tapnet), commit `730cda1`).
  Zholus et al., *TAPNext: Tracking Any Point (TAP) as Next Token
  Prediction*, 2025; Jung et al., *TAPNext++*, CVPR 2026 Findings.
* Demo clips are derived from third-party sample data: Great Wall and
  Colosseum videos from the [VGGT](https://github.com/facebookresearch/vggt)
  examples, the fox images from
  [instant-ngp](https://github.com/NVlabs/instant-ngp). Check their terms
  before redistributing the clips in `assets/`.
* COLTAP code: BSD-3-Clause, like COLMAP.
