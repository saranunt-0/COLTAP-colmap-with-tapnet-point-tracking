# COLTAP architecture

COLTAP = COLMAP + TAPNext++ point tracking. Stage 1 targets **static scenes
captured as ordered sequences (video / walk-through photos)**.

## Where TAPNext++ plugs into COLMAP

```
                 stock COLMAP                              COLTAP
images ──► feature_extractor (SIFT)            images ──► coltap feature_tracker
       ──► sequential/exhaustive_matcher                    (TAPNext++ tracks
           (+ geometric verification)                        + COLMAP verification)
                    │                                               │
                    ▼                                               ▼
            database.db  (cameras, images, keypoints, matches, two_view_geometries)
                    │   ── identical schema, written through pycolmap ──
                    ▼
     colmap mapper / global_mapper / bundle_adjuster / image_undistorter /
     patch_match_stereo / stereo_fusion / model_converter ...   (unchanged)
                    ▼
            sparse/0/{cameras,images,points3D}.bin  (+ dense/)
```

The database is the integration seam. Everything COLMAP does after
correspondence search consumes only `cameras`, `images`, `keypoints` and
`two_view_geometries`, so writing those tables from tracks makes every
downstream module work unchanged ("same input, same output").

## Package layout (`coltap/`)

| File | Role |
|------|------|
| `coltap/third_party/tapnext/` | Vendored TAPNext/TAPNext++ PyTorch model (Apache-2.0, upstream commit 730cda1) |
| `coltap/model.py` | Checkpoint download/slimming, preprocessing, one online step, confidence outputs |
| `coltap/queries.py` | Coverage grid + Shi-Tomasi query seeding in uncovered cells |
| `coltap/tracking.py` | Multi-instance online tracking: seeding, instance cap, handoff, optional backward pass |
| `coltap/tracks.py` | `Tracks` container (N x T arrays, npz I/O) |
| `coltap/database.py` | Track selection (confidence, length, static score) and COLMAP database writer |
| `coltap/pipeline.py` | `run_feature_tracker`, `run_sift_baseline`, `run_mapper`, `automatic_reconstruction` |
| `coltap/cli.py` | `coltap` CLI mirroring `colmap` (unknown commands are forwarded to the `colmap` binary) |
| `coltap/visualize.py` | Track rendering, side-by-side COLMAP vs COLTAP MP4/GIF |
| `coltap/evaluate.py` | Reconstruction statistics, ATE + relative rotation error vs ground truth |
| `coltap/synthetic.py` | Ray-cast static textured scene with exact GT poses and 2D tracks |

## Key design decisions

1. **Python on top of pycolmap, no C++ changes.** TAPNext++ is a PyTorch
   model; a C++ integration would need ONNX/TorchScript export of a recurrent
   model plus a COLMAP rebuild. The database seam gives identical outputs
   with zero changes to COLMAP itself. C++/ONNX port is a stage-2 option.
2. **One tracker instance per query batch** (same as DeepMind's own TAPNext++
   VOTS tracker): queries are always on the first frame of an instance, which
   is the regime the model was trained/evaluated in.
3. **Coverage-driven seeding + instance cap + handoff** keeps cost at
   <= `max_active_instances` forward passes per frame while still producing
   long tracks.
4. **Static score from COLMAP's own two-view verification**: a track's
   fraction of verified image pairs in which it is an epipolar inlier.
   Dynamic points violate the background epipolar geometry.
5. **Pairing schedule = COLMAP sequential matcher** (overlap 10 + quadratic),
   so COLTAP and the SIFT baseline are compared on identical image pairs.
