# Module: siftap
<!-- SIFT + TAPNext++ fused tracking of user-selected points/areas in one
     video, with calibrated "GPS-style" 95 % uncertainty radii. Not SfM. -->

---

## [2026-11-10] SIFTAP: fused single-target tracking with uncertainty radius

**Type**: `feature` + `research`
**Status**: `resolved` (stage 1, see "Next steps")

### Context
User request (branch `SIFTAP`, stacked on the COLTAP branch):
1. Given one video, the user selects a point or an area; it is tracked.
2. Show the confidence of SIFT and TAPNet as an uncertainty radius
   (GPS-like: small when sure, large when confused).
3. Research how to combine their strengths based on confidence
   (Bayesian / Kalman were the user's guesses); pros and cons.
4. Render demo videos.

### Research findings (synthetic street scene with exact GT)
Benchmark: `siftap/benchmark/evaluate.py`. Ray-traced city with four
surface types (low-texture sky dome, repetitive window facades, photo
posters, ground), occlusions, 1x/2x speed and motion blur. 40 targets per
sequence. Calibration on `calib` + `calib2` (seeds 0, 3); `test` (seed 1)
and `fast` (seed 2, 2x speed, blur) held out.

1. **SIFT is sharp, sparse and heavy-tailed.** Measurement availability
   (oracle window): posters 87 %, facades 19 %, ground 3 %, sky 0 %.
   Median error 0.24 px on posters. Gross mismatches (> 3 px) in 11 % of
   measurements, ~37 px rms. The LS covariance alone covers only 66 % of
   errors in its 95 % circle.
   - Folding P(mismatch) into one Gaussian (moment matching) gave 10 px
     radii on posters with 0.2 px errors (useless). Kept as a mixture:
     core covariance + `p_outlier` from a logistic model on
     log(inliers) and inlier ratio (0 % gross below p = 0.01, 69 % above
     p = 0.5). Drawn radius = 95 % radius of the mixture: 95.1 % coverage.
   - Not detectable by SIFT itself: consistent matches to the next period
     of a periodic texture (ground: 7-10 inliers, ratio 1.0, 10-15 px off).
     Only an independent source (TAP) + gating catches these.
2. **TAPNext++ is dense, smooth, always available, but drifts.** Errors are
   strongly autocorrelated (lag-5 autocorrelation 0.93-0.98 on sky and
   ground) and grow with time (ground 0.6 -> 3 px, sky 1-2 -> 6-8 px over
   60 frames). The heatmap spread (softmax over the 256 coordinate bins,
   trained with CE) is informative (log-corr 0.6 with the error) but is a
   per-frame quantity: on the held-out sky it was *more* peaked than on the
   calibration sky while errors were larger (coverage 64 %).
   - Checked my GT before blaming TAP: the dome is at finite distance, so
     "sky" GT has parallax; TAP follows the dome (4-8 px error) not
     rotation-only motion (54-98 px), so the GT is right and the sky is a
     valid low-texture test (described as such, not as a real sky).
   - The Lucas-Kanade / Shi-Tomasi texture information (1/sqrt(lambda_min)
     of the structure tensor, Cramer-Rao bound of patch alignment) predicts
     TAP's late error better than TAP's own heatmap: log-corr 0.86 vs 0.74
     per target; sky has 25x less texture information than posters.
3. **Fusion design** (`siftap/fusion.py`, default `kalman`):
   constant-velocity Kalman with state [x, y, vx, vy, bx, by]; TAP measures
   position + drift b, SIFT (matches back to the frozen reference frame)
   measures position only -> SIFT makes TAP's drift observable (GPS + IMU
   analogy); drift does a random walk with per-frame std
   kappa * texture_sigma. Chi-square gate (99.7 %), mixture (PDA-style)
   update using SIFT's p_outlier, rejection widens P by 0.25 * y y^T,
   re-acquisition after 2 fully rejected frames.
4. **Calibration criterion**: the drawn 95 % circle must cover 95 % of the
   errors in every cell of sigma-quintile x time-third (and
   texture-quintile x time-third), not just on average.
   - TAP white-noise model: coverage calib 0.99 / test 0.92 / fast 0.95,
     median radius 3.9 px; sky test 0.66.
   - TAP + texture drift: 0.96 / 0.91 / 0.98, median radius 1.3 px; sky
     test 1.00 at ~13 px, posters ~1 px (test posters 0.83: slightly
     overconfident there). Heatmap weight drops to 0.19: most of TAP's
     error is drift, not per-frame noise.

5. **Fused coverage was ~90 %, not 95 %.** Hypothesis "the filter
   averages autocorrelated TAP noise": falsified (accel_sigma 1.5 -> 6 moved
   coverage 90 -> 91 %). Diagnosis: the gap is on ground (76 %), where SIFT
   measurements are 1-3 px off while claiming 0.4 px and pass the gate.
   Fix: same texture cue for SIFT, cov += (c * texture_sigma)^2. The fit
   then gives the LS covariance almost no weight (scale 0.1, floor 0.5 px,
   c = 35): texture predicts SIFT's error better than its fit residuals.
   Fused coverage on the calibration sequences 92 / 91.5 % (ground 78-83 %).

### Fusion comparison (held-out `test` / `fast`)
Full tables in `siftap/README.md` (`--stage tables`).
- Default Kalman: median error 0.95 / 0.54 px, truth inside circle
  89 % / 92 %; TAPNext++ alone 0.97 / 0.55 px, 50 % / 69 %.
- Accuracy gain from SIFT is local: posters 0.35 vs 0.51 px on `test`.
- Naive product of Gaussians: worse than TAP alone (1.50 px, 38 %);
  Kalman without gate: 2.43 px, 18 % failures.
- TAP-only Kalman with drift: 90 % / 97 % coverage, smaller radius than
  the fusion (SIFT wrong-window conflicts on facades widen the circle).
- Remaining gaps: posters 82-84 %, ground 79 % (test).

### Errors / dead ends
- Kalman overshoot after a jump (filter explained the jump as velocity,
  reported a tiny radius while a confident source disagreed) -> covariance
  inflation on rejection; the radius now covers the conflict.
- Moment-matched SIFT mismatch model -> replaced by the mixture (above).
- TAP drift proportional to the heatmap sigma: kappa fitted ~0 (heatmap does
  not see drift) -> texture-driven drift.
- One calibration sequence was not enough for the drift (its sky errors
  happened to be constant) -> added `calib2`.
- Calibration stage must build its "raw" models with explicit unit
  parameters once the fitted values became library defaults.

### Next steps
- Real-video calibration: the texture term depends on image contrast/noise
  (compression); fit kappa on a real clip with hand-labelled points.
- Multimodal posterior (particle filter / Gaussian sum) for repetitive
  structure where both hypotheses must be kept.
- Use TAP's visibility head for occlusion-aware gating; dynamic objects.
