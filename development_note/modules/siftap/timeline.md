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

---

## [2026-11-10] Occlusion policies and stabilized focus mode

**Type**: `feature` + `research`
**Status**: `resolved`

### Context
User feedback on SIFTAP:
1. Two options for occluded targets: (a) keep reporting TAPNet's trajectory
   while confidence > x %; (b) drop frames that are occluded or below x and
   continue when the target reappears.
2. Demo "focus mode": a view stabilized on the selected feature, so popping
   or jitter is easy to see (a stable track shows a static feature).

Interpretation (stated to the user): "confidence" = P(reported position
within `--tolerance` px of the truth) from the calibrated fused
uncertainty; "remove the key frame" = report no position for that frame.

### Research findings
- TAPNext++ occlusion head (benchmark): flags 95-98 % of truly hidden
  frames, 99-100 % of visible frames as visible. Its estimate of a hidden
  point: median 2.6-4.1 px, p90 4-15 px; heatmap sigma_global grows with
  the occlusion length and correlates with the error (log-corr 0.43-0.82).
  Calibrated occluded std = 0.513 * sigma_global (calib coverage 94 %,
  held-out 97-100 %).
- The previous fallback (rigid fit of visible support points while the
  centre is hidden) was 4x worse (8.5 vs 2.0 px median): the visible
  neighbours are the ones next to the occluder. Replaced by TAP's own
  estimate.
- The old benchmark had no occlusion that ends inside the frame (targets
  went behind buildings for good), so "resume" could not be measured.
  Added `occl` (seed 4): thin near pillars sweep across the view, 60
  hide-and-reappear episodes of 3-14 frames.
- First implementation: drop mode stopped feeding TAP's hidden estimate.
  The filter covariance ballooned, the gate opened, and SIFT matched the
  occluder (pillars share the facade texture): 13 % of hidden frames leaked
  as "tracked" with a 23 px median error. In hidden frames, accepted SIFT
  matches were on the occluder in 15/17 (predict) and 59/61 (drop) cases.
  Fix: SIFT is ignored while TAP says hidden, and both policies run the
  same filter (fed with TAP's hidden estimate) and differ only in
  reporting. Leak: 3 %, 1.9 px median, 0 % failures.
- Results on `occl` (x = min_confidence, tolerance 4 px):
  predict x=0: hidden frames 1.07 px median, 0.3 % fail, 99 % coverage,
  resume 0 frames; x=50 %: 49 % of hidden frames reported, 0.61 px.
  drop x=50 %: 68 % of visible frames reported, 0 % fail; resume 0.9 frames.
  On `fast` the threshold ranks hidden guesses poorly (4-5 % failures among
  reported hidden frames at any x).

### Focus mode
- `siftap/focus.py`: per target, cubic re-sampling around the fused
  position (translation; areas also rotation/scale) so the tracked point
  stays on the panel centre. COLMAP (+0.5) vs OpenCV pixel conventions are
  handled explicitly.
- Residual jitter = shift between consecutive stabilized crops.
  Phase correlation was tried first and rejected: on smooth, nearly aligned
  crops it reported 1-3 px for a perfect track (it weights every frequency
  equally; high frequencies are noise). Replaced by template matching
  (coarse, pop-proof) + ECC refinement: 0.03-0.12 px on exact tracks;
  injected pops 0.3/1.5/6 px read as 0.29/1.48/5.98 px.
- Measured only between two tracked frames (while hidden the panel shows
  the occluder sliding past). Rises when an occluder enters the window
  (Great Wall hillside p95 3 px).
- Real clips (median / p95 px): arch 0.07/0.18, lawn 0.06/0.19, road
  0.23/0.60, tower 0.16/0.34, haze 0.46/1.03.

### Next steps
- A learned or texture-based ranking of hidden-frame reliability (x ranks
  hidden guesses poorly on fast motion).
- Real-footage occlusion ground truth (hand-labelled re-appearances).
