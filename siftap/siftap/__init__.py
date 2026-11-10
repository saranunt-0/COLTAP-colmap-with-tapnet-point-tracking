# SPDX-License-Identifier: BSD-3-Clause
"""SIFTAP: track user-selected points or areas in a video by fusing SIFT and
TAPNext++ with calibrated uncertainties (a Kalman filter with gating).

Each tracker reports a position *and* a covariance for the target. The
covariances are calibrated against ground truth so that the drawn
"uncertainty radius" behaves like a GPS accuracy circle: it contains the true
position ~95 % of the time, small when the tracker is sure and large when it
is confused.
"""

__version__ = "0.1.0"
