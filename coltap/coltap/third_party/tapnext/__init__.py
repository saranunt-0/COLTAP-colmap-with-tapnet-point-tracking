# SPDX-License-Identifier: Apache-2.0
"""Vendored TAPNext / TAPNext++ PyTorch model from google-deepmind/tapnet.

Upstream: https://github.com/google-deepmind/tapnet
Commit:   730cda1c730877cfedbe01bf87fb1cadb78a565d (2026-09-15)
Files:    tapnet/tapnext/{tapnext_torch,tapnext_lru_modules,pscan}.py and the
          certainty helpers from tapnet/tapnext/tapnext_torch_utils.py.
Changes:  package-relative imports only; no functional modifications.
License:  Apache License 2.0 (see LICENSE in this directory).
"""

from .tapnext_torch import TAPNext, TAPNextTrackingState
from .tapnext_torch_utils import tracker_certainty

__all__ = ["TAPNext", "TAPNextTrackingState", "tracker_certainty"]
