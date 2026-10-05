# SPDX-License-Identifier: BSD-3-Clause
"""COLTAP: COLMAP with TAPNext++ point tracking.

COLTAP replaces COLMAP's correspondence search (feature extraction + matching)
for ordered image sequences with dense, long-term TAPNext++ point tracks and
writes them into a standard COLMAP database, so every downstream COLMAP module
(mapper, bundle adjuster, undistorter, MVS, ...) runs unchanged.
"""

__version__ = "0.1.0"
