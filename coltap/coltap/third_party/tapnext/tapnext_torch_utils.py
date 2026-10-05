# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Utils for TAPNext torch implementation."""
"""Certainty utilities for TAPNext (subset of upstream tapnext_torch_utils)."""

import torch
import torch.nn.functional as F

def get_window(coord, softmax, radius: int = 8):
  b = coord.shape[0]
  start = torch.floor(coord - radius - 0.5).int()
  start.clamp_(min=0)
  indices = start + torch.arange(radius * 2 + 1, device=softmax.device).repeat(
      b, 1
  )
  # this is to simulate one corner case of jax implementation
  shift = (indices.max(1).values - softmax.shape[1] + 1).clamp(min=0)
  indices -= shift.unsqueeze(1)
  softmax = softmax.gather(dim=1, index=indices)
  return softmax, indices + 0.5


def tracker_certainty(coord_yx, track_logits, radius=8):
  """Computes the certainty of the tracker."""
  shape = coord_yx.shape[:-1]
  coord_yx = coord_yx.flatten(0, -2)
  track_logits = track_logits.flatten(0, -2)
  # track_logits.shape == [b, 512]
  # coord_yx.shape == [b, 2]
  logits_y, logits_x = track_logits.chunk(2, dim=-1)
  track_softmax_y = F.softmax(logits_y, dim=-1)
  track_softmax_x = F.softmax(logits_x, dim=-1)
  sm_y, coord_y = get_window(coord_yx[:, 0:1], track_softmax_y)
  sm_x, coord_x = get_window(coord_yx[:, 1:2], track_softmax_x)
  sm = sm_y[..., :, None] * sm_x[..., None, :]
  grid_x, grid_y = torch.vmap(torch.meshgrid)(coord_x, coord_y)
  # grid_x.shape == [b, N, N]
  grid = torch.stack([grid_y, grid_x], dim=-1)
  in_radius = ((grid - coord_yx[:, None, None]) ** 2).sum(-1) <= (
      (radius**2) + 1e-8
  )
  return (sm * in_radius).sum(-1).sum(-1).reshape(*shape, 1)


