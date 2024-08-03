#ifndef VIEW_SELECTION_H__
#define VIEW_SELECTION_H__

#include "macros.h"
#include <ATen/ATen.h>
#include <pybind11/pybind11.h>
#include <torch/extension.h>

void computeViewcost(
    at::Tensor rays_o,
    at::Tensor rays_d,
    at::Tensor pts, 
    at::Tensor ks, 
    at::Tensor rts,
    at::Tensor &costs,
    int height, int width);


void proj2neighbor_forward(
    at::Tensor pts,
    at::Tensor ks, at::Tensor rts,
    at::Tensor nei_views, at::Tensor nei_valid,
    // at::Tensor &proj_depth,
    at::Tensor &nei_origin, at::Tensor &nei_direction,
    at::Tensor &grid);

void proj2neighbor_backward(
    at::Tensor pts, 
    at::Tensor ks, at::Tensor rts,
    at::Tensor nei_views, at::Tensor nei_valid,
    at::Tensor dL_dgrid, 
    at::Tensor &grad_pts, at::Tensor &grad_rts);
#endif 