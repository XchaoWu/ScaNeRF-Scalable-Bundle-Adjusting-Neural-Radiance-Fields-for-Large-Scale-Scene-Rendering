#ifndef HELPER_H__
#define HELPER_H__

#include "macros.h"
#include <ATen/ATen.h>
#include <pybind11/pybind11.h>
#include <torch/extension.h>


void ray_aabb_intersection(
    at::Tensor rays_o,
    at::Tensor rays_d,
    at::Tensor aabb_center,
    at::Tensor aabb_size,
    at::Tensor &bounds);

void ray_aabb_intersection_v2(
    at::Tensor rays_o,
    at::Tensor rays_d,
    at::Tensor aabb_center,
    at::Tensor aabb_size,
    at::Tensor &bounds);

    
void proj2pixel_and_fetch_color(
    at::Tensor pts,
    at::Tensor Ks, 
    at::Tensor C2Ws,
    at::Tensor RGBs,
    at::Tensor &fetched_pixels,
    at::Tensor &fetched_colors);

at::Tensor sample_points_contract(
    at::Tensor rays_o, 
    at::Tensor rays_d, 
    at::Tensor &z_vals,
    at::Tensor block_corner,
    at::Tensor block_size,
    at::Tensor occupied_gird);


void sample_points_grid(
    at::Tensor rays_o, 
    at::Tensor rays_d, 
    at::Tensor &z_vals,
    at::Tensor &dists,
    at::Tensor block_corner,
    at::Tensor block_size,
    at::Tensor occupied_gird,
    at::Tensor log2dim);
    
#endif 