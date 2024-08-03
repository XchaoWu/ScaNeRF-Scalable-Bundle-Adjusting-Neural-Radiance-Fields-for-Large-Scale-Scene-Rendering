#include "macros.h"
#include "cutil_math.h"
#include "cuda_utils.h"
#include <ATen/ATen.h>
#include <stdint.h>
#include <ATen/TensorAccessor.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAUtils.h>
#include <iostream>
#include <stdio.h>
#include <math.h>
#include <torch/extension.h>
#include "interpolation.h"
#include "dda.h"

__global__
void proj2pixel_and_fetch_color_kernel(
    float3* pts, // B 
    float* Ks, // N x 9
    float* C2Ws, // N x 12
    float3* RGBs, // N x H x W  
    float3* fetched_pixels, // B x N 
    float3* fetched_colors, // B x N 每个点从每个view拿到的颜色
    int height, int width)
{
    // blockDim.x = 512  gridDim.x = batch_size, gridDim.y = num_camera 
    int taskIdx = threadIdx.x + blockIdx.x * blockDim.x;
    int total_thread = blockDim.x * gridDim.x; 

    while(taskIdx < gridDim.x)
    {
        float* K = Ks + blockIdx.y * 9;
        float* C2W = C2Ws + blockIdx.y * 12;
        float3* RGB = RGBs + blockIdx.y * height * width;
        float3* fetched_rgb = fetched_colors + taskIdx * gridDim.y + blockIdx.y;
        float3* fetched_loc = fetched_pixels + taskIdx * gridDim.y + blockIdx.y;

        float3 p = pts[taskIdx];

        p.x -= C2W[3];
        p.y -= C2W[7];
        p.z -= C2W[11];

        float x_cam = C2W[0] * p.x + C2W[4] * p.y + C2W[8] * p.z;
        float y_cam = C2W[1] * p.x + C2W[5] * p.y + C2W[9] * p.z;
        float z_cam = C2W[2] * p.x + C2W[6] * p.y + C2W[10] * p.z;

        float pixel_x = K[0] * x_cam + K[1] * y_cam + K[2] * z_cam;
        float pixel_y = K[3] * x_cam + K[4] * y_cam + K[5] * z_cam;
        float pixel_z = K[6] * x_cam + K[7] * y_cam + K[8] * z_cam;

        if (pixel_z > 0)
        {
            pixel_x /= pixel_z;
            pixel_y /= pixel_z;

            if (pixel_x >= 0 && pixel_x <= width-1 && pixel_y >=0 && pixel_y <= height-1)
            {
                Bilinear<float, 3>((float*)RGB, (float*)fetched_rgb, height, width, 
                                    make_float2(pixel_x, pixel_y));
                fetched_loc[0] = make_float3(pixel_x, pixel_y, z_cam);
            }else{
                fetched_rgb[0] = make_float3(0.0f, 0.0f, 0.0f);
                fetched_loc[0] = make_float3(-1.0f, -1.0f, -1.0f);
            }

            
        }else{
            fetched_rgb[0] = make_float3(0.0f, 0.0f, 0.0f);
            fetched_loc[0] = make_float3(-1.0f, -1.0f, -1.0f);
        }

        taskIdx += total_thread;
    }
}

void proj2pixel_and_fetch_color(
    at::Tensor pts,
    at::Tensor Ks, 
    at::Tensor C2Ws,
    at::Tensor RGBs,
    at::Tensor &fetched_pixels,
    at::Tensor &fetched_colors)
{
    int batch_size = pts.size(0);
    int height = RGBs.size(1);
    int width = RGBs.size(2);
    int num_camera = Ks.size(0);

    dim3 num_block = dim3(batch_size, num_camera);

    proj2pixel_and_fetch_color_kernel<<<num_block, NUM_THREAD>>>(
        (float3*)pts.contiguous().data_ptr<float>(),
        Ks.contiguous().data_ptr<float>(),
        C2Ws.contiguous().data_ptr<float>(),
        (float3*)RGBs.contiguous().data_ptr<float>(),
        (float3*)fetched_pixels.contiguous().data_ptr<float>(),
        (float3*)fetched_colors.contiguous().data_ptr<float>(),
        height, width);

    AT_CUDA_CHECK(cudaGetLastError());
    return; 
}


__global__ 
void ray_aabb_intersection_kernel(
    float3* rays_o, 
    float3* rays_d,
    float3* aabb_center,
    float3* aabb_size,
    float2* bounds, 
    int batch_size)
{
    int taskIdx = threadIdx.x + blockIdx.x * blockDim.x;
    int total_thread = blockDim.x * gridDim.x; 

    while(taskIdx < batch_size)
    {
        float3 origin = rays_o[taskIdx];
        float3 direction = rays_d[taskIdx];
        
        bounds[taskIdx] = RayAABBIntersection(origin, direction, aabb_center[0], aabb_size[0] / 2.0f);

        taskIdx += total_thread;
    }
}

void ray_aabb_intersection(
    at::Tensor rays_o,
    at::Tensor rays_d,
    at::Tensor aabb_center,
    at::Tensor aabb_size,
    at::Tensor &bounds)
{
    int batch_size = rays_o.size(0);
    ray_aabb_intersection_kernel<<<NUM_BLOCK(batch_size), NUM_THREAD>>>(
        (float3*)rays_o.contiguous().data_ptr<float>(),
        (float3*)rays_d.contiguous().data_ptr<float>(),
        (float3*)aabb_center.contiguous().data_ptr<float>(),
        (float3*)aabb_size.contiguous().data_ptr<float>(),
        (float2*)bounds.contiguous().data_ptr<float>(), batch_size);

    AT_CUDA_CHECK(cudaGetLastError());
    return; 

}



__global__ 
void ray_aabb_intersection_v2_kernel(
    float3* rays_o,  // B x 3 
    float3* rays_d,  // B x 3 
    float3* aabb_center, //  K x 3
    float3* aabb_size, //  K x 3
    float2* bounds, // B x K 
    int num_aabb, 
    int batch_size)
{
    int taskIdx = threadIdx.x + blockIdx.x * blockDim.x;
    int total_thread = blockDim.x * gridDim.x; 
    int box_idx = blockIdx.y;

    while(taskIdx < batch_size)
    {
        float3 origin = rays_o[taskIdx];
        float3 direction = rays_d[taskIdx];
        
        bounds[taskIdx*num_aabb+box_idx] = RayAABBIntersection(origin, direction, aabb_center[box_idx], aabb_size[box_idx] / 2.0f);

        taskIdx += total_thread;
    }
}

void ray_aabb_intersection_v2(
    at::Tensor rays_o,
    at::Tensor rays_d,
    at::Tensor aabb_center,
    at::Tensor aabb_size,
    at::Tensor &bounds)
{
    int batch_size = rays_o.size(0);
    int num_aabb = aabb_center.size(0);

    ray_aabb_intersection_v2_kernel<<<dim3(NUM_BLOCK(batch_size), num_aabb), NUM_THREAD>>>(
        (float3*)rays_o.contiguous().data_ptr<float>(),
        (float3*)rays_d.contiguous().data_ptr<float>(),
        (float3*)aabb_center.contiguous().data_ptr<float>(),
        (float3*)aabb_size.contiguous().data_ptr<float>(),
        (float2*)bounds.contiguous().data_ptr<float>(), num_aabb, batch_size);

    AT_CUDA_CHECK(cudaGetLastError());
    return; 

}

// __device__ 
// void get_next_region(float3 p, float3 direction,
//                      float3 &new_orgin, float3 &new_direction, 
//                      float3 &region_center, 
//                      float3 &region_size)
// {
//     // outgoing point p 

//     /*
//     |x| = 1; |y| = 1; |z| = 1
//     +x -x +y -y +z -z 
//     */
//     // flag for rotattion 
//     static constexpr float R[6][9] = {{1,0,0,0,0,-1,0,1,0}, // Rx anti-clockwise
//                                       {1,0,0,0,0,1,0,-1,0}, // Rx clockwise
//                                       {0,0,1,0,1,0,-1,0,0}, // Ry anti-clockwise
//                                       {0,0,-1,0,1,0,1,0,0}, // Ry clockwise
//                                       {0,-1,0,1,0,0,0,0,1}, // Rz anti-clockwise
//                                       {0,1,0,-1,0,0,0,0,1}};// Rz clockwise
// /*
//     static constexpr uint8_t flags[24] = {
//        2, // x = 1, |y| <= 1, z >= 1
//        3, // x = 1, |y| <= 1, z <= -1
//        5, // x = 1, |z| <= 1, y >= 1
//        4, // x = 1, |z| <= 1, y <= -1
//        3, // x = -1, |y| <= 1, z >= 1
//        2, // x = -1, |y| <= 1, z <= -1
//        4, // x = -1, |z| <= 1, y >= 1
//        5, // x = -1, |z| <= 1, y <= -1
//        1, // y = 1, |x| <= 1, z >= 1
//        0, // y = 1, |x| <= 1, z <= -1
//        4, // y = 1, |z| <= 1, x >= 1
//        5, // y = 1, |z| <= 1, x <= -1
//        0,1,5,4,
//        0, // z = 1. |x| <= 1, y >= 1
//        1, // z = 1. |x| <= 1, y <= -1
//        3, // z = 1, |y| <= 1, x >= 1
//        2, // z = 1, |y| <= 1, x <= -1
//        1,0,3,2};
    
//     // 找 abs_p 中值最小的维度， 绕着那个axis旋转
//     // 三个维度

//     || <= 1   决定旋转轴，但是不确定  clockwise or anti-clockwise
//     || >= 1 and || = 1 同号， anti-clockwise, 0 异号 clockwise  1
// */ 

//     float3 abs_p = fabs(p);
//     // int offset = (int)(abs_p.x < 1) * 0 + (int)(abs_p.y < 1) * 2 + (int)(abs_p.z < 1) * 4;
    
//     int index = 0;
//     if (abs_p.x < 1)
//         index = (int)(p.y > 0) ^ (int)(p.z > 0) + 0;
//     else if (abs_p.y < 1)
//         index = (int)(p.x > 0) ^ (int)(p.z > 0) + 2;
//     else 
//         index = (int)(p.x > 0) ^ (int)(p.y > 0) + 4;
     
//     // #pragma unroll 
//     // for (int i=0; i<9; i++)
//     //     rotation[i] = R[index][i];
//     float rotation[9] = R[index];


// /*
// x = 1, region center = (1.5, 0, 0)   region size = (1, 2, 2)
// x = -1, region center = (-1.5, 0, 0) region size = (1, 2, 2)
// */
//     if (abs_p.x == 1){
//         region_center = p.x * make_float3(1.5f,0,0);
//         region_size = make_float3(1,2,2);
//     }else if (abs_p.y == 1){
//         region_center = p.y * make_float3(0,1.5f,0);
//         region_size = make_float3(2,1,2);
//     }else{
//         region_center = p.z * make_float3(0,0,1.5f);
//         region_size = make_float3(2,2,1);
//     }

//     // new direction 
//     new_direction.x = rotation[0] * direction.x + rotation[1] * direction.y + rotation[2] * direction.z;
//     new_direction.y = rotation[3] * direction.x + rotation[4] * direction.y + rotation[5] * direction.z;
//     new_direction.z = rotation[6] * direction.x + rotation[7] * direction.y + rotation[8] * direction.z;

//     // new point 
//     p = signf(p) * (abs_p - 1.0f); 
//     new_orgin.x = rotation[0] * p.x + rotation[1] * p.y + rotation[2] * p.z;
//     new_orgin.y = rotation[3] * p.x + rotation[4] * p.y + rotation[5] * p.z;
//     new_orgin.z = rotation[6] * p.x + rotation[7] * p.y + rotation[8] * p.z;

// }

// __device__ 
// void sample_points_single_ray_contract(
//     float3 origin,
//     float3 direction,
//     int num_sample,
//     float3* points,
//     bool* occupied_gird,
//     int3 resolution)
// {

//     // total contract space
//     static constexpr float3 block_center = make_float3(0,0,0);
//     static constexpr float3 block_size = make_float3(4,4,4);
//     static constexpr float3 block_corner = block_center - block_size / 2.0f;

//     // dda 
//     float3 grid_size = block_size / make_float3(resolution);
//     DDASatateScene_v2 dda;

//     // origin is already in contract space 
//     float3 abs_origin = fabs(origin);
//     float3 outside_face = make_float3((float)(abs_origin.x==1), (float)(abs_origin.y==1), (float)(abs_origin.z==1));
//     float3 region_center = outside_face * 1.5f;
//     float3 region_size = 2.0f - outside_face;
//     // bound with first region, not sure if we have second 
//     float2 region_bound = RayAABBIntersection(origin, direction, region_center - region_size/2.0f, region_size/2.0f);

//     // new origin 
//     float3 outgoing_pts = origin + region_bound.y * direction;
//     float3 origin2, direction2, region_center2, region_size2;
//     get_next_region(outgoing_pts, direction, origin2, direction2, region_center2, region_size2);
//     float2 region_bound2 = RayAABBIntersection(origin2, direction2, region_center2 - region_size2/2.0f, region_size2/2.0f);
    
//     dda.init(origin-block_corner, direction, region_bound, resolution, grid_size);
//     float total_length = 0.0f; int count = 0;
//     while(!dda.terminate() && dda.t.x < region_bound.y)
//     {
//         dda.next();
//         uint32_t n = dda.current_tile.x * (resolution.y * resolution.z) + dda.current_tile.y * resolution.z + dda.current_tile.z;
//         if (occupied_gird[n])
//         {
//             float len = dda.t.y - dda.t.x;
//             if (len > 0)
//             {
//                 total_length += len; count++;
//             }
//         }
//         dda.step();
//     }

//     dda.init(origin2-block_corner, direction2, region_bound2, resolution, grid_size);
//     while(!dda.terminate() && dda.t.x < region_bound2.y)
//     {
//         dda.next();
//         uint32_t n = dda.current_tile.x * (resolution.y * resolution.z) + dda.current_tile.y * resolution.z + dda.current_tile.z;
//         if (occupied_gird[n])
//         {
//             float len = dda.t.y - dda.t.x;
//             if (len > 0)
//             {
//                 total_length += len; count++;
//             }
//         }
//         dda.step();
//     }

//     if (count == 0) return;
//     dda.init(origin-block_corner, direction, region_bound, resolution, grid_size);
//     int left_sample = num_sample;
//     int sample_count = 0;
//     while(!dda.terminate() && dda.t.x < region_bound.y)
//     {
//         dda.next();
//         uint32_t n = dda.current_tile.x * (resolution.y * resolution.z) + dda.current_tile.y * resolution.z + dda.current_tile.z;
//         if (occupied_gird[n])
//         {
//             float len = dda.t.y - dda.t.x;
//             if (len > 0)
//             {
//                 int num = min(max((int)(num_sample * len / total_length), 1), left_sample);
//                 if (sample_count == count - 1) 
//                 {
//                     num = left_sample;
//                 }
//                 uniform_sample_bound_v2(z_vals+num_sample-left_sample, dda.t.x, dda.t.y, num);
//                 left_sample = left_sample-num;
//                 sample_count++;
//             }
//         }

//         dda.step();
//     }
//     dda.init(origin2-block_corner, direction2, region_bound2, resolution, grid_size);
//     while(!dda.terminate() && dda.t.x < region_bound2.y)
//     {
//         dda.next();
//         uint32_t n = dda.current_tile.x * (resolution.y * resolution.z) + dda.current_tile.y * resolution.z + dda.current_tile.z;
//         if (occupied_gird[n])
//         {
//             float len = dda.t.y - dda.t.x;
//             if (len > 0)
//             {
//                 int num = min(max((int)(num_sample * len / total_length), 1), left_sample);
//                 if (sample_count == count - 1) 
//                 {
//                     num = left_sample;
//                 }
//                 uniform_sample_bound_v2(z_vals+num_sample-left_sample, dda.t.x, dda.t.y, num);
//                 left_sample = left_sample-num;
//                 sample_count++;
//             }
//         }

//         dda.step();
//     }
// }


__device__ 
void sample_points_single_ray(
    float3 origin,
    float3 direction,
    int num_sample,
    float* z_vals, 
    float3 block_corner, 
    float3 block_size,
    bool* occupied_gird,
    int3 resolution)
{
    float2 bound = RayAABBIntersection(origin, direction, block_corner+block_size/2.0f, block_size/2.0f);

    if (bound.x == -1) return;

    // int3 resolution = make_int3(1 << log2dim.x, 1 << log2dim.y, 1 << log2dim.z);
    float3 grid_size = block_size / make_float3(resolution);

    DDASatateScene_v2 dda;
    dda.init(origin-block_corner, direction, bound, resolution, grid_size);

    float total_length = 0.0f;
    int count = 0;

    while(!dda.terminate())
    {
        dda.next();
        uint32_t n = dda.current_tile.x * (resolution.y * resolution.z) + dda.current_tile.y * resolution.z + dda.current_tile.z;
        // uint32_t n = (dda.current_tile.x << (log2dim.y + log2dim.z)) | (dda.current_tile.y << log2dim.z) | dda.current_tile.z;
        if (occupied_gird[n])
        {
            float len = dda.t.y - dda.t.x;

            if (len > 0)
            {
                total_length += len;
                count++;
            }
        }

        dda.step();
    }

    if (count == 0) return;

    dda.init(origin-block_corner, direction, bound, resolution, grid_size);

    int left_sample = num_sample;
    int sample_count = 0;
    while(!dda.terminate())
    {
        dda.next();
        uint32_t n = dda.current_tile.x * (resolution.y * resolution.z) + dda.current_tile.y * resolution.z + dda.current_tile.z;
        // uint32_t n = (dda.current_tile.x << (log2dim.y + log2dim.z)) | (dda.current_tile.y << log2dim.z) | dda.current_tile.z;
        if (occupied_gird[n])
        {
            float len = dda.t.y - dda.t.x;
            if (len > 0)
            {
                int num = min(max((int)(num_sample * len / total_length), 1), left_sample);
                if (sample_count == count - 1) 
                {
                    num = left_sample;
                }

                uniform_sample_bound_v2(z_vals+num_sample-left_sample, dda.t.x, dda.t.y, num);

                left_sample = left_sample-num;
                sample_count++;
            }
        }

        dda.step();
    }
}

__global__ 
void sample_points_kernel(
    float3* rays_o,
    float3* rays_d,
    int num_sample, 
    float* z_vals,
    float3* block_corner,
    float3* block_size,
    bool* occupied_gird,
    int rx, int ry, int rz,
    int batch_size)
{
    int taskIdx = threadIdx.x + blockIdx.x * blockDim.x;
    int total_thread = blockDim.x * gridDim.x; 

    while(taskIdx < batch_size)
    {

        sample_points_single_ray(rays_o[taskIdx], rays_d[taskIdx], num_sample, z_vals+taskIdx*num_sample,
                                block_corner[0], block_size[0], occupied_gird, make_int3(rx,ry,rz));

        taskIdx += total_thread;
    }
}


at::Tensor sample_points_contract(
    at::Tensor rays_o, 
    at::Tensor rays_d, 
    at::Tensor &z_vals,
    at::Tensor block_corner,
    at::Tensor block_size,
    at::Tensor occupied_gird)
{
    int batch_size = rays_o.size(0);
    int rx = occupied_gird.size(0);
    int ry = occupied_gird.size(1);
    int rz = occupied_gird.size(2);
    int num_sample = z_vals.size(1);

    sample_points_kernel<<<NUM_BLOCK(batch_size), NUM_THREAD>>>(
        (float3*)rays_o.contiguous().data_ptr<float>(),
        (float3*)rays_d.contiguous().data_ptr<float>(),
        num_sample, z_vals.contiguous().data_ptr<float>(),
        (float3*)block_corner.contiguous().data_ptr<float>(),
        (float3*)block_size.contiguous().data_ptr<float>(),
        (bool*)occupied_gird.contiguous().data_ptr<bool>(),
        rx, ry, rz,
        batch_size);
    AT_CUDA_CHECK(cudaGetLastError());
    return;
}


__device__ inline 
void sample_points_sparse_single_ray(
    float3 origin,
    float3 direction,
    int num_sample,
    float* z_vals, 
    float* dists, 
    float3 block_corner, 
    float3 block_size,
    bool* occupied_gird,
    int3 log2dim)
{
    float2 bound = RayAABBIntersection(origin, direction, block_corner+block_size/2.0f, block_size/2.0f);

    if (bound.x == -1) return;

    int3 resolution = make_int3(1 << log2dim.x, 1 << log2dim.y, 1 << log2dim.z);
    float3 grid_size = block_size / make_float3(resolution);

    DDASatateScene_v2 dda;
    dda.init(origin-block_corner, direction, bound, resolution, grid_size);

    float total_length = 0.0f;
    int count = 0;

    while(!dda.terminate())
    {
        dda.next();
        uint32_t n = (dda.current_tile.x << (log2dim.y + log2dim.z)) | (dda.current_tile.y << log2dim.z) | dda.current_tile.z;
        // uint32_t n = dda.current_tile.x * (resolution.y * resolution.z) + dda.current_tile.y * resolution.z + dda.current_tile.z;
        if (occupied_gird[n])
        {
            float len = dda.t.y - dda.t.x;

            if (len > 0)
            {
                total_length += len;
                count++;
            }
        }

        dda.step();
    }

    if (count == 0) return;

    dda.init(origin-block_corner, direction, bound, resolution, grid_size);

    int left_sample = num_sample;
    int sample_count = 0;
    while(!dda.terminate())
    {
        dda.next();
        uint32_t n = (dda.current_tile.x << (log2dim.y + log2dim.z)) | (dda.current_tile.y << log2dim.z) | dda.current_tile.z;
        // uint32_t n = dda.current_tile.x * (resolution.y * resolution.z) + dda.current_tile.y * resolution.z + dda.current_tile.z;
        if (occupied_gird[n])
        {
            float len = dda.t.y - dda.t.x;
            if (len > 0)
            {
                int num = min(max((int)(num_sample * len / total_length), 1), left_sample);
                if (sample_count == count - 1) 
                {
                    num = left_sample;
                }

                uniform_sample_bound_v3(z_vals+num_sample-left_sample, 
                                        dists+num_sample-left_sample,
                                        dda.t.x, dda.t.y, num);

                left_sample = left_sample-num;
                sample_count++;
            }
        }
        dda.step();
    }
}

__global__ 
void sample_points_sparse_kernel(
    float3* rays_o,
    float3* rays_d,
    int num_sample, 
    float* z_vals,
    float* dists,
    float* block_corner,
    float* block_size,
    bool* occupied_gird,
    int* log2dim,
    int batch_size)
{
    int taskIdx = threadIdx.x + blockIdx.x * blockDim.x;
    int total_thread = blockDim.x * gridDim.x; 

    while(taskIdx < batch_size)
    {
        sample_points_sparse_single_ray(rays_o[taskIdx], rays_d[taskIdx], num_sample, 
                                z_vals+taskIdx*num_sample, dists+taskIdx*num_sample,
                                make_float3(block_corner[0], block_corner[1], block_corner[2]), 
                                make_float3(block_size[0],block_size[1],block_size[2]), 
                                occupied_gird, make_int3(log2dim[0],log2dim[1],log2dim[2]));

        taskIdx += total_thread;
    }
}

void sample_points_grid(
    at::Tensor rays_o, 
    at::Tensor rays_d, 
    at::Tensor &z_vals,
    at::Tensor &dists,
    at::Tensor block_corner,
    at::Tensor block_size,
    at::Tensor occupied_gird,
    at::Tensor log2dim)
{
    int batch_size = rays_o.size(0);
    int num_sample = z_vals.size(1);

    sample_points_sparse_kernel<<<NUM_BLOCK(batch_size), NUM_THREAD>>>(
        (float3*)rays_o.contiguous().data_ptr<float>(),
        (float3*)rays_d.contiguous().data_ptr<float>(),
        num_sample, 
        z_vals.contiguous().data_ptr<float>(),
        dists.contiguous().data_ptr<float>(),
        (float*)block_corner.contiguous().data_ptr<float>(),
        (float*)block_size.contiguous().data_ptr<float>(),
        (bool*)occupied_gird.contiguous().data_ptr<bool>(),
        (int*)log2dim.contiguous().data_ptr<int>(),
        batch_size);
    AT_CUDA_CHECK(cudaGetLastError());
    return;
}