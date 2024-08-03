#include "macros.h"
#include "cutil_math.h"
#include "cuda_utils.h"
#include "camera.h"
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

#define GAMMA 0.1f

__global__ 
void computeViewcost_kernel(
    float3* pts, // B  world space point 
    float3* rays_o, // B 
    float3* rays_d, // B 
    PinholeCameraManager cams, // N  all cameras 
    // float* mask, // N x B   
    // float thresh, 
    float* costs, // N x B 
    int num_cameras, 
    int height, int width,
    int batch_size)
{
    int taskIdx = threadIdx.x + blockIdx.x * blockDim.x;
    int total_thread = blockDim.x * gridDim.x;

    // give the world 2 camera 
    PinholeCamera cam = cams[blockIdx.y];
    float3 neighbor_origin = cam.camera_center();

    while(taskIdx < batch_size)
    {

        float3 origin = rays_o[taskIdx];
        float3 direction = normalize(rays_d[taskIdx]);
        float3 p = pts[taskIdx];

        float3 uv = cam.world2pixel(p);
        if (uv.z <= 0.001) //  a small value, the point close to cam center
        {
            costs[blockIdx.y * batch_size + taskIdx] = 1.0f;
            taskIdx += total_thread;
            continue;
        }
        float x = uv.x / uv.z;
        float y = uv.y / uv.z;
        if (x <= 0 || x >= width-1 || y <= 0 || y >= height-1)
        {
            costs[blockIdx.y * batch_size + taskIdx] = 1.0f;
            taskIdx += total_thread;
            continue;
        }

        float3 nei_direction = normalize(p - neighbor_origin);
        // float3 nei_direction = normalize(neighbor_origin - p);

        // [0,1]
        float angle_cost = 1.0f - dot(direction, nei_direction);

        // [0,1]
        float dis_cost = fmaxf(0.0f,  1.0f - norm(p - origin) / norm(p - neighbor_origin) );

        float cost = (1.0f - GAMMA) * angle_cost + GAMMA * dis_cost;
        costs[blockIdx.y * batch_size + taskIdx] = cost;
        // if (cost > thresh) mask[blockIdx.y * batch_size + taskIdx] = 0;

        taskIdx += total_thread;
    }
}

void computeViewcost(
    at::Tensor rays_o,
    at::Tensor rays_d,
    at::Tensor pts, 
    at::Tensor ks, 
    at::Tensor rts,
    at::Tensor &costs,
    int height, int width)
{

    CHECK_INPUT(rays_o);
    CHECK_INPUT(rays_d);
    CHECK_INPUT(pts);
    CHECK_INPUT(ks);
    CHECK_INPUT(rts);
    CHECK_INPUT(costs);

    int batch_size = costs.size(1);
    int num_camera = costs.size(0);

    // printf("num camera %d batch_size %d\n", num_camera, batch_size);

    PinholeCameraManager cams( (Intrinsic*)ks.contiguous().data_ptr<float>(), 
                               (Extrinsic*)rts.contiguous().data_ptr<float>(), num_camera);
    
    computeViewcost_kernel<<< dim3(NUM_BLOCK(batch_size), num_camera, 1), NUM_THREAD>>>(
        (float3*)pts.contiguous().data_ptr<float>(), 
        (float3*)rays_o.contiguous().data_ptr<float>(), 
        (float3*)rays_d.contiguous().data_ptr<float>(), cams,
        costs.contiguous().data_ptr<float>(), num_camera, 
        height, width, batch_size);

    AT_CUDA_CHECK(cudaGetLastError());
    return;  
} 


__global__ 
void proj2neighbor_forward_kernel(
    float3* pts, // B  world space point
    PinholeCameraManager cams, // N  all cameras 
    int* nei_views, // B x K 
    bool* nei_valid, // B x K 
    // float* proj_depth, // B x K x 1
    float3* nei_origin, // B x K x 3
    float3* nei_direction, // B x K x 3
    float3* grid, // B x K x 2 
    int batch_size, int num_neighbors)
{
    int taskIdx = threadIdx.x + blockIdx.x * blockDim.x;
    int total_thread = blockDim.x * gridDim.x;

    // printf("taskIdx %d\n", taskIdx);

    while(taskIdx < batch_size * num_neighbors)
    {
        int bidx = taskIdx / num_neighbors;
        // int nidx = taskIdx % num_neighbors;

        int nei_idx = nei_views[taskIdx];
        bool valid = nei_valid[taskIdx];

        // printf("bidx %d nidx %d nei idx  %d\n", bidx, nidx, nei_idx);

        if (!valid)
        {
            taskIdx += total_thread;
            continue;
        }

        PinholeCamera cam = cams[nei_idx];

        float3 pts_cam = cam.world2cam(pts[bidx]);
        float3 pixel = cam.cam2pixel(pts_cam);

        float3 nei_rays_o = cam.camera_center();
        float3 nei_rays_d = make_float3(pts_cam.x / (pts_cam.z+1e-8), 
                                        pts_cam.y / (pts_cam.z+1e-8),
                                        1.0f);
        nei_rays_d = cam.cam2world_rotate(nei_rays_d);

        // [DEBUG]
        // if (nei_idx == 0)
        // {
        //     printf("======= nei_idx %d pixel: %f %f %f\n", nei_idx, pixel.x, pixel.y, pixel.z);
        // }
        // [DEBUG]
        
        // if ((bidx == batch_size - 1) && (nidx == 0))
        // {
        //     printf("nei_idx %d pts %f %f %f uv %f %f %f\n", nei_idx, 
        //     pts[bidx].x, pts[bidx].y, pts[bidx].z,
        //     pixel.x, pixel.y, pixel.z);
        // }

        grid[taskIdx] = pixel;
        // proj_depth[taskIdx] = pixel.z;
        nei_origin[taskIdx] = nei_rays_o;
        nei_direction[taskIdx] = nei_rays_d;


        taskIdx += total_thread;
    }
}

void proj2neighbor_forward(
    at::Tensor pts,
    at::Tensor ks, at::Tensor rts,
    at::Tensor nei_views, at::Tensor nei_valid,
    // at::Tensor &proj_depth,
    at::Tensor &nei_origin, at::Tensor &nei_direction,
    at::Tensor &grid)
{
    int batch_size = pts.size(0);
    int num_neighbors = nei_views.size(1);
    int num_camera = ks.size(0);

    // printf("batch size %d num neighbor %d\n", batch_size, num_neighbors);

    PinholeCameraManager cams( (Intrinsic*)ks.contiguous().data_ptr<float>(), 
                               (Extrinsic*)rts.contiguous().data_ptr<float>(), num_camera);

    proj2neighbor_forward_kernel<<<NUM_BLOCK(batch_size*num_neighbors), NUM_THREAD>>>(
        (float3*)pts.contiguous().data_ptr<float>(), cams,
        (int*)nei_views.contiguous().data_ptr<int>(),
        (bool*)nei_valid.contiguous().data_ptr<bool>(),
        // (float*)proj_depth.contiguous().data_ptr<float>(),
        (float3*)nei_origin.contiguous().data_ptr<float>(),
        (float3*)nei_direction.contiguous().data_ptr<float>(),
        (float3*)grid.contiguous().data_ptr<float>(), batch_size, num_neighbors);

    AT_CUDA_CHECK(cudaGetLastError());
    return;  

}

__global__ 
void proj2neighbor_backward_kernel(
    float3* pts, // B  world space point
    // PinholeCameraManager cams, // N  all cameras 
    float* ks,  // N x 3 x 3
    float* rts, // N x 3 x 4 
    int* nei_views, // B x K 
    bool* nei_valid, // B x K 
    float3* dL_dgrid, // B x K x 3
    // float2* grid, // B x K x 2 
    float3* grad_pts, // B x 3
    float* grad_rts, // N x 3 x 4 
    int batch_size, int num_neighbors)
{
    int taskIdx = threadIdx.x + blockIdx.x * blockDim.x;
    int total_thread = blockDim.x * gridDim.x;


    while(taskIdx < batch_size * num_neighbors)
    {
        int bidx = taskIdx / num_neighbors;
        // int nidx = taskIdx % num_neighbors;

        int nei_idx = nei_views[taskIdx];
        bool valid = nei_valid[taskIdx];

        if (!valid)
        {
            taskIdx += total_thread;

            // // [DEBUG]
            // if (nei_idx == 0)
            // {
            //     printf("======= nei_idx %d\n", nei_idx);
            // }
            // [DEBUG]
            continue;
        }


        float3 grad_in = dL_dgrid[taskIdx];

        float dL_du = grad_in.x, dL_dv = grad_in.y, dL_dw = grad_in.z;
        float* k = ks + nei_idx * 9;
        float* rt = rts + nei_idx * 12;
        float* grad_rt = grad_rts + nei_idx * 12;

        float k1=k[0], k2=k[1], k3=k[2];
        float k4=k[3], k5=k[4], k6=k[5];
        float k7=k[6], k8=k[7], k9=k[8];
        float r1=rt[0], r2=rt[1], r3=rt[2];
        float r4=rt[4], r5=rt[5], r6=rt[6];
        float r7=rt[8], r8=rt[9], r9=rt[10];
        float t1=rt[3], t2=rt[7], t3=rt[11];

        float3 p = pts[bidx];
        float x = p.x, y = p.y, z = p.z;




        float dL_dx = (k1 * r1 + k2 * r4 + k3 * r7) * dL_du + 
                      (k4 * r1 + k5 * r4 + k6 * r7) * dL_dv + 
                      (k7 * r1 + k8 * r4 + k9 * r7) * dL_dw;

        float dL_dy = (k1 * r2 + k2 * r5 + k3 * r8) * dL_du + 
                      (k4 * r2 + k5 * r5 + k6 * r8) * dL_dv + 
                      (k7 * r2 + k8 * r5 + k9 * r8) * dL_dw;

        float dL_dz = (k1 * r3 + k2 * r6 + k3 * r9) * dL_du + 
                      (k4 * r3 + k5 * r6 + k6 * r9) * dL_dv +
                      (k7 * r3 + k8 * r6 + k9 * r9) * dL_dw;


        float dL_dt1 = k1 * dL_du + k4 * dL_dv + k7 * dL_dw;
        float dL_dt2 = k2 * dL_du + k5 * dL_dv + k8 * dL_dw;
        float dL_dt3 = k3 * dL_du + k6 * dL_dv + k9 * dL_dw;
        
        float dL_dr1 = dL_dt1 * x;
        float dL_dr2 = dL_dt1 * y;
        float dL_dr3 = dL_dt1 * z;
        float dL_dr4 = dL_dt2 * x;
        float dL_dr5 = dL_dt2 * y;
        float dL_dr6 = dL_dt2 * z;
        float dL_dr7 = dL_dt3 * x;
        float dL_dr8 = dL_dt3 * y;
        float dL_dr9 = dL_dt3 * z;
        
        /*
        r1 r2 r3 t1 
        */
        atomicAdd(&grad_rt[0],  dL_dr1);
        atomicAdd(&grad_rt[1],  dL_dr2);
        atomicAdd(&grad_rt[2],  dL_dr3);
        atomicAdd(&grad_rt[3],  dL_dt1);
        atomicAdd(&grad_rt[4],  dL_dr4);
        atomicAdd(&grad_rt[5],  dL_dr5);
        atomicAdd(&grad_rt[6],  dL_dr6);
        atomicAdd(&grad_rt[7],  dL_dt2);
        atomicAdd(&grad_rt[8],  dL_dr7);
        atomicAdd(&grad_rt[9],  dL_dr8);
        atomicAdd(&grad_rt[10], dL_dr9);
        atomicAdd(&grad_rt[11], dL_dt3);
    
        atomicAdd(&grad_pts[bidx].x,  dL_dx);
        atomicAdd(&grad_pts[bidx].y,  dL_dy);
        atomicAdd(&grad_pts[bidx].z,  dL_dz);

        taskIdx += total_thread;
    }
}


void proj2neighbor_backward(
    at::Tensor pts, 
    at::Tensor ks, at::Tensor rts,
    at::Tensor nei_views, at::Tensor nei_valid,
    at::Tensor dL_dgrid, 
    at::Tensor &grad_pts, at::Tensor &grad_rts)
{
    int batch_size = pts.size(0);
    int num_neighbors = nei_views.size(1);

    // printf("batch_size %d num_neighbors %d\n", batch_size, num_neighbors);

    proj2neighbor_backward_kernel<<<NUM_BLOCK(batch_size*num_neighbors), NUM_THREAD>>>(
        (float3*)pts.contiguous().data_ptr<float>(),
        (float*)ks.contiguous().data_ptr<float>(),
        (float*)rts.contiguous().data_ptr<float>(),
        (int*)nei_views.contiguous().data_ptr<int>(),
        (bool*)nei_valid.contiguous().data_ptr<bool>(),
        (float3*)dL_dgrid.contiguous().data_ptr<float>(),
        (float3*)grad_pts.contiguous().data_ptr<float>(),
        (float*)grad_rts.contiguous().data_ptr<float>(),
        batch_size, num_neighbors);

    AT_CUDA_CHECK(cudaGetLastError());
    return;  
}