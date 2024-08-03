#include "macros.h"
#include "cuda_utils.h"
#include "buffer.h"
#include "dda.h"

__device__ __constant__ float4 scene_info;

/// @brief Return the number of lower set bits in mask up to but excluding the i'th bit
static inline __device__ uint32_t countOn(uint32_t i, uint64_t* mWords)
{
    uint32_t n = i >> 6, sum = CountOn( mWords[n] & ((uint64_t(1) << (i & 63u))-1u) );
    for (const uint64_t* w = mWords; n--; ++w) sum += CountOn(*w);
    return sum;
}

    /// @brief Return true if the given bit is set.
static inline __device__ bool isOn(uint32_t n, uint64_t* mWords)
{ 
    return 0 != (mWords[n >> 6] & (uint64_t(1) << (n & 63))); 
}


__device__ 
void sample_points_along_ray(float3 ray_o, float3 ray_d, float* z_vals, float t_start, 
                             int num_sample, DeviceData device_data)
{
    if (t_start == -1.0f) return;
    uint64_t* bitmask = device_data.getBitmask();
    float3 local_ray_o = (ray_o - make_float3(scene_info.x, scene_info.y, scene_info.z)) / scene_info.w;
    t_start = t_start / scene_info.w;
    DDASatate<LOG2DIM, 0> dda;

    int count = 0;
    float total_length = 0;

    dda.init(local_ray_o, ray_d, make_float2(t_start,0));
    while (!dda.terminate())
    {
        dda.next();
        uint3 loc = dda.indexSpaceLoc();
        uint32_t n = (loc.x << (2 * LOG2DIM)) | (loc.y << LOG2DIM) | loc.z;
        
        if (isOn(n, bitmask))
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
    dda.init(local_ray_o, ray_d, make_float2(t_start,0));

    int left_sample = num_sample; 
    int sample_count = 0;
    while (!dda.terminate())
    {
        dda.next();
        uint3 loc = dda.indexSpaceLoc();
        uint32_t n = (loc.x << (2 * LOG2DIM)) | (loc.y << LOG2DIM) | loc.z;
        
        if (isOn(n, bitmask))
        {
            float len = dda.t.y - dda.t.x;
            if (len > 0)
            {
                int num = min(max((int)(num_sample * len / total_length), 1), left_sample);
                if (sample_count == count - 1) num = left_sample;

                uniform_sample_bound_v2(z_vals+num_sample-left_sample, dda.t.x, dda.t.y, num);

                left_sample = left_sample-num;
                sample_count++;
            }
        }
        dda.step();
    }

    for (int i=0; i<num_sample; i++)
    {
        z_vals[i] = z_vals[i] * scene_info.w;
    }
}

__global__ 
void sample_points_kernel(
    float3* rays_o,
    float3* rayd_d,
    float* z_vals,
    float* t_start,
    int num_sample, 
    DeviceData device_data,
    int batch_size)
{
    int taskIdx = threadIdx.x + blockIdx.x * blockDim.x;
    int total_thread = blockDim.x * gridDim.x; 
    while (taskIdx < batch_size)
    {
        sample_points_along_ray(rays_o[taskIdx], rayd_d[taskIdx], z_vals+taskIdx*num_sample, t_start[taskIdx], num_sample, device_data);
        taskIdx += total_thread;
    }
}

void sample_points_cuda(at::Tensor rays_o, at::Tensor rays_d, at::Tensor &z_vals, at::Tensor t_start, DeviceData device_data)
{
    int batch_size = rays_o.size(0);
    int num_sample = z_vals.size(1);

    cudaMemcpyToSymbol( scene_info.x, device_data.buffer, sizeof(float4), 0, cudaMemcpyDeviceToDevice);

    sample_points_kernel<<<NUM_BLOCK(batch_size), NUM_THREAD>>>(
        (float3*)rays_o.contiguous().data_ptr<float>(),
        (float3*)rays_d.contiguous().data_ptr<float>(),
        z_vals.contiguous().data_ptr<float>(), 
        t_start.contiguous().data_ptr<float>(),
        num_sample, device_data, batch_size);
    AT_CUDA_CHECK(cudaGetLastError());
    return; 
}


__device__ 
float singleRay_firstEnter(uint32_t taskIdx, float3 ray_o, float3 ray_d, DeviceData device_data)
{
    float3* vertices = device_data.getVertices();
    int3* faces = device_data.getFaces();
    uint64_t* bitmask = device_data.getBitmask();
    uint32_t* tileFaceIdx = device_data.getTileFaceIdx();
    uint2* startNum = device_data.getStartNum();
    // [FIXME] camera must be inside the scene 
    float3 local_ray_o = (ray_o - make_float3(scene_info.x, scene_info.y, scene_info.z)) / scene_info.w;

    DDASatate<LOG2DIM, 0> dda;
    dda.init(local_ray_o, ray_d, make_float2(0,0));
    float t_near = INF;

    while (!dda.terminate())
    {
        dda.next();
        // printf("taskIdx %d\n", taskIdx);
        uint3 loc = dda.indexSpaceLoc();
        uint32_t n = (loc.x << (2 * LOG2DIM)) | (loc.y << LOG2DIM) | loc.z;
        
        // printf("loc %d %d %d\n", loc.x, loc.y, loc.z);
        
        if (isOn(n, bitmask))
        {
            uint2 start_and_num = startNum[countOn(n, bitmask)];

            uint32_t start = start_and_num.x;
            uint32_t num = start_and_num.y;
            uint32_t* faceidx = tileFaceIdx + start;

            float3 min_corner = make_float3(INF, INF, INF);
            float3 max_corner = make_float3(-INF, -INF, -INF);

            for(int i=0; i<num; i++)
            {
                int3 vidx = faces[faceidx[i]];
                float3 A = vertices[vidx.x];
                float3 B = vertices[vidx.y];
                float3 C = vertices[vidx.z];

                float3 cur_max_corner = fmaxf(fmaxf(A,B),C);
                float3 cur_min_corner = fminf(fminf(A,B),C);

                min_corner = fminf(min_corner, cur_min_corner);
                max_corner = fmaxf(max_corner, cur_max_corner);

                // float3 center = (max_corner + min_corner) / 2.0f;
                // float3 size = max_corner - min_corner;

                // float2 bound = RayAABBIntersection(ray_o, ray_d, center, size / 2.0f);

                // if (bound.x > 0 && bound.x < t_near) t_near = bound.x;
            }

            float3 center = (max_corner + min_corner) / 2.0f;
            float3 size = max_corner - min_corner;

            float2 bound = RayAABBIntersection(ray_o, ray_d, center, size / 2.0f);

            if (bound.x >= 0) return bound.x;
            // if (t_near != INF) return t_near;
        }
        dda.step();
    }
    return 0.0f;

}

__global__ 
void firstEnter_kernel(
    float3* rays_o,
    float3* rayd_d,
    float* z_depth,
    DeviceData device_data,
    int batch_size)
{
    int taskIdx = threadIdx.x + blockIdx.x * blockDim.x;
    int total_thread = blockDim.x * gridDim.x; 
    while (taskIdx < batch_size)
    {

        z_depth[taskIdx] = singleRay_firstEnter(taskIdx, rays_o[taskIdx], rayd_d[taskIdx], device_data );

        taskIdx += total_thread;
    }
}

void firstEnter_cuda(at::Tensor rays_o, at::Tensor rays_d, at::Tensor &z_depth, DeviceData device_data)
{
    int batch_size = rays_o.size(0);

    cudaMemcpyToSymbol( scene_info.x, device_data.buffer, sizeof(float4), 0, cudaMemcpyDeviceToDevice);

    firstEnter_kernel<<<NUM_BLOCK(batch_size), NUM_THREAD>>>(
        (float3*)rays_o.contiguous().data_ptr<float>(),
        (float3*)rays_d.contiguous().data_ptr<float>(),
        z_depth.contiguous().data_ptr<float>(), device_data, batch_size);

    AT_CUDA_CHECK(cudaGetLastError());
    return; 
}


__device__ 
float singleRay_firstHit(uint32_t taskIdx, float3 ray_o, float3 ray_d, DeviceData device_data)
{
    float3* vertices = device_data.getVertices();
    int3* faces = device_data.getFaces();
    uint64_t* bitmask = device_data.getBitmask();
    uint32_t* tileFaceIdx = device_data.getTileFaceIdx();
    uint2* startNum = device_data.getStartNum();
    // to index space 
    // [FIXME] camera must be inside the scene 
    float3 local_ray_o = (ray_o - make_float3(scene_info.x, scene_info.y, scene_info.z)) / scene_info.w;

    // uint32_t side = 1u << LOG2DIM;
    // if (local_ray_o.x < -0.1f || local_ray_o.x > side + 0.1f ||
    //     local_ray_o.y < -0.1f || local_ray_o.y > side + 0.1f ||
    //     local_ray_o.z < -0.1f || local_ray_o.z > side + 0.1f) return 0.0f;

    DDASatate<LOG2DIM, 0> dda;
    dda.init(local_ray_o, ray_d, make_float2(0,0));
    float t_near = INF;
    while (!dda.terminate())
    {
        dda.next();
        // printf("taskIdx %d\n", taskIdx);
        uint3 loc = dda.indexSpaceLoc();
        uint32_t n = (loc.x << (2 * LOG2DIM)) | (loc.y << LOG2DIM) | loc.z;
        
        // printf("loc %d %d %d\n", loc.x, loc.y, loc.z);
        
        if (isOn(n, bitmask))
        {
            uint2 start_and_num = startNum[countOn(n, bitmask)];

            uint32_t start = start_and_num.x;
            uint32_t num = start_and_num.y;
            uint32_t* faceidx = tileFaceIdx + start;
            for(int i=0; i<num; i++)
            {
                int3 vidx = faces[faceidx[i]];
                float3 A = vertices[vidx.x];
                float3 B = vertices[vidx.y];
                float3 C = vertices[vidx.z];

                float3 AB = B - A;
                float3 BC = C - B;
                float3 tri_normal = normalize(cross(AB, BC));
                float sim = dot(ray_d, tri_normal);

                float t; 
                if (sim > 0)
                    t = RayTriangleIntersection(ray_o, ray_d, A, C, B, 0.0f);
                else 
                    t = RayTriangleIntersection(ray_o, ray_d, A, B, C, 0.0f);

                if (t > 0 && t < t_near)
                {
                    t_near = t;
                }
            }

            if (t_near != INF) return t_near;
        }
        dda.step();
    }
    return 0.0f;
}

__global__ 
void firstHit_kernel(
    float3* rays_o,
    float3* rayd_d,
    float* z_depth,
    DeviceData device_data,
    int batch_size)
{
    int taskIdx = threadIdx.x + blockIdx.x * blockDim.x;
    int total_thread = blockDim.x * gridDim.x; 
    while (taskIdx < batch_size)
    {

        z_depth[taskIdx] = singleRay_firstHit(taskIdx, rays_o[taskIdx], rayd_d[taskIdx], device_data );

        taskIdx += total_thread;
    }
}

void firstHit_cuda(at::Tensor rays_o, at::Tensor rays_d, at::Tensor &z_depth, DeviceData device_data)
{
    int batch_size = rays_o.size(0);

    cudaMemcpyToSymbol( scene_info.x, device_data.buffer, sizeof(float4), 0, cudaMemcpyDeviceToDevice);

    firstHit_kernel<<<NUM_BLOCK(batch_size), NUM_THREAD>>>(
        (float3*)rays_o.contiguous().data_ptr<float>(),
        (float3*)rays_d.contiguous().data_ptr<float>(),
        z_depth.contiguous().data_ptr<float>(), device_data, batch_size);

    AT_CUDA_CHECK(cudaGetLastError());
    return; 
}