// #include <torch/extension.h>
// #include <ATen/TensorAccessor.h>
// #include <ATen/cuda/CUDAContext.h>
// #include <ATen/cuda/CUDAContext.h>
// #include <ATen/cuda/CUDAUtils.h>
// #include <ATen/ATen.h>
// #include <cassert>
// #include <cuda.h>
// #include "cuda_fp16.h"
// #include "macros.h"
// #include <cuda_runtime.h>
// #include <thrust/sort.h>
// #include <thrust/execution_policy.h>
// #include <thrust/unique.h>
// #include "cutil_math.h"
// #include "cuda_utils.h"
// #include "decoder.h"

// __device__ inline
// uint32_t hash(int3 loc, int hashmap_size)
// {
//     constexpr uint32_t primes[7] = { 1, 2654435761, 805459861, 3674653429, 2097192037, 1434869437, 2165219737 };
//     uint32_t result = 0;
//     result ^= loc.x * primes[0];
//     result ^= loc.y * primes[1];
//     result ^= loc.z * primes[2];

//     return (hashmap_size - 1) & result;
// }

// __device__ inline  
// void linear_weight(float* weight, float3 offset)
// {
//     weight[0] = (1-offset.x) * (1-offset.y) * (1-offset.z);
//     weight[1] = (1-offset.x) * (1-offset.y) * offset.z;
//     weight[2] = (1-offset.x) * offset.y * (1-offset.z);
//     weight[3] = (1-offset.x) * offset.y * offset.z;
//     weight[4] = offset.x * (1-offset.y) * (1-offset.z);
//     weight[5] = offset.x * (1-offset.y) * offset.z;
//     weight[6] = offset.x * offset.y * (1-offset.z);
//     weight[7] = offset.x * offset.y * offset.z;
// }

// __device__ inline 
// void hash_feature_index(uint32_t* indices, int3 bottom_left_idx, int hashmap_size)
// {
//     indices[0] = hash(bottom_left_idx + make_int3(0,0,0), hashmap_size);
//     indices[1] = hash(bottom_left_idx + make_int3(0,0,1), hashmap_size);
//     indices[2] = hash(bottom_left_idx + make_int3(0,1,0), hashmap_size);
//     indices[3] = hash(bottom_left_idx + make_int3(0,1,1), hashmap_size);
//     indices[4] = hash(bottom_left_idx + make_int3(1,0,0), hashmap_size);
//     indices[5] = hash(bottom_left_idx + make_int3(1,0,1), hashmap_size);
//     indices[6] = hash(bottom_left_idx + make_int3(1,1,0), hashmap_size);
//     indices[7] = hash(bottom_left_idx + make_int3(1,1,1), hashmap_size);
// }

// __device__ inline 
// void hash_features(half2* hashed_features, half2* level_feature, int3 bottom_left_idx, int hashmap_size)
// {
//     uint32_t indices[8];
//     hash_feature_index(indices, bottom_left_idx, hashmap_size);
//     #pragma unroll
//     for (int i=0; i<8; i++)
//     {
//         hashed_features[i] = level_feature[indices[i]];
//     }

// }


// template<int N_LEVELS>
// __device__ inline bool pts_inference(
//     float3 pts, float3 direction,
//     float sample_step, 
//     int num_blocks, // total number of sparse blocks
//     float3* block_corners, 
//     float3* block_sizes, 
//     bool* grid_occupied, 
//     uint64_t* grid_starts, // cus each grid may has its own shape 
//     int3* grid_log2dim,  // num_blocks x 3 
//     half2* features_tables, // num_blocks x n_levels x size x 2
//     int hashmap_size,
//     int3* resolution, // N_LEVELS
//     half* params, // num_blocks 
//     float3 &output_diffuse,
//     float3 &output_specular,
//     float &output_alpha)
// {

//     float total_weight = 0.0f;
//     for (int bidx=0; bidx<num_blocks; bidx++)
//     {
//         float3 block_corner = block_corners[bidx];
//         float3 block_size = block_sizes[bidx];
//         float3 half_size = block_size / 2.0f;
//         float3 block_center = block_corner + half_size;
//         float3 distance = fabs(pts - block_center);
//         int3 cur_grid_log2dim = grid_log2dim[bidx];

//         int3 grid_resolution = make_int3(1 << cur_grid_log2dim.x, 1 << cur_grid_log2dim.y, 1 << cur_grid_log2dim.z);
//         float3 grid_size = block_size / make_float3((float)grid_resolution.x, 
//                                                     (float)grid_resolution.y, 
//                                                     (float)grid_resolution.z); 
        

//         if (distance.x < half_size.x && distance.y < half_size.y && distance.z < half_size.z)
//         {
//             // check if this point is inside occupied grid 
//             int3 grid_loc = make_int3((pts - block_corner) / grid_size);

//             int grid_offset = (grid_loc.x << (cur_grid_log2dim.y + cur_grid_log2dim.z)) | (grid_loc.y << cur_grid_log2dim.z) | grid_loc.z;
            
//             bool* block_grid_occupied = grid_occupied + grid_starts[bidx];
//             // bool* block_grid_occupied = grid_occupied + bidx * grid_resolution.x * grid_resolution.y * grid_resolution.z;

//             float3 dis_w = clamp(half_size - distance, 0.0f, half_size);
//             float blend_weight = dis_w.x * dis_w.y * dis_w.z;
//             if(block_grid_occupied[grid_offset])
//             {   
//                 // N_LEVELS x size x 2 
//                 half2* block_features = features_tables + bidx * (N_LEVELS * hashmap_size);

//                 half feature[N_LEVELS*2];
//                 #pragma unroll
//                 for (int i=0; i<N_LEVELS*2; i++)  
//                 {
//                     feature[i] = __float2half(0.0f);
//                 }

//                 #pragma unroll
//                 for (int i=0; i<N_LEVELS; i++)  
//                 {
//                     int3 level_resolution = resolution[bidx * N_LEVELS + i];
//                     // map to [-1,1]
//                     float3 contract_x = (pts - block_center) / half_size;
//                     contract_x = (contract_x + 2.0f) / 4.0f;
//                     float3 voxel_min_vertex = pts * make_float3(level_resolution-1);
//                     int3 bottom_left_idx = make_int3(voxel_min_vertex);
//                     float3 offset = voxel_min_vertex - make_float3(bottom_left_idx);
//                     float weight[8];
//                     linear_weight(weight, offset);
//                     half2 neighbor_features[8];
//                     hash_features(neighbor_features, block_features+i*hashmap_size, bottom_left_idx, hashmap_size);
                    
//                     half2 interpolated_feature = make_half2(__float2half(0.0f), __float2half(0.0f));
//                     #pragma unroll
//                     for (int j=0; j<8; j++)
//                     {
//                         feature[i*2+0] = __hfma(neighbor_features[j].x, __float2half(weight[j]), feature[i*2+0]);
//                         feature[i*2+1] = __hfma(neighbor_features[j].y, __float2half(weight[j]), feature[i*2+1]);
//                     }
//                 }

//                 float3 diffuse, specular; float sigma;
//                 Decoder decoder(params+bidx*PARAMSIZE);
//                 decoder.inference(feature, direction, diffuse, specular, sigma);
//                 float alpha = 1.0f - expf(-1.0f * sigma * sample_step);
//                 output_diffuse = output_diffuse + blend_weight * alpha * diffuse;
//                 output_specular = output_specular + blend_weight * alpha * specular;
//                 output_alpha = output_alpha + blend_weight * alpha;
//                 total_weight += blend_weight;
//             }else{
//                 total_weight += blend_weight;
//             }
//         }


//     }

//     if (total_weight > 0)
//     {
//         output_diffuse = output_diffuse / total_weight;
//         output_specular = output_specular / total_weight;
//         output_alpha = output_alpha / total_weight;
//         return true;
//     }else{
//         return false;
//     }
// } 

// template<int N_LEVELS>
// __device__ inline 
// void render_single_ray_single_block(
//     float3 origin, float3 direction,
//     int bidx, 
//     float sample_step, 
//     int num_blocks, // total number of sparse blocks
//     float3* block_corners, 
//     float3* block_sizes, 
//     bool* grid_occupied, // 
//     uint64_t* grid_starts, // cus each grid may has its own shape 
//     int3* grid_log2dim, 
//     half2* features_tables, // num_blocks x n_levels x size x 2
//     int hashmap_size,
//     int3* resolution, // N_LEVELS
//     half* params, // num_blocks 
//     float3 &output_diffuse,
//     float3 &output_specular,
//     float &output_depth)
// {
//     float3 block_corner = block_corners[bidx];
//     float3 block_size = block_sizes[bidx];
//     float3 half_size = block_size / 2.0f;
//     float3 block_center = block_corner + half_size;
//     int3 cur_grid_log2dim = grid_log2dim[bidx];
//     int3 grid_resolution = make_int3(1 << cur_grid_log2dim.x, 1 << cur_grid_log2dim.y, 1 << cur_grid_log2dim.z);
//     bool* block_grid_occupied = grid_occupied + grid_starts[bidx];
//     Decoder decoder(params+bidx*PARAMSIZE);
//     half2* block_features = features_tables + bidx * (N_LEVELS * hashmap_size);

//     // [NOTE] origin must be inside the block 
//     float2 bound = RayAABBIntersection(origin, direction, block_center, block_size);
//     assert (bound.x >= 0 && bound.y >= 0);

//     float z_outside = bound.y;
    
//     output_diffuse = make_float3(0,0,0);
//     output_specular = make_float3(0,0,0);
//     output_depth = 0.0f;
//     float transparency = 1.0f; 

//     // rendering foreground first 
//     float z_val = bound.x;

//     while(z_val < z_outside)
//     {
//         float3 pts = origin + z_val * direction;

//         // [-1,1]
//         pts = (pts - block_center) / half_size;

//         int3 grid_loc = make_int3((pts + 1.0) / 2.0f * make_float3(grid_resolution)); 
//         int grid_offset = (grid_loc.x << (cur_grid_log2dim.y + cur_grid_log2dim.z)) | (grid_loc.y << cur_grid_log2dim.z) | grid_loc.z;
        
//         if(block_grid_occupied[grid_offset])
//         {

//             half feature[N_LEVELS*2];
//             #pragma unroll
//             for (int i=0; i<N_LEVELS*2; i++)  
//             {
//                 feature[i] = __float2half(0.0f);
//             }

//             #pragma unroll
//             for (int i=0; i<N_LEVELS; i++) 
//             {
//                 int3 level_resolution = resolution[bidx * N_LEVELS + i];
//                 float3 voxel_min_vertex = (pts + 2.0f) / 4.0f * make_float3(level_resolution-1);
//                 int3 bottom_left_idx = make_int3(voxel_min_vertex);
//                 float3 offset = voxel_min_vertex - make_float3(bottom_left_idx);
//                 float weight[8];
//                 linear_weight(weight, offset);
//                 half2 neighbor_features[8];
//                 hash_features(neighbor_features, block_features+i*hashmap_size, bottom_left_idx, hashmap_size);
//                 #pragma unroll
//                 for (int j=0; j<8; j++)
//                 {
//                     feature[i*2+0] = __hfma(neighbor_features[j].x, __float2half(weight[j]), feature[i*2+0]);
//                     feature[i*2+1] = __hfma(neighbor_features[j].y, __float2half(weight[j]), feature[i*2+1]);
//                 }
//             }

//             float3 diffuse, specular; float sigma;
//             decoder.inference(feature, direction, diffuse, specular, sigma);
//             float alpha = 1.0f - expf(-1.0f * sigma * sample_step);

//             output_diffuse = output_diffuse + transparency * alpha * diffuse;
//             output_specular = output_specular + transparency * alpha * specular; 
//             output_depth = output_depth + transparency * alpha * z_val;

//             transparency = transparency * (1.0f - alpha);

//         }

//         z_val += sample_step;
//         if (transparency <= 0.000001f) return;
//     }

//     // rendering background 
//     float near = z_val;
//     float far = fmaxf(block_size.x, fmaxf(block_size.y, block_size.z)) * 10; // FIXME here
//     constexpr int NUM_BG_SAMPLE = 64;
//     #pragma unroll
//     for (int i=0; i<NUM_BG_SAMPLE; i++)
//     {
//         float t = 1.0f * i / NUM_BG_SAMPLE;

//         float z_val = 1.0f / ( 1.0f / near * (1.0f - t) + 1.0f / far * t );

//         float pts[3];
//         pts[0] = origin.x + z_val * direction.x;
//         pts[1] = origin.y + z_val * direction.y;
//         pts[2] = origin.z + z_val * direction.z;

//         pts[0] = (pts[0] - block_center.x) / half_size.x;
//         pts[1] = (pts[1] - block_center.y) / half_size.y;
//         pts[2] = (pts[2] - block_center.z) / half_size.z;

//         float abs_pts[3];
//         #pragma unroll
//         for (int j=0; j<3; j++)
//         {
//             abs_pts[j] = pts[j];
//             if(pts[j] < 0) abs_pts[j] = -1.0f * pts[j];
//         }

//         int infinity_loc = 0;
//         float infinity_norm = abs_pts[0];
//         #pragma unroll
//         for (int j=1; j<3; j++)
//         {
//             if(abs_pts[j] > infinity_norm)
//             {
//                 infinity_norm = abs_pts[j];
//                 infinity_loc = j;
//             }
//         }

//         #pragma unroll
//         for (int j=0; j<3; j++)
//         {
//             if (j == infinity_loc)
//             {
//                 pts[j] = (2.0f - 1.0f / abs_pts[j]) * pts[j] / abs_pts[j];
//             }else{
//                 pts[j] = pts[j] / infinity_norm;
//             }
//         }

//         float3 contract_x = make_float3(pts[0],pts[1],pts[2]);

//         half feature[N_LEVELS*2];
//         #pragma unroll
//         for (int i=0; i<N_LEVELS; i++) 
//         {
//             int3 level_resolution = resolution[bidx * N_LEVELS + i];
//             float3 voxel_min_vertex = (contract_x + 2.0f) / 4.0f * make_float3(level_resolution-1);
//             int3 bottom_left_idx = make_int3(voxel_min_vertex);
//             float3 offset = voxel_min_vertex - make_float3(bottom_left_idx);
//             float weight[8];
//             linear_weight(weight, offset);
//             half2 neighbor_features[8];
//             hash_features(neighbor_features, block_features+i*hashmap_size, bottom_left_idx, hashmap_size);
//             #pragma unroll
//             for (int j=0; j<8; j++)
//             {
//                 feature[i*2+0] = __hfma(neighbor_features[j].x, __float2half(weight[j]), feature[i*2+0]);
//                 feature[i*2+1] = __hfma(neighbor_features[j].y, __float2half(weight[j]), feature[i*2+1]);
//             }
//         }

//         float3 diffuse, specular; float sigma;
//         decoder.inference(feature, direction, diffuse, specular, sigma);
//         float alpha = 1.0f - expf(-1.0f * sigma * sample_step);

//         output_diffuse = output_diffuse + transparency * alpha * diffuse;
//         output_specular = output_specular + transparency * alpha * specular; 
//         output_depth = output_depth + transparency * alpha * z_val;

//         transparency = transparency * (1.0f - alpha);

//         if (transparency <= 0.000001f) return;
//     }
// }

// template<int N_LEVELS>
// __global__ void rendering_kernel(
//     float3* rays_o, 
//     float3* rays_d,
//     float3* output_diffuse, 
//     float3* output_specular,
//     float* output_depth, 
//     float sample_step, 
//     int num_blocks, // total number of sparse blocks
//     float3* block_corners, 
//     float3* block_sizes, 
//     bool* grid_occupied, // num_blocks x  (2**grid_log2dim)^3
//     uint64_t* grid_starts, // cus each grid may has its own shape 
//     int3* grid_log2dim, 
//     half* features_tables, // num_blocks x n_levels x size x 2
//     int hashmap_size,
//     int3* resolution, // N_LEVELS
//     half* params, // num_blocks 
//     float* scene_center, 
//     float* scene_size, 
//     int batch_size)
// {
//     int taskIdx = threadIdx.x + blockIdx.x * blockDim.x;
//     int total_thread = blockDim.x * gridDim.x; 

//     while (taskIdx < batch_size)
//     {
//         float3 origin = rays_o[taskIdx];
//         float3 direction = rays_d[taskIdx];

//         // float2 bound = RayAABBIntersection(origin, direction, 
//         //     make_float3(scene_center[0],scene_center[1],scene_center[2]), 
//         //     make_float3(scene_size[0],scene_size[1],scene_size[2])/2.0f);
//         // if (bound.x == -1)
//         // {
//         //     taskIdx += total_thread;
//         //     continue;
//         // }

//         float total_weight = 0.0f;
//         float3 final_diffuse = make_float3(0,0,0);
//         float3 final_specular = make_float3(0,0,0);
//         float final_depth = 0.0f;
//         for (int bidx=0; bidx < num_blocks; bidx++)
//         {
//             float3 block_corner = block_corners[bidx];
//             float3 block_size = block_sizes[bidx];
//             float3 half_size = block_size / 2.0f;
//             float3 block_center = block_corner + half_size;
//             float3 distance = fabs(origin - block_center);

//             if (distance.x < half_size.x && distance.y < half_size.y && distance.z < half_size.z)
//             {
//                 float3 diffuse, specular;
//                 float depth;

//                 float3 dis_w = clamp(half_size - distance, 0.0f, half_size);
//                 float blend_weight = dis_w.x * dis_w.y * dis_w.z;

//                 render_single_ray_single_block<N_LEVELS>(origin, direction, bidx, sample_step, num_blocks,
//                         block_corners, block_sizes, grid_occupied, grid_starts, grid_log2dim,
//                         (half2*)features_tables, hashmap_size, resolution, params, 
//                         diffuse, specular, depth);
                
//                 final_diffuse += blend_weight * diffuse;
//                 final_specular += blend_weight * specular;
//                 final_depth += blend_weight * depth;
//             }
//         }

//         if (total_weight > 0)
//         {
//             final_diffuse /= total_weight;
//             final_specular /= total_weight;
//             final_depth /= total_weight;
//         }

//         output_diffuse[taskIdx] = final_diffuse;
//         output_specular[taskIdx] = final_specular;
//         output_depth[taskIdx] = final_depth;

//         taskIdx += total_thread;
//     }
// } 



// void rendering_outdoor_v1_cuda(
//     at::Tensor rays_o,
//     at::Tensor rays_d,
//     at::Tensor &output_diffuse,
//     at::Tensor &output_specular,
//     at::Tensor &output_depth,
//     float sample_step,
//     at::Tensor block_corners,
//     at::Tensor block_sizes,
//     at::Tensor grid_occupied, 
//     at::Tensor grid_starts, 
//     at::Tensor grid_log2dim,
//     at::Tensor features_tables,
//     at::Tensor resolution,
//     at::Tensor params, 
//     at::Tensor scene_center,
//     at::Tensor scene_size)
// {
//     int batch_size = rays_d.size(0);
//     int num_blocks = block_corners.size(0);
//     int hashmap_size = features_tables.size(2);

//     rendering_kernel<16><<<NUM_BLOCK_256(batch_size), NUM_THREAD_256>>>(
//         (float3*)rays_o.contiguous().data_ptr<float>(),
//         (float3*)rays_d.contiguous().data_ptr<float>(),
//         (float3*)output_diffuse.contiguous().data_ptr<float>(),
//         (float3*)output_specular.contiguous().data_ptr<float>(),
//         (float*)output_depth.contiguous().data_ptr<float>(),
//         sample_step, num_blocks, 
//         (float3*)block_corners.contiguous().data_ptr<float>(),
//         (float3*)block_sizes.contiguous().data_ptr<float>(),
//         (bool*)grid_occupied.contiguous().data_ptr<bool>(),
//         (uint64_t*)grid_starts.contiguous().data_ptr<int64_t>(),
//         (int3*)grid_log2dim.contiguous().data_ptr<int>(),
//         (half*)features_tables.contiguous().data_ptr<at::Half>(),
//         hashmap_size,
//         (int3*)resolution.contiguous().data_ptr<int>(),
//         (half*)params.contiguous().data_ptr<at::Half>(),
//         (float*)scene_center.contiguous().data_ptr<float>(),
//         (float*)scene_size.contiguous().data_ptr<float>(), batch_size);

//     AT_CUDA_CHECK(cudaGetLastError());
//     return; 

// }