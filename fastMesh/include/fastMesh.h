#ifndef __FAST_MESH__H__
#define __FAST_MESH__H__


#include "tile_structure.h"

void firstHit_cuda(at::Tensor rays_o, at::Tensor rays_d, at::Tensor &z_depth, DeviceData device_data);
void firstEnter_cuda(at::Tensor rays_o, at::Tensor rays_d, at::Tensor &z_depth, DeviceData device_data);
void sample_points_cuda(at::Tensor rays_o, at::Tensor rays_d, at::Tensor &z_vals, at::Tensor t_start, DeviceData device_data);

class fastMesh
{
public:
    DeviceData* buffer;
    float scene_bound[6];

    fastMesh(){}
    ~fastMesh(){}

    void build(std::string model_path){

        buffer = new DeviceData();
        break_into_tiles(model_path, buffer, scene_bound);
    }

    at::Tensor getSceneBound()
    {
        auto options = torch::TensorOptions().dtype(torch::kFloat32);
        at::Tensor scene_info = torch::full({6}, 0, options);
        float* scene_info_ptr = scene_info.contiguous().data_ptr<float>();
        for (int i=0; i<6; i++)
        {
            scene_info_ptr[i] = scene_bound[i];
        }
        return scene_info;
    }

    void fisrtHit(at::Tensor rays_o, at::Tensor rays_d, at::Tensor &z_depth)
    {
        firstHit_cuda(rays_o, rays_d, z_depth, *buffer);
    }

    void firstEnter(at::Tensor rays_o, at::Tensor rays_d, at::Tensor &z_depth)
    {
        firstEnter_cuda(rays_o, rays_d, z_depth, *buffer);
    }

    void sample_points(at::Tensor rays_o, at::Tensor rays_d, at::Tensor t_start, at::Tensor &z_vals)
    {
        sample_points_cuda(rays_o, rays_d, z_vals, t_start, *buffer);
    }

    void destroy()
    {
        cudaFree(buffer->buffer);
    }
};


#endif 