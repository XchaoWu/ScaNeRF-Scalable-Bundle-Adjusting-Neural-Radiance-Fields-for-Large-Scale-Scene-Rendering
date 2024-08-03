#ifndef __TILE_STRUCTURE_H__
#define __TILE_STRUCTURE_H__

#include "buffer.h"
#include "cutil_math.h"
#include "cuda_utils.h"
#include "plyIO.h"
#include "dda.h"

/*
Separate the mesh into tiles to accelerate the ray-triangle intersection
*/
static inline void break_into_tiles(std::string model_path, DeviceData* device_data,
                                    float* scene_bound)
{
    Root* root = new Root();
    HostData* host_data = new HostData();

    std::vector<float2> uv_array;
    read_plyFile(model_path, host_data->vertices, host_data->faces, uv_array);


    float3 scene_min_corner = min_dim( (float3*)host_data->vertices.data(), host_data->vertices.size() );
    float3 scene_max_corner = max_dim( (float3*)host_data->vertices.data(), host_data->vertices.size() );

    scene_bound[0] = scene_min_corner.x; scene_bound[1] = scene_min_corner.y; scene_bound[2] = scene_min_corner.z;
    scene_bound[3] = scene_max_corner.x; scene_bound[4] = scene_max_corner.y; scene_bound[5] = scene_max_corner.z;
    printf("scene_min_corner %f %f %f\n", scene_min_corner.x, scene_min_corner.y, scene_min_corner.z);
    printf("scene_max_corner %f %f %f\n", scene_max_corner.x, scene_max_corner.y, scene_max_corner.z);

    float3 scene_size = scene_max_corner - scene_min_corner;
    float max_size = max(max(scene_size.x, scene_size.y), scene_size.z);

    uint32_t side = (1u << LOG2DIM);

    float tile_size = max_size / side;
    scene_max_corner = scene_min_corner + side * tile_size;    

    host_data->scene_info = make_float4(scene_min_corner.x, scene_min_corner.y, scene_min_corner.z, tile_size);


    printf("start recording face idx for each tile ... \n");
    for (int i=0; i<host_data->faces.size(); i++)
    {
        int3 vidx = host_data->faces[i];
        float3 A = host_data->vertices[vidx.x];
        float3 B = host_data->vertices[vidx.y];
        float3 C = host_data->vertices[vidx.z];

        float3 min_c = fminf(fminf(A, B), C);
        float3 max_c = fmaxf(fmaxf(A, B), C);

        float3 tri_center = (min_c + max_c) / 2.0f;
        float3 tri_size = (max_c - min_c);
        float3 half_trisize = tri_size / 2.0f;
        min_c = tri_center - half_trisize;
        max_c = tri_center + half_trisize;

        // -------------------------
        if (max_c.x <= scene_min_corner.x || max_c.y <= scene_min_corner.y || max_c.z <= scene_min_corner.z ||
            min_c.x >= scene_max_corner.x || min_c.y >= scene_max_corner.y || min_c.z >= scene_max_corner.z ) continue;

        min_c = min_c - scene_min_corner;
        max_c = max_c - scene_min_corner;

        int3 min_idx = make_int3(min_c / tile_size);
        int3 max_idx = make_int3(max_c / tile_size);

        min_idx = clamp(min_idx, 0, side-1);
        max_idx = clamp(max_idx, 0, side-1);

        for (int x=min_idx.x; x<=max_idx.x; x++)
        {
            for (int y=min_idx.y; y<=max_idx.y; y++)
            {
                for (int z=min_idx.z; z<=max_idx.z; z++)
                {
                    root->addFace(make_uint3(x,y,z), i);
                }
            }
        }
    }


    // extract topology 
    printf("start extracting the informations ... \n");
    for (int i=0; i<Root::SIZE; i++)
    {
        if (root->child_list[i] != nullptr)
        {
            host_data->bitmask.setOn(i);
            Tile* t = root->child_list[i];

            uint32_t start = host_data->data.size();
            uint32_t num = t->faces.size();
            host_data->locate.emplace_back(make_uint2(start, num));
            host_data->data.insert(host_data->data.end(), t->faces.begin(), t->faces.end());
        }
    }

    printf("start uploding to GPU ... \n");
    device_data->upload(host_data);

    delete root;
    delete host_data;
}


#endif 