#ifndef DDA_H__
#define DDA_H__

#include "cutil_math.h"

template<u_int32_t LOG2DIM, uint32_t CLOG2DIM>
struct DDASatate{

    static constexpr uint32_t SIDE = 1u << LOG2DIM;
    static constexpr uint32_t TILE_SIZE = 1u << CLOG2DIM;

    int3 tstep;
    float3 tMax;
    float3 tDelta;
    int3 current_tile; // could be nodes and voxels
    int3 mask;
    float2 t;

    // __device__ void init(float3 origin, float3 direction, float2 t_start)
    // {
    //     tstep = signf(direction);
    //     t = t_start;

    //     tDelta = safe_divide(make_float3(tstep * TILE_SIZE), direction);

    //     float3 pFlt = (origin + t.x * direction) / TILE_SIZE;
        
    //     tMax = ((floor(pFlt) - pFlt + 0.5f) * make_float3(tstep) + 0.5f) * tDelta + t.x;

    //     current_tile = make_int3(pFlt);
    //     current_tile = clamp(current_tile, 0, (int)SIDE-1);
    // }

    __device__ void init(float3 origin, float3 direction, float2 t_start)
    {
        // printf("TILE_SIZE %d\n", TILE_SIZE);

        // [TODO] may should clamp origin ?
        origin = origin + t_start.x * direction;
        current_tile = make_int3(origin / TILE_SIZE);
        current_tile = clamp(current_tile, 0, (int)SIDE-1);

        // printf("current_tile %d %d %d\n",current_tile.x, current_tile.y, current_tile.z);
        // printf("origin %f %f %f\n", origin.x, origin.y, origin.z);

        tstep = signf(direction);

        // printf("tstep: %d %d %d direction %f %f %f\n", 
        //     tstep.x, tstep.y, tstep.z, direction.x, direction.y, direction.z);

        float3 next_boundary = make_float3(current_tile + tstep) * TILE_SIZE;

        if (tstep.x < 0) next_boundary.x += TILE_SIZE;
        if (tstep.y < 0) next_boundary.y += TILE_SIZE;
        if (tstep.z < 0) next_boundary.z += TILE_SIZE;

        // printf("next_boundary: %f %f %f\n", next_boundary.x, next_boundary.y, next_boundary.z);

        t = t_start; // init 

        tMax = fmaxf(safe_divide(next_boundary-origin, direction), 0.0f) + t.x;
        tDelta = fabs(safe_divide(make_float3(TILE_SIZE), direction));

        // printf("tMax: %f %f %f tDelta %f %f %f\n", 
        //        tMax.x, tMax.y, tMax.z, tDelta.x, tDelta.y, tDelta.z);

        // if (tMax.x < t.x || tMax.y < t.x || tMax.z < t.x)
        // {
        //     printf("Error tMax %f %f %f\tt %f\n", tMax.x, tMax.y, tMax.z, t.x);
        // }
        // if (tDelta.x <= 0 || tDelta.y <= 0 || tDelta.z <= 0)
        // {
        //     printf("Error tDelta %f %f %f\n", tDelta.x, tDelta.y, tDelta.z);
        // }
    }

    __device__ void next()
    {
        // if (tMax.x < tMax.y)
        // {
        //     if (tMax.x < tMax.z)
        //     {
        //         mask.x = 1;
        //         t.y = tMax.x;
        //     }else{
        //         mask.z = 1;
        //         t.y = tMax.z;
        //     }
        // }else{
        //     if (tMax.y < tMax.z)
        //     {
        //         mask.y = 1;
        //         t.y = tMax.y;
        //     }else{
        //         mask.z = 1;
        //         t.y = tMax.z;
        //     }  
        // }

		mask.x = int((tMax.x < tMax.y) & (tMax.x <= tMax.z));
		mask.y = int((tMax.y < tMax.z) & (tMax.y <= tMax.x));
		mask.z = int((tMax.z < tMax.x) & (tMax.z <= tMax.y));
        t.y = mask.x ? tMax.x : (mask.y ? tMax.y : tMax.z);
        // if (t.y == t.x)
        // {
        //     printf("tMax: %f %f %f tDelta %f %f %f\n", 
        //         tMax.x, tMax.y, tMax.z, tDelta.x, tDelta.y, tDelta.z);  
        // }
    }

    __device__ void step()
    {
        t.x = t.y;
        tMax += make_float3(mask) * tDelta;
        current_tile += mask * tstep;
    }

    __device__ bool terminate()
    {
        return current_tile.x < 0 || current_tile.y < 0 || current_tile.z < 0 || 
               current_tile.x >= SIDE || current_tile.y >= SIDE || current_tile.z >= SIDE ||
               (tMax.x <= 0 && tMax.y <= 0 && tMax.z <= 0);
    }

    __device__ uint3 indexSpaceLoc()
    {

        return make_uint3(current_tile * TILE_SIZE);
    }
};


#endif 