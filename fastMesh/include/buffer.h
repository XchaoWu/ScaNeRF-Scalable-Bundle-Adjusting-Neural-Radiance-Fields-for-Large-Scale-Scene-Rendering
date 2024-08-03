#ifndef __BUFFER_H____
#define __BUFFER_H____

#include "macros.h"
#include "tile.h"

struct HostData{
    float4 scene_info; // 0             16Byte
    std::vector<uint32_t> data; // 1  4Byte
    Mask<Root::SIZE> bitmask; // 2    8Byte
    std::vector<uint2> locate; // 3   8Byte
    std::vector<float3> vertices; // 4  12 Byte 
    std::vector<int3> faces; // 5     12 Byte

    uint8_t* buffer;

    HostData(){}
    ~HostData(){ delete buffer; }

    void remap_mem(uint64_t* offset)
    {   
        this->getOffset(offset);
        buffer = new uint8_t[offset[6]];

        std::memcpy(buffer+offset[0], &scene_info.x, offset[1]-offset[0]);
        std::memcpy(buffer+offset[1], data.data(), offset[2]-offset[1]);
        std::memcpy(buffer+offset[2], &bitmask, offset[3]-offset[2]);
        std::memcpy(buffer+offset[3], locate.data(), offset[4]-offset[3]);
        std::memcpy(buffer+offset[4], vertices.data(), offset[5]-offset[4]);
        std::memcpy(buffer+offset[5], faces.data(), offset[6]-offset[5]);
    }
    void align_mem(uint64_t &offset, uint64_t align_bytes)
    {
        offset = ((offset / align_bytes) + 1) * align_bytes;
    }

    void getOffset(uint64_t* offset)
    {
        offset[0] = 0;
        offset[1] = sizeof(float4) + offset[0]; 
        offset[2] = sizeof(uint32_t) * data.size() + offset[1];
        align_mem(offset[2], 24);
        offset[3] = sizeof(Mask<Root::SIZE>) + offset[2];
        offset[4] = sizeof(uint2) * locate.size() + offset[3];
        offset[5] = sizeof(float3) * vertices.size() + offset[4];
        align_mem(offset[5], 24);
        offset[6] = sizeof(int3) * faces.size() + offset[5];
        align_mem(offset[6], 24);

        for (int i=0; i<6; i++)
        {
            printf("i %d    offset %ld    Mem %ld\n", i, offset[i], offset[i+1]-offset[i]);
        }
    }
};

struct DeviceData{

    uint8_t* buffer;
    uint64_t offset[7];

    __hostdev__ DeviceData(){}
    __hostdev__ ~DeviceData(){}

    void upload(HostData* host_data)
    {
        host_data->remap_mem(offset);
    
        cudaMalloc((void**)&buffer, offset[6]);

        cudaMemcpyAsync(buffer, host_data->buffer, offset[6], cudaMemcpyHostToDevice);

        cudaError_t err = cudaGetLastError();
        if ( err != cudaSuccess )
        {
            printf("CUDA Error: %s\n", cudaGetErrorString(err));       
        }else{
            printf("Upload to device successfully!\n");
        }
        
        printf("Total mem usage %.3f MB\n", 1.0f * offset[6] / (1u << 20));
    }

    __hostdev__ float3* getVertices()
    {
        return (float3*)(buffer+offset[4]);
    }

    __hostdev__ int3* getFaces()
    {
        return (int3*)(buffer+offset[5]);
    }

    __hostdev__ uint64_t* getBitmask()
    {
        return (uint64_t*)(buffer+offset[2]);
    }

    __hostdev__ uint32_t* getTileFaceIdx()
    {
        return (uint32_t*)(buffer+offset[1]);
    }

    __hostdev__ uint2* getStartNum()
    {
        return (uint2*)(buffer+offset[3]);
    }
};


#endif 