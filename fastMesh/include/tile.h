#ifndef _TILE_H__
#define _TILE_H__

#include <iostream>
#include <fstream>
#include "cutil_math.h"
#include "mask.h"

static constexpr uint32_t LOG2DIM = 6; 

class Tile
{
public:
    std::vector<uint32_t> faces;
    Tile(){}
    ~Tile(){}
};

class Root
{
public:
    static constexpr uint32_t SIDE = 1u << LOG2DIM;
    static constexpr uint32_t SIZE = 1u << (3 * LOG2DIM);
    Tile* child_list[SIZE];

    Root(){std::fill_n(child_list, SIZE, nullptr);}
    ~Root(){
        for (int i=0; i<SIZE; i++)
        {
            if (child_list[i] != nullptr) delete child_list[i];
        }
    }

    void addFace(uint3 ijk, uint32_t face_idx)
    {
        uint32_t n = (ijk.x << (2 * LOG2DIM)) | (ijk.y << LOG2DIM) | ijk.z;

        if (child_list[n] == nullptr)
        {
            child_list[n] = new Tile();
        }
        child_list[n]->faces.emplace_back(face_idx);
    }
};

// static inline void write_tiles(std::string out_path, 
//     float3 scene_corner, float tile_size,
//     Mask<Root::SIZE> bitmask, 
//     std::vector<uint32_t> data, std::vector<uint2> locate)
// {
//     // write out  (save scene_corner & tile size here )
//     uint64_t offset0 = 0;
//     uint64_t offset1 = bitmask.memUsage(); + offset0;
//     uint64_t offset2 = sizeof(uint32_t) * data.size() + offset1;
//     uint64_t offset3 = sizeof(uint2) * locate.size() + offset2;

//     printf("start writeing %d bytes to bin file\n", offset3);

//     std::ofstream outFile(out_path, std::ios::out | std::ios::binary);

//     outFile.write( (char*)&scene_corner.x, sizeof(float3) );
//     outFile.write( (char*)&tile_size, sizeof(float) );
//     outFile.write( (char*)&offset0, sizeof(uint64_t) );
//     outFile.write( (char*)&offset1, sizeof(uint64_t) );
//     outFile.write( (char*)&offset2, sizeof(uint64_t) );
//     outFile.write( (char*)&offset3, sizeof(uint64_t) );

//     outFile.write( (char*)&bitmask, offset1 - offset0 );
//     outFile.write( (char*)data.data(), offset2 - offset1 );
//     outFile.write( (char*)locate.data(), offset3 - offset2 );

//     outFile.close();
// }

// static inline void load_tiles(std::string file_path,
//     float3 &scene_corner, float &tile_size,
//     Mask<Root::SIZE> &bitmask, 
//     uint32_t* &data, uint2* &locate)
// {
//     std::ifstream inFile(file_path, std::ios::in | std::ios::binary);

//     if(!inFile) {
//         std::cout << "error" << std::endl;
//         return;
//     }
    
//     inFile.read( (char*)&scene_corner.x, sizeof(float3) );
//     inFile.read( (char*)&tile_size,  sizeof(float) );

//     uint64_t offset0, offset1, offset2, offset3;
//     inFile.read( (char*)&offset0, sizeof(uint64_t) );
//     inFile.read( (char*)&offset1, sizeof(uint64_t) );
//     inFile.read( (char*)&offset2, sizeof(uint64_t) );
//     inFile.read( (char*)&offset3, sizeof(uint64_t) );

//     data = new uint32_t[ (offset2 - offset1) / sizeof(uint32_t) ];
//     locate = new uint2[ (offset3 - offset2) / sizeof(uint2) ];
//     inFile.read( (char*)&bitmask, offset1 - offset0 );
//     inFile.read( (char*)data,  offset2 - offset1);
//     inFile.read( (char*)locate, offset3 - offset2 );
// }

#endif 