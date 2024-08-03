#include "sample.h"
#include "compute_ray.h"
#include "helper.h"
#include "view_selection.h"
#include "adam.h"
#include "grid_sample.h"
#include "build_blocks.h"
#include "voxelize.h"

PYBIND11_MODULE(CUDA_EXT, m){
    m.doc() = "pybind11 torch extension";
    m.def("sample_insideout_block", &sample_insideout_block, "");

    m.def("compute_ray_forward", &compute_ray_forward, "");
    m.def("compute_ray_backward", &compute_ray_backward, "");
    
    m.def("ray_aabb_intersection", &ray_aabb_intersection, "");
    m.def("ray_aabb_intersection_v2", &ray_aabb_intersection_v2, "");

    m.def("sample_points_contract", &sample_points_contract, "");
    m.def("sample_points_grid", &sample_points_grid, "");
    m.def("proj2pixel_and_fetch_color", &proj2pixel_and_fetch_color, "");

    m.def("computeViewcost", &computeViewcost, "");

    m.def("voxelize_mesh", &voxelize_mesh, "");

    m.def("background_sampling_cuda", &background_sampling_cuda, "");

    m.def("adam_step_cuda", &adam_step_cuda, "");

    m.def("adam_step_cuda_fp16", &adam_step_cuda_fp16, "");

    m.def("grid_sample_forward_cuda", &grid_sample_forward_cuda, "");

    m.def("grid_sample_backward_cuda", &grid_sample_backward_cuda, "");

    m.def("gaussian_grid_sample_forward_cuda", &gaussian_grid_sample_forward_cuda, "");

    m.def("gaussian_grid_sample_backward_cuda", &gaussian_grid_sample_backward_cuda, "");

    m.def("grid_sample_bool_cuda", &grid_sample_bool_cuda, "");


    m.def("proj2neighbor_forward", &proj2neighbor_forward, "");
    m.def("proj2neighbor_backward", &proj2neighbor_backward, "");

    py::class_<BlockBuilder>(m, "BlockBuilder")
        .def(py::init<>())
        .def("load_mesh", &BlockBuilder::load_mesh)
        .def("load_camera", &BlockBuilder::load_camera)
        .def("init_blocks", &BlockBuilder::init_blocks)
        .def("view_selection_for_single_block", &BlockBuilder::view_selection_for_single_block);
}