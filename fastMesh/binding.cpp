#include <pybind11/pybind11.h>
#include "fastMesh.h"

namespace py = pybind11;


PYBIND11_MODULE(fastMesh, m){

    m.doc()="Pytorch version of ray mesh intersection";

    py::class_<fastMesh>(m, "fastMesh")
        .def(py::init<>())
        .def("build", &fastMesh::build)
        .def("getSceneBound", &fastMesh::getSceneBound)
        .def("destroy", &fastMesh::destroy)
        .def("sample_points", &fastMesh::sample_points)
        .def("fisrtHit", &fastMesh::fisrtHit)
        .def("firstEnter", &fastMesh::firstEnter);

}