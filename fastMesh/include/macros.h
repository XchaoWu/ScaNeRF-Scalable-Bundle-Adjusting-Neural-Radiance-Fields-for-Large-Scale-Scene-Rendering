#pragma once
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/ATen.h>
#include <stdint.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <ATen/TensorAccessor.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAUtils.h>
#include <torch/extension.h>
#include <iostream>
#include <fstream>
#include <string.h>
#include <assert.h>

#define __hostdev__ __host__ __device__

#define CHECK_INPUT(x)                                                         \
  TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor");                   \
  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor");

#define MIN_VALUE(a,b) a < b ? a : b
#define NUM_THREAD 512
#define NUM_BLOCK(x) MIN_VALUE(65535, (x + NUM_THREAD - 1) / NUM_THREAD)