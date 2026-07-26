// IntegratorZarrGPU_kernel.cu
//
// CUDA kernels and C-callable API for the GPU port of IntegratorZarrOMP.
// The host code (IntegratorZarrGPU.c) is a straight copy of the OMP integrator,
// modified only where the per-frame OMP compute loop lives (OMP.c lines
// 1709-1779), which is replaced by a call to gpu_integrate_frame() defined
// here.
//
// Milestone 1: stub only — kernels are added in milestone 2/3.

#include <cuda_runtime.h>

extern "C" void integrator_zarr_gpu_placeholder(void) {
    // Present so nvcc has something to compile and the linker pulls in the .cu
    // translation unit. Will be replaced by real init/integrate/teardown API
    // in milestone 2.
}
