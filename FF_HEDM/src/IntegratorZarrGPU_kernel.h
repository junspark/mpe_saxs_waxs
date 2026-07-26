/* IntegratorZarrGPU_kernel.h
 *
 * C-callable API for the CUDA integration kernels used by IntegratorZarrGPU.c.
 * The host code (IntegratorZarrGPU.c) is a straight copy of IntegratorZarrOMP.c
 * with the per-frame `#pragma omp parallel for` compute region replaced by
 * calls to this API. Kernel implementations mirror those in
 * IntegratorFitPeaksGPUStream.cu (integrate_noMapMask, PrecomputeOffsets_kernel,
 * initialize_PerFrameArr_Area_kernel).
 */

#ifndef INTEGRATOR_ZARR_GPU_KERNEL_H
#define INTEGRATOR_ZARR_GPU_KERNEL_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct GpuIntegratorCtx GpuIntegratorCtx;

/* One-time GPU setup. pxList_bytes points at the Map.bin AoS data
 * (struct data[]): the .cu translation unit has an identical struct
 * definition so passing raw bytes here is safe. Runs the one-time
 * PrecomputeOffsets and initialize_PerFrameArr_Area kernels and returns
 * an opaque context. Returns NULL on failure. */
GpuIntegratorCtx *gpu_integrator_init(
    const void *pxList_bytes, size_t pxList_bytes_len,
    const int *nPxList, size_t bigArrSize,
    int NrPixelsY, int NrPixelsZ,
    int nRBins, int nEtaBins,
    const double *RBinsLow, const double *RBinsHigh,
    const double *EtaBinsLow, const double *EtaBinsHigh,
    double px, double Lsd, double Lam,
    int doBinSort, int gradientCorrection);

/* Copy host PerFrameArr[bigArrSize*5] from device. Call once, right after
 * init, if you need the R/2theta/eta/area/Q meta arrays (OMP writes them
 * to /REtaMap for frame 0). */
int gpu_integrator_copy_perframe_arr(GpuIntegratorCtx *ctx,
                                     double *perFrameArrHost);

/* Per-frame integration. Uploads image (dark-subtracted host doubles,
 * NrPixelsY*NrPixelsZ), launches integrate_noMapMask, copies intensity
 * results back. Returns 0 on success. */
int gpu_integrator_process_frame(GpuIntegratorCtx *ctx,
                                 const double *imageHost,
                                 double *intArrHost,
                                 int Normalize, int sumImages, int frameIdx,
                                 float BC_y, float BC_z);

/* Copy back accumulated sum matrix if sumImages mode was used. */
int gpu_integrator_copy_sum_matrix(GpuIntegratorCtx *ctx, double *sumMatrixHost);

/* Teardown. Safe to call with NULL. */
void gpu_integrator_teardown(GpuIntegratorCtx *ctx);

#ifdef __cplusplus
}
#endif

#endif /* INTEGRATOR_ZARR_GPU_KERNEL_H */
