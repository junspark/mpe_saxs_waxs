// IntegratorZarrGPU_kernel.cu
//
// CUDA kernels + C-callable API implementation for the GPU port of
// IntegratorZarrOMP. Kernels mirror those in IntegratorFitPeaksGPUStream.cu
// (integrate_noMapMask, PrecomputeOffsets_kernel, initialize_PerFrameArr_Area_kernel).
//
// Host code in IntegratorZarrGPU.c calls the extern-C API declared in
// IntegratorZarrGPU_kernel.h. All CUDA state lives inside GpuIntegratorCtx.

#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "IntegratorZarrGPU_kernel.h"

// ============================================================================
// Constants and helpers
// ============================================================================

#define THREADS_PER_BLOCK 512
#define AREA_THRESHOLD    1e-9
#define RAD2DEG           (180.0 / M_PI)

#define GPUERR_RET(call, retval)                                               \
  do {                                                                         \
    cudaError_t _e = (call);                                                   \
    if (_e != cudaSuccess) {                                                   \
      fprintf(stderr, "[IntegratorZarrGPU] CUDA error at %s:%d: %s\n",         \
              __FILE__, __LINE__, cudaGetErrorString(_e));                     \
      return retval;                                                           \
    }                                                                          \
  } while (0)

#define GPUERR_NULL(call) GPUERR_RET(call, NULL)
#define GPUERR_ONE(call)  GPUERR_RET(call, 1)

// Must match struct data in IntegratorZarrOMP.c and IntegratorFitPeaksGPUStream.cu.
struct data {
  float y, z;
  double frac;
  float deltaR, areaWeight;
};
static_assert(sizeof(struct data) == 24,
              "struct data size must match OMP integrator layout (24 bytes)");

// ============================================================================
// Context
// ============================================================================

struct GpuIntegratorCtx {
  // Geometry / config
  size_t bigArrSize;
  int NrPixelsY, NrPixelsZ;
  size_t totalPixels;
  int gradientCorrection;
  double px, Lsd;

  // Persistent device buffers
  struct data *dPxList;
  int *dNPxList;
  int *dSortedIndices;
  int *dPixelOffsets;
  float *dPixelWeights;
  float *dPixelAreaWeights;
  float *dDeltaR;
  float *dPxY, *dPxZ;
  double *dPerFrame;
  double *dEtaLo, *dEtaHi, *dRLo, *dRHi;

  // Per-frame reusable device buffers
  float *dImage;
  double *dIntArrPerFrame;

  // Host staging (image needs double -> float downcast before H2D)
  float *hImageStaging;
};

typedef struct {
  int index;
  int count;
} BinSort;

static int compareBinSort(const void *a, const void *b) {
  const BinSort *bA = (const BinSort *)a;
  const BinSort *bB = (const BinSort *)b;
  return bB->count - bA->count; // descending
}

// ============================================================================
// Kernels (verbatim from IntegratorFitPeaksGPUStream.cu)
// ============================================================================

__global__ void
initialize_PerFrameArr_Area_kernel(double *dPerFrameArr, size_t bigArrSize,
                                   int nRBins, int nEtaBins,
                                   const double *dRBinsLow,
                                   const double *dRBinsHigh,
                                   const double *dEtaBinsLow,
                                   const double *dEtaBinsHigh,
                                   const struct data *dPxList,
                                   const int *dNPxList, int NrPixelsY,
                                   int NrPixelsZ, double px, double Lsd,
                                   double Lam) {
  const size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= bigArrSize)
    return;

  double RMean = 0.0;
  double EtaMean = 0.0;
  double TwoTheta = 0.0;

  if (nEtaBins > 0) {
    int j = idx / nEtaBins; // R bin index
    int k = idx % nEtaBins; // Eta bin index
    if (j < nRBins && k < nEtaBins) {
      RMean = (dRBinsLow[j] + dRBinsHigh[j]) * 0.5;
      EtaMean = (dEtaBinsLow[k] + dEtaBinsHigh[k]) * 0.5;
      TwoTheta = RAD2DEG * atan(RMean * px / Lsd);
    }
  }

  double totArea = 0.0;
  long long nPixels = 0;
  long long dataPos = 0;
  const size_t nPxListIndex = 2 * idx;
  const size_t totalPixels = (size_t)NrPixelsY * NrPixelsZ;

  if (nPxListIndex + 1 < 2 * bigArrSize) {
    nPixels = dNPxList[nPxListIndex];
    dataPos = dNPxList[nPxListIndex + 1];
  }

  if (nPixels > 0 && dataPos >= 0) {
    for (long long l = 0; l < nPixels; l++) {
      struct data ThisVal = dPxList[dataPos + l];
      if (ThisVal.y < 0 || ThisVal.y >= NrPixelsY || ThisVal.z < 0 ||
          ThisVal.z >= NrPixelsZ)
        continue;
      long long testPos = (long long)ThisVal.z * NrPixelsY + ThisVal.y;
      if (testPos < 0 || testPos >= (long long)totalPixels)
        continue;
      totArea += ThisVal.areaWeight;
    }
  }

  if (idx < bigArrSize) {
    dPerFrameArr[0 * bigArrSize + idx] = RMean;
    dPerFrameArr[1 * bigArrSize + idx] = TwoTheta;
    dPerFrameArr[2 * bigArrSize + idx] = EtaMean;
    dPerFrameArr[3 * bigArrSize + idx] = totArea;
    double twoTheta_rad = atan(RMean * px / Lsd);
    dPerFrameArr[4 * bigArrSize + idx] =
        (Lam > 0) ? (4.0 * M_PI / Lam) * sin(twoTheta_rad / 2.0) : 0.0;
  }
}

__global__ void PrecomputeOffsets_kernel(const struct data *dPxList,
                                         const int *dNPxList, int nBins,
                                         int NrPixelsY, int *dOffsets,
                                         float *dWeights, float *dAreaWeights,
                                         float *dDeltaR, float *dPxY,
                                         float *dPxZ) {
  const size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= (size_t)nBins)
    return;

  long long nPixels = dNPxList[2 * idx];
  long long dataPos = dNPxList[2 * idx + 1];

  for (long long l = 0; l < nPixels; l++) {
    struct data ThisVal = dPxList[dataPos + l];
    long long offset = (long long)ThisVal.z * NrPixelsY + ThisVal.y;
    dOffsets[dataPos + l] = (int)offset;
    dWeights[dataPos + l] = ThisVal.frac;
    dAreaWeights[dataPos + l] = ThisVal.areaWeight;
    dDeltaR[dataPos + l] = ThisVal.deltaR;
    dPxY[dataPos + l] = ThisVal.y;
    dPxZ[dataPos + l] = ThisVal.z;
  }
}

__global__ void
integrate_noMapMask(double px, double Lsd, size_t bigArrSize, int Normalize,
                    int sumImages, int frameIdx, const int *dNPxList,
                    int NrPixelsY, int NrPixelsZ, const float *dImage,
                    double *dIntArrPerFrame, double *dSumMatrix,
                    const int *dOffsets, const float *dWeights,
                    const float *dAreaWeights, const int *dSortedIndices,
                    int gradientCorrection, const float *dDeltaR,
                    const float *dPxY, const float *dPxZ, float BC_y,
                    float BC_z) {
  const size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= bigArrSize)
    return;

  const int idx = dSortedIndices[tid];

  float Intensity = 0.0f;
  float totArea = 0.0f;

  long long nPixels = 0;
  long long dataPos = 0;
  const size_t nPxListIndex = 2 * idx;

  nPixels = dNPxList[nPxListIndex];
  dataPos = dNPxList[nPxListIndex + 1];

  for (long long l = 0; l < nPixels; l++) {
    float weight = dWeights[dataPos + l];
    float pixVal;
    if (gradientCorrection) {
      float py = dPxY[dataPos + l];
      float pz = dPxZ[dataPos + l];
      float deltaR = dDeltaR[dataPos + l];
      float dy = py - BC_y;
      float dz = pz - BC_z;
      float R = sqrtf(dy * dy + dz * dz);
      float ry = py, rz = pz;
      if (R > 1.0f) {
        ry -= deltaR * dy / R;
        rz -= deltaR * dz / R;
      }
      int iy = (int)floorf(ry), iz = (int)floorf(rz);
      float fy = ry - iy, fz = rz - iz;
      if (iy < 0) { iy = 0; fy = 0; }
      if (iy >= NrPixelsY - 1) { iy = NrPixelsY - 2; fy = 1; }
      if (iz < 0) { iz = 0; fz = 0; }
      if (iz >= NrPixelsZ - 1) { iz = NrPixelsZ - 2; fz = 1; }
      pixVal =
          __ldg(&dImage[(size_t)iz * NrPixelsY + iy]) * (1 - fy) * (1 - fz) +
          __ldg(&dImage[(size_t)iz * NrPixelsY + iy + 1]) * fy * (1 - fz) +
          __ldg(&dImage[(size_t)(iz + 1) * NrPixelsY + iy]) * (1 - fy) * fz +
          __ldg(&dImage[(size_t)(iz + 1) * NrPixelsY + iy + 1]) * fy * fz;
    } else {
      int testPos = dOffsets[dataPos + l];
      pixVal = __ldg(&dImage[testPos]);
    }
    Intensity += pixVal * weight;
    totArea += dAreaWeights[dataPos + l];
  }

  if (totArea > AREA_THRESHOLD) {
    if (Normalize)
      Intensity /= totArea;
    dIntArrPerFrame[idx] = Intensity;
    if (sumImages && dSumMatrix)
      atomicAdd(&dSumMatrix[idx], (double)Intensity);
  } else {
    dIntArrPerFrame[idx] = 0.0;
  }
}

// ============================================================================
// C-callable API
// ============================================================================

extern "C" GpuIntegratorCtx *gpu_integrator_init(
    const void *pxList_bytes, size_t pxList_bytes_len,
    const int *nPxList, size_t bigArrSize,
    int NrPixelsY, int NrPixelsZ,
    int nRBins, int nEtaBins,
    const double *RBinsLow, const double *RBinsHigh,
    const double *EtaBinsLow, const double *EtaBinsHigh,
    double px, double Lsd, double Lam,
    int doBinSort, int gradientCorrection) {
  if (!pxList_bytes || !nPxList || !RBinsLow || !RBinsHigh || !EtaBinsLow ||
      !EtaBinsHigh) {
    fprintf(stderr, "[IntegratorZarrGPU] init: NULL argument\n");
    return NULL;
  }

  GpuIntegratorCtx *ctx =
      (GpuIntegratorCtx *)calloc(1, sizeof(GpuIntegratorCtx));
  if (!ctx) {
    fprintf(stderr, "[IntegratorZarrGPU] init: calloc failed\n");
    return NULL;
  }
  ctx->bigArrSize = bigArrSize;
  ctx->NrPixelsY = NrPixelsY;
  ctx->NrPixelsZ = NrPixelsZ;
  ctx->totalPixels = (size_t)NrPixelsY * NrPixelsZ;
  ctx->gradientCorrection = gradientCorrection;
  ctx->px = px;
  ctx->Lsd = Lsd;

  // Report device
  int devId = 0;
  cudaDeviceProp prop;
  if (cudaGetDeviceProperties(&prop, devId) == cudaSuccess) {
    printf("[IntegratorZarrGPU] GPU Device 0: %s (CC %d.%d)\n", prop.name,
           prop.major, prop.minor);
  }

  const size_t szPxList = pxList_bytes_len;
  const size_t szNPxList = 2 * bigArrSize * sizeof(int);
  const size_t pxListCount = szPxList / sizeof(struct data);

  // Persistent device buffers
  GPUERR_NULL(cudaMalloc(&ctx->dPxList, szPxList));
  GPUERR_NULL(cudaMalloc(&ctx->dNPxList, szNPxList));
  GPUERR_NULL(cudaMalloc(&ctx->dPerFrame, bigArrSize * 5 * sizeof(double)));
  GPUERR_NULL(cudaMalloc(&ctx->dEtaLo, nEtaBins * sizeof(double)));
  GPUERR_NULL(cudaMalloc(&ctx->dEtaHi, nEtaBins * sizeof(double)));
  GPUERR_NULL(cudaMalloc(&ctx->dRLo, nRBins * sizeof(double)));
  GPUERR_NULL(cudaMalloc(&ctx->dRHi, nRBins * sizeof(double)));
  GPUERR_NULL(cudaMalloc(&ctx->dPixelOffsets, pxListCount * sizeof(int)));
  GPUERR_NULL(cudaMalloc(&ctx->dPixelWeights, pxListCount * sizeof(float)));
  GPUERR_NULL(cudaMalloc(&ctx->dPixelAreaWeights, pxListCount * sizeof(float)));
  GPUERR_NULL(cudaMalloc(&ctx->dDeltaR, pxListCount * sizeof(float)));
  GPUERR_NULL(cudaMalloc(&ctx->dPxY, pxListCount * sizeof(float)));
  GPUERR_NULL(cudaMalloc(&ctx->dPxZ, pxListCount * sizeof(float)));
  GPUERR_NULL(cudaMalloc(&ctx->dSortedIndices, bigArrSize * sizeof(int)));
  GPUERR_NULL(cudaMalloc(&ctx->dImage, ctx->totalPixels * sizeof(float)));
  GPUERR_NULL(cudaMalloc(&ctx->dIntArrPerFrame, bigArrSize * sizeof(double)));

  // Upload map and bin edges
  GPUERR_NULL(
      cudaMemcpy(ctx->dPxList, pxList_bytes, szPxList, cudaMemcpyHostToDevice));
  GPUERR_NULL(
      cudaMemcpy(ctx->dNPxList, nPxList, szNPxList, cudaMemcpyHostToDevice));
  GPUERR_NULL(cudaMemcpy(ctx->dEtaLo, EtaBinsLow, nEtaBins * sizeof(double),
                         cudaMemcpyHostToDevice));
  GPUERR_NULL(cudaMemcpy(ctx->dEtaHi, EtaBinsHigh, nEtaBins * sizeof(double),
                         cudaMemcpyHostToDevice));
  GPUERR_NULL(cudaMemcpy(ctx->dRLo, RBinsLow, nRBins * sizeof(double),
                         cudaMemcpyHostToDevice));
  GPUERR_NULL(cudaMemcpy(ctx->dRHi, RBinsHigh, nRBins * sizeof(double),
                         cudaMemcpyHostToDevice));

  // Build sorted indices on host, upload
  int *hSortedIndices = (int *)malloc(bigArrSize * sizeof(int));
  if (!hSortedIndices) {
    fprintf(stderr, "[IntegratorZarrGPU] init: malloc hSortedIndices failed\n");
    gpu_integrator_teardown(ctx);
    return NULL;
  }
  if (doBinSort) {
    BinSort *hBinSort = (BinSort *)malloc(bigArrSize * sizeof(BinSort));
    if (!hBinSort) {
      fprintf(stderr, "[IntegratorZarrGPU] init: malloc hBinSort failed\n");
      free(hSortedIndices);
      gpu_integrator_teardown(ctx);
      return NULL;
    }
    for (size_t i = 0; i < bigArrSize; i++) {
      hBinSort[i].index = (int)i;
      hBinSort[i].count = nPxList[2 * i];
    }
    qsort(hBinSort, bigArrSize, sizeof(BinSort), compareBinSort);
    for (size_t i = 0; i < bigArrSize; i++)
      hSortedIndices[i] = hBinSort[i].index;
    free(hBinSort);
  } else {
    for (size_t i = 0; i < bigArrSize; i++)
      hSortedIndices[i] = (int)i;
  }
  cudaError_t e = cudaMemcpy(ctx->dSortedIndices, hSortedIndices,
                             bigArrSize * sizeof(int), cudaMemcpyHostToDevice);
  free(hSortedIndices);
  if (e != cudaSuccess) {
    fprintf(stderr, "[IntegratorZarrGPU] init: dSortedIndices copy failed: %s\n",
            cudaGetErrorString(e));
    gpu_integrator_teardown(ctx);
    return NULL;
  }

  // Host staging for image downcast (double -> float)
  ctx->hImageStaging = (float *)malloc(ctx->totalPixels * sizeof(float));
  if (!ctx->hImageStaging) {
    fprintf(stderr, "[IntegratorZarrGPU] init: hImageStaging malloc failed\n");
    gpu_integrator_teardown(ctx);
    return NULL;
  }

  // Fire one-time kernels: PrecomputeOffsets + initialize_PerFrameArr_Area
  const int TPB = 256;
  const int blocks = (int)((bigArrSize + TPB - 1) / TPB);
  PrecomputeOffsets_kernel<<<blocks, TPB>>>(
      ctx->dPxList, ctx->dNPxList, (int)bigArrSize, NrPixelsY,
      ctx->dPixelOffsets, ctx->dPixelWeights, ctx->dPixelAreaWeights,
      ctx->dDeltaR, ctx->dPxY, ctx->dPxZ);
  initialize_PerFrameArr_Area_kernel<<<blocks, TPB>>>(
      ctx->dPerFrame, bigArrSize, nRBins, nEtaBins, ctx->dRLo, ctx->dRHi,
      ctx->dEtaLo, ctx->dEtaHi, ctx->dPxList, ctx->dNPxList, NrPixelsY,
      NrPixelsZ, px, Lsd, Lam);
  GPUERR_NULL(cudaDeviceSynchronize());

  printf("[IntegratorZarrGPU] init: %zu bins, %dx%d px, gradientCorrection=%d, "
         "doBinSort=%d\n",
         bigArrSize, NrPixelsY, NrPixelsZ, gradientCorrection, doBinSort);
  return ctx;
}

extern "C" int gpu_integrator_copy_perframe_arr(GpuIntegratorCtx *ctx,
                                                double *perFrameArrHost) {
  if (!ctx || !perFrameArrHost)
    return 1;
  GPUERR_ONE(cudaMemcpy(perFrameArrHost, ctx->dPerFrame,
                        ctx->bigArrSize * 5 * sizeof(double),
                        cudaMemcpyDeviceToHost));
  return 0;
}

extern "C" int gpu_integrator_process_frame(GpuIntegratorCtx *ctx,
                                            const double *imageHost,
                                            double *intArrHost, int Normalize,
                                            int sumImages, int frameIdx,
                                            float BC_y, float BC_z) {
  if (!ctx || !imageHost || !intArrHost)
    return 1;

  // Downcast double -> float on host, then H2D.
  for (size_t p = 0; p < ctx->totalPixels; p++) {
    ctx->hImageStaging[p] = (float)imageHost[p];
  }
  GPUERR_ONE(cudaMemcpy(ctx->dImage, ctx->hImageStaging,
                        ctx->totalPixels * sizeof(float),
                        cudaMemcpyHostToDevice));

  // Launch integrate kernel: one thread per bin, sorted by workload.
  const int TPB = THREADS_PER_BLOCK;
  const int blocks = (int)((ctx->bigArrSize + TPB - 1) / TPB);
  integrate_noMapMask<<<blocks, TPB>>>(
      ctx->px, ctx->Lsd, ctx->bigArrSize, Normalize, sumImages, frameIdx,
      ctx->dNPxList, ctx->NrPixelsY, ctx->NrPixelsZ, ctx->dImage,
      ctx->dIntArrPerFrame, /*dSumMatrix=*/NULL, ctx->dPixelOffsets,
      ctx->dPixelWeights, ctx->dPixelAreaWeights, ctx->dSortedIndices,
      ctx->gradientCorrection, ctx->dDeltaR, ctx->dPxY, ctx->dPxZ, BC_y, BC_z);

  // D2H the result. cudaMemcpy is synchronous wrt the default stream, so no
  // separate sync needed.
  GPUERR_ONE(cudaMemcpy(intArrHost, ctx->dIntArrPerFrame,
                        ctx->bigArrSize * sizeof(double),
                        cudaMemcpyDeviceToHost));

  cudaError_t last = cudaGetLastError();
  if (last != cudaSuccess) {
    fprintf(stderr, "[IntegratorZarrGPU] kernel launch error: %s\n",
            cudaGetErrorString(last));
    return 1;
  }
  return 0;
}

extern "C" int gpu_integrator_copy_sum_matrix(GpuIntegratorCtx *ctx,
                                              double *sumMatrixHost) {
  (void)ctx;
  (void)sumMatrixHost;
  // sumImages/dSumMatrix path not supported in this initial GPU port —
  // callers should keep sumImages=0 (which is the default). Follow-up work.
  fprintf(stderr,
          "[IntegratorZarrGPU] sumMatrix copy not supported in this build\n");
  return 1;
}

extern "C" void gpu_integrator_teardown(GpuIntegratorCtx *ctx) {
  if (!ctx)
    return;
#define FREE(p)                                                                \
  do {                                                                         \
    if (ctx->p)                                                                \
      cudaFree(ctx->p);                                                        \
  } while (0)
  FREE(dPxList);
  FREE(dNPxList);
  FREE(dPerFrame);
  FREE(dEtaLo);
  FREE(dEtaHi);
  FREE(dRLo);
  FREE(dRHi);
  FREE(dPixelOffsets);
  FREE(dPixelWeights);
  FREE(dPixelAreaWeights);
  FREE(dDeltaR);
  FREE(dPxY);
  FREE(dPxZ);
  FREE(dSortedIndices);
  FREE(dImage);
  FREE(dIntArrPerFrame);
#undef FREE
  if (ctx->hImageStaging)
    free(ctx->hImageStaging);
  free(ctx);
}
