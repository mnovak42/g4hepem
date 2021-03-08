
#include "Declaration.hh"

#include "G4HepEmData.hh"
#include "G4HepEmGammaData.hh"
#include "G4HepEmMaterialData.hh"

#include <cuda_runtime.h>
#include "G4HepEmCuUtils.hh"

// Pull in implementation of element selector fo conversion
#include "G4HepEmGammaInteractionConversion.icc"

__global__
void GammaElemSelectorKernel ( const struct G4HepEmGammaData* theGammaData_d,
     const struct G4HepEmMaterialData* theMaterialData_d, int* tsInImat_d,
     double* tsInEkin_d, double* tsInLogEkin_d, double* tsInRngVals_d,
     int* tsOutRes_d, int numTestCases ) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < numTestCases; i += blockDim.x * gridDim.x) {
    // get number of elements this material is composed of
    const int indxMaterial = tsInImat_d[i];
    const struct G4HepEmMatData& theMatData = theMaterialData_d->fMaterialData[indxMaterial];
    const int numOfElement = theMatData.fNumOfElement;
    int targetElemIndx = 0;
    if (numOfElement > 1) {
      targetElemIndx = SelectTargetAtom( theGammaData_d, indxMaterial, tsInEkin_d[i], tsInLogEkin_d[i], tsInRngVals_d[i]);
    }
    tsOutRes_d[i] = targetElemIndx;
  }
}

void TestGammaElemSelectorDataOnDevice ( const struct G4HepEmData* hepEmData, int* tsInImat_h,
     double* tsInEkin_h, double* tsInLogEkin_h, double* tsInRngVals_h, int* tsOutRes_h,
     int numTestCases ) {
  //
  // --- Allocate device side memory for the input/output data and copy all input
  //     data from host to device
  int*        tsInImat_d = nullptr;
  double*     tsInEkin_d = nullptr;
  double*  tsInLogEkin_d = nullptr;
  double*  tsInRngVals_d = nullptr;
  int*        tsOutRes_d = nullptr;
  //
  gpuErrchk ( cudaMalloc ( &tsInImat_d,    sizeof( int )    * numTestCases ) );
  gpuErrchk ( cudaMalloc ( &tsInEkin_d,    sizeof( double ) * numTestCases ) );
  gpuErrchk ( cudaMalloc ( &tsInLogEkin_d, sizeof( double ) * numTestCases ) );
  gpuErrchk ( cudaMalloc ( &tsInRngVals_d, sizeof( double ) * numTestCases ) );
  gpuErrchk ( cudaMalloc ( &tsOutRes_d,    sizeof( int ) * numTestCases ) );
  //
  // --- Copy the input data from host to device (test material-cut index, ekin and log-ekin arrays)
  gpuErrchk ( cudaMemcpy ( tsInImat_d,    tsInImat_h,    sizeof( int )    * numTestCases, cudaMemcpyHostToDevice) );
  gpuErrchk ( cudaMemcpy ( tsInEkin_d,    tsInEkin_h,    sizeof( double ) * numTestCases, cudaMemcpyHostToDevice) );
  gpuErrchk ( cudaMemcpy ( tsInLogEkin_d, tsInLogEkin_h, sizeof( double ) * numTestCases, cudaMemcpyHostToDevice) );
  gpuErrchk ( cudaMemcpy ( tsInRngVals_d, tsInRngVals_h, sizeof( double ) * numTestCases, cudaMemcpyHostToDevice) );
  //
  // --- Launch the kernels
  const struct G4HepEmGammaData*       theGammaData_d = hepEmData->fTheGammaData_gpu;
  const struct G4HepEmMaterialData* theMaterialData_d = hepEmData->fTheMaterialData_gpu;
  int numThreads = 512;
  int numBlocks  = std::ceil( float(numTestCases)/numThreads );
  GammaElemSelectorKernel <<< numBlocks, numThreads >>> ( theGammaData_d, theMaterialData_d, tsInImat_d, tsInEkin_d, tsInLogEkin_d, tsInRngVals_d, tsOutRes_d, numTestCases );
  //
  // --- Synchronize to make sure that completed on the device
  cudaDeviceSynchronize();
  //
  // --- Copy the results from the device to the host
  gpuErrchk ( cudaMemcpy ( tsOutRes_h, tsOutRes_d, sizeof( int ) * numTestCases, cudaMemcpyDeviceToHost ) );
  //
  // --- Free all dynamically allocated (device side) memory
  cudaFree ( tsInImat_d    );
  cudaFree ( tsInEkin_d    );
  cudaFree ( tsInLogEkin_d );
  cudaFree ( tsInRngVals_d );
  cudaFree ( tsOutRes_d    );
}
