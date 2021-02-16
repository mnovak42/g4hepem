
#include "Declaration.hh"

#include "G4HepEmData.hh"
#include "G4HepEmElectronData.hh"

#include <cuda_runtime.h>
#include "G4HepEmCuUtils.hh"

// Pull in implementation of Brem element selector
#include "G4HepEmElectronInteractionBrem.icc"

template <bool TisSBModel>
__global__
void TestElemSelectorDataBremKernel ( const struct G4HepEmElectronData* theElectronData_d,
     const struct G4HepEmMatCutData* theMatCutData_d, const struct G4HepEmMaterialData* theMaterialData_d,
     int* tsInImc_d, double* tsInEkin_d, double* tsInLogEkin_d, double* tsInRngVals_d,
     int* tsOutRes_d, int numTestCases ) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < numTestCases; i += blockDim.x * gridDim.x) {
    // get number of elements this material (from the currecnt material-cuts)
    // is composed of
    const int imc = tsInImc_d[i];
    const int indxMaterial = theMatCutData_d->fMatCutData[imc].fHepEmMatIndex;
    const struct G4HepEmMatData& theMatData = theMaterialData_d->fMaterialData[indxMaterial];
    const int numOfElement = theMatData.fNumOfElement;
    int targetElemIndx = 0;
    if (numOfElement > 1) {
      targetElemIndx = SelectTargetAtomBrem( theElectronData_d, imc, tsInEkin_d[i], tsInLogEkin_d[i], tsInRngVals_d[i], TisSBModel);
    }
    tsOutRes_d[i] = targetElemIndx;
  }
}

void TestElemSelectorDataOnDevice ( const struct G4HepEmData* hepEmData, int* tsInImc_h,
     double* tsInEkin_h, double* tsInLogEkin_h, double* tsInRngVals_h,
     int* tsOutRes_h, int numTestCases, int indxModel, bool iselectron ) {
  //
  // --- Allocate device side memory for the input/output data and copy all input
  //     data from host to device
  int*        tsInImc_d = nullptr;
  double*    tsInEkin_d = nullptr;
  double* tsInLogEkin_d = nullptr;
  double* tsInRngVals_d = nullptr;
  int*       tsOutRes_d = nullptr;
  //
  gpuErrchk ( cudaMalloc ( &tsInImc_d,     sizeof( int )    * numTestCases ) );
  gpuErrchk ( cudaMalloc ( &tsInEkin_d,    sizeof( double ) * numTestCases ) );
  gpuErrchk ( cudaMalloc ( &tsInLogEkin_d, sizeof( double ) * numTestCases ) );
  gpuErrchk ( cudaMalloc ( &tsInRngVals_d, sizeof( double ) * numTestCases ) );
  gpuErrchk ( cudaMalloc ( &tsOutRes_d,    sizeof( int ) * numTestCases ) );
  //
  // --- Copy the input data from host to device (test material-cut index, ekin and log-ekin arrays)
  gpuErrchk ( cudaMemcpy ( tsInImc_d,     tsInImc_h,     sizeof( int )    * numTestCases, cudaMemcpyHostToDevice) );
  gpuErrchk ( cudaMemcpy ( tsInEkin_d,    tsInEkin_h,    sizeof( double ) * numTestCases, cudaMemcpyHostToDevice) );
  gpuErrchk ( cudaMemcpy ( tsInLogEkin_d, tsInLogEkin_h, sizeof( double ) * numTestCases, cudaMemcpyHostToDevice) );
  gpuErrchk ( cudaMemcpy ( tsInRngVals_d, tsInRngVals_h, sizeof( double ) * numTestCases, cudaMemcpyHostToDevice) );
  //
  // --- Launch the kernels
  const struct G4HepEmElectronData* theElectronData_d = iselectron ? hepEmData->fTheElectronData_gpu : hepEmData->fThePositronData_gpu;
  const struct G4HepEmMatCutData*   theMatCutData_d   = hepEmData->fTheMatCutData_gpu;
  const struct G4HepEmMaterialData* theMaterialData_d = hepEmData->fTheMaterialData_gpu;
  int numThreads = 512;
  int numBlocks  = std::ceil( float(numTestCases)/numThreads );
  switch (indxModel) {
    case 0: // not used
      break;

    case 1:

      TestElemSelectorDataBremKernel < true >  <<< numBlocks, numThreads >>> ( theElectronData_d, theMatCutData_d, theMaterialData_d, tsInImc_d, tsInEkin_d, tsInLogEkin_d, tsInRngVals_d, tsOutRes_d, numTestCases );
      break;

    case 2:
      TestElemSelectorDataBremKernel < false > <<< numBlocks, numThreads >>> ( theElectronData_d, theMatCutData_d, theMaterialData_d, tsInImc_d, tsInEkin_d, tsInLogEkin_d, tsInRngVals_d, tsOutRes_d, numTestCases );
      break;
  }
  //
  // --- Synchronize to make sure that completed on the device
  cudaDeviceSynchronize();
  //
  // --- Copy the results from the device to the host
  gpuErrchk ( cudaMemcpy ( tsOutRes_h, tsOutRes_d, sizeof( int ) * numTestCases, cudaMemcpyDeviceToHost ) );
  //
  // --- Free all dynamically allocated (device side) memory
  cudaFree ( tsInImc_d     );
  cudaFree ( tsInEkin_d    );
  cudaFree ( tsInLogEkin_d );
  cudaFree ( tsInRngVals_d );
  cudaFree ( tsOutRes_d    );
}
