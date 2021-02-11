
#include "Declaration.hh"

#include "G4HepEmData.hh"
#include "G4HepEmElectronData.hh"

// don't worry it's just for testing
#define private public
#include "G4HepEmElectronManager.hh"

#include <cuda_runtime.h>
#include "G4HepEmCuUtils.hh"

// Pull in implementation
#include "G4HepEmElectronManager.icc"
#include "G4HepEmRunUtils.icc"

 __global__
 void TestResMacXSecDataKernel ( const struct G4HepEmElectronData* theElectronData_d,
                                 int* tsInImc_d, double* tsInEkin_d, double* tsInLogEkin_d,
                                 double* tsOutRes_d, bool isIoni, int numTestCases) {
   for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < numTestCases; i += blockDim.x * gridDim.x) {
     G4HepEmElectronManager theElectronMgr;
     tsOutRes_d[i] = theElectronMgr.GetRestMacXSec (theElectronData_d, tsInImc_d[i], tsInEkin_d[i], tsInLogEkin_d[i], isIoni);
   }
 }

void TestResMacXSecDataOnDevice ( const struct G4HepEmData* hepEmData, int* tsInImc_h,
     double* tsInEkinIoni_h, double* tsInLogEkinIoni_h, double* tsInEkinBrem_h, double* tsInLogEkinBrem_h,
     double* tsOutResMXIoni_h, double* tsOutResMXBrem_h, int numTestCases, bool iselectron ) {
  //
  // --- Allocate device side memory for the input/output data and copy all input
  //     data from host to device
  int*             tsInImc_d = nullptr;
  double*     tsInEkinIoni_d = nullptr;
  double*  tsInLogEkinIoni_d = nullptr;
  double*     tsInEkinBrem_d = nullptr;
  double*  tsInLogEkinBrem_d = nullptr;
  double*   tsOutResMXIoni_d = nullptr;
  double*   tsOutResMXBrem_d = nullptr;
  //
  gpuErrchk ( cudaMalloc ( &tsInImc_d,         sizeof( int )    * numTestCases ) );
  gpuErrchk ( cudaMalloc ( &tsInEkinIoni_d,    sizeof( double ) * numTestCases ) );
  gpuErrchk ( cudaMalloc ( &tsInLogEkinIoni_d, sizeof( double ) * numTestCases ) );
  gpuErrchk ( cudaMalloc ( &tsInEkinBrem_d,    sizeof( double ) * numTestCases ) );
  gpuErrchk ( cudaMalloc ( &tsInLogEkinBrem_d, sizeof( double ) * numTestCases ) );
  gpuErrchk ( cudaMalloc ( &tsOutResMXIoni_d,  sizeof( double ) * numTestCases ) );
  gpuErrchk ( cudaMalloc ( &tsOutResMXBrem_d,  sizeof( double ) * numTestCases ) );
  //
  // --- Copy the input data from host to device (test material-cut index, ekin and log-ekin arrays)
  gpuErrchk ( cudaMemcpy ( tsInImc_d,         tsInImc_h,         sizeof( int )    * numTestCases, cudaMemcpyHostToDevice) );
  gpuErrchk ( cudaMemcpy ( tsInEkinIoni_d,    tsInEkinIoni_h,    sizeof( double ) * numTestCases, cudaMemcpyHostToDevice) );
  gpuErrchk ( cudaMemcpy ( tsInLogEkinIoni_d, tsInLogEkinIoni_h, sizeof( double ) * numTestCases, cudaMemcpyHostToDevice) );
  gpuErrchk ( cudaMemcpy ( tsInEkinBrem_d,    tsInEkinBrem_h,    sizeof( double ) * numTestCases, cudaMemcpyHostToDevice) );
  gpuErrchk ( cudaMemcpy ( tsInLogEkinBrem_d, tsInLogEkinBrem_h, sizeof( double ) * numTestCases, cudaMemcpyHostToDevice) );
  //
  // --- Launch the kernels
  int numThreads = 512;
  int numBlocks  = std::ceil( float(numTestCases)/numThreads );
  //  std::cout << " N = " << numTestCases << " numBlocks = " << numBlocks << " numThreads = " << numThreads << " x = " << numBlocks*numThreads << std::endl;
  const G4HepEmElectronData* theElectronData_d = iselectron ? hepEmData->fTheElectronData_gpu : hepEmData->fThePositronData_gpu;
  // ioni
  TestResMacXSecDataKernel <<< numBlocks, numThreads >>> (theElectronData_d, tsInImc_d, tsInEkinIoni_d, tsInLogEkinIoni_d, tsOutResMXIoni_d, true,  numTestCases );
  // brem
  TestResMacXSecDataKernel <<< numBlocks, numThreads >>> (theElectronData_d, tsInImc_d, tsInEkinBrem_d, tsInLogEkinBrem_d, tsOutResMXBrem_d, false, numTestCases );
  //
  // --- Synchronize to make sure that completed on the device
  cudaDeviceSynchronize();
  //
  // --- Copy the results from the device to the host
  gpuErrchk ( cudaMemcpy ( tsOutResMXIoni_h,     tsOutResMXIoni_d,     sizeof( double ) * numTestCases, cudaMemcpyDeviceToHost ) );
  gpuErrchk ( cudaMemcpy ( tsOutResMXBrem_h,     tsOutResMXBrem_d,     sizeof( double ) * numTestCases, cudaMemcpyDeviceToHost ) );
  //
  // --- Free all dynamically allocated (device side) memory
  cudaFree ( tsInImc_d          );
  cudaFree ( tsInEkinIoni_d    );
  cudaFree ( tsInLogEkinIoni_d );
  cudaFree ( tsInEkinBrem_d    );
  cudaFree ( tsInLogEkinBrem_d );
  cudaFree ( tsOutResMXIoni_d  );
  cudaFree ( tsOutResMXBrem_d  );
}
