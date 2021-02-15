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
void TestElossDataKernel  ( struct G4HepEmElectronData* theElectronData_d, int* tsInImc_d,
                            double* tsInEkin_d, double* tsInLogEkin_d, double* tsOutResRange_d,
                            double* tsOutResDEDX_d, double* tsOutResInvRange_d, int numTestCases ) {
   for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < numTestCases; i += blockDim.x * gridDim.x) {
     G4HepEmElectronManager theElectronMgr;
     tsOutResRange_d[i]    = theElectronMgr.GetRestRange(theElectronData_d, tsInImc_d[i], tsInEkin_d[i], tsInLogEkin_d[i]);
     tsOutResDEDX_d[i]     = theElectronMgr.GetRestDEDX (theElectronData_d, tsInImc_d[i], tsInEkin_d[i], tsInLogEkin_d[i]);
     tsOutResInvRange_d[i] = theElectronMgr.GetInvRange (theElectronData_d, tsInImc_d[i], tsOutResRange_d[i]);
   }
 }

void TestElossDataOnDevice ( const struct G4HepEmData* hepEmData,
     int* tsInImc_h, double* tsInEkin_h, double* tsInLogEkin_h,
     double* tsOutResRange_h, double* tsOutResDEDX_h, double* tsOutResInvRange_h,
     int numTestCases, bool iselectron ) {
  //
  // --- Allocate device side memory for the input/output data and copy all input
  //     data from host to device
  int*             tsInImc_d = nullptr;
  double*         tsInEkin_d = nullptr;
  double*      tsInLogEkin_d = nullptr;
  double*     tsOutResDEDX_d = nullptr;
  double*    tsOutResRange_d = nullptr;
  double* tsOutResInvRange_d = nullptr;
  //
  gpuErrchk ( cudaMalloc ( &tsInImc_d,          sizeof( int )    * numTestCases ) );
  gpuErrchk ( cudaMalloc ( &tsInEkin_d,         sizeof( double ) * numTestCases ) );
  gpuErrchk ( cudaMalloc ( &tsInLogEkin_d,      sizeof( double ) * numTestCases ) );
  gpuErrchk ( cudaMalloc ( &tsOutResDEDX_d,     sizeof( double ) * numTestCases ) );
  gpuErrchk ( cudaMalloc ( &tsOutResRange_d,    sizeof( double ) * numTestCases ) );
  gpuErrchk ( cudaMalloc ( &tsOutResInvRange_d, sizeof( double ) * numTestCases ) );
  //
  // --- Copy the input data from host to device (test material-cut index, ekin and log-ekin arrays)
  gpuErrchk ( cudaMemcpy ( tsInImc_d,     tsInImc_h,     sizeof( int )    * numTestCases, cudaMemcpyHostToDevice) );
  gpuErrchk ( cudaMemcpy ( tsInEkin_d,    tsInEkin_h,    sizeof( double ) * numTestCases, cudaMemcpyHostToDevice) );
  gpuErrchk ( cudaMemcpy ( tsInLogEkin_d, tsInLogEkin_h, sizeof( double ) * numTestCases, cudaMemcpyHostToDevice) );
  //
  // --- Launch the kernels
  int numThreads = 512;
  int numBlocks  = std::ceil( float(numTestCases)/numThreads );
  // std::cout << " N = " << numTestCases << " numBlocks = " << numBlocks << " numThreads = " << numThreads << " x = " << numBlocks*numThreads << std::endl;
  struct G4HepEmElectronData* elData_d = iselectron ? hepEmData->fTheElectronData_gpu : hepEmData->fThePositronData_gpu;
  TestElossDataKernel <<< numBlocks, numThreads >>> (elData_d, tsInImc_d, tsInEkin_d, tsInLogEkin_d, tsOutResRange_d, tsOutResDEDX_d, tsOutResInvRange_d, numTestCases );
  //
  // --- Synchronize to make sure that completed on the device
  cudaDeviceSynchronize();
  //
  // --- Copy the results from the device to the host
  gpuErrchk ( cudaMemcpy ( tsOutResDEDX_h,     tsOutResDEDX_d,     sizeof( double ) * numTestCases, cudaMemcpyDeviceToHost ) );
  gpuErrchk ( cudaMemcpy ( tsOutResRange_h,    tsOutResRange_d,    sizeof( double ) * numTestCases, cudaMemcpyDeviceToHost ) );
  gpuErrchk ( cudaMemcpy ( tsOutResInvRange_h, tsOutResInvRange_d, sizeof( double ) * numTestCases, cudaMemcpyDeviceToHost ) );
  //
  // --- Free all dynamically allocated (device side) memory
  cudaFree ( tsInImc_d          );
  cudaFree ( tsInEkin_d         );
  cudaFree ( tsInLogEkin_d      );
  cudaFree ( tsOutResDEDX_d     );
  cudaFree ( tsOutResRange_d    );
  cudaFree ( tsOutResInvRange_d );
}
