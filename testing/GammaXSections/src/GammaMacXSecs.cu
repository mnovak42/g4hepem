
#include "Declaration.hh"

#include "G4HepEmData.hh"
#include "G4HepEmGammaData.hh"

// don't worry it's just for testing
#define private public
#include "G4HepEmGammaManager.hh"

#include <cuda_runtime.h>
#include "G4HepEmCuUtils.hh"

// Pull in implementation
#include "G4HepEmGammaManager.icc"
#include "G4HepEmRunUtils.icc"

 __global__
 void TestMacXSecDataKernel ( const struct G4HepEmGammaData* theGammaData_d,
                              int* tsInImat_d, double* tsInEkin_d, double* tsInLogEkin_d,
                              double* tsOutRes_d, int iprocess, int numTestCases) {
   for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < numTestCases; i += blockDim.x * gridDim.x) {
     G4HepEmGammaManager theGammaMgr;
     tsOutRes_d[i] = theGammaMgr.GetMacXSec (theGammaData_d, tsInImat_d[i], tsInEkin_d[i], tsInLogEkin_d[i], iprocess);
   }
 }

void TestMacXSecDataOnDevice ( const struct G4HepEmData* hepEmData, int* tsInImat_h,
     double* tsInEkinConv_h, double* tsInLogEkinConv_h, double* tsInEkinComp_h, double* tsInLogEkinComp_h,
     double* tsOutMXConv_h, double* tsOutMXComp_h, int numTestCases ) {
  //
  // --- Allocate device side memory for the input/output data and copy all input
  //     data from host to device
  int*            tsInImat_d = nullptr;
  double*     tsInEkinConv_d = nullptr;
  double*  tsInLogEkinConv_d = nullptr;
  double*     tsInEkinComp_d = nullptr;
  double*  tsInLogEkinComp_d = nullptr;
  double*      tsOutMXConv_d = nullptr;
  double*      tsOutMXComp_d = nullptr;
  //
  gpuErrchk ( cudaMalloc ( &tsInImat_d,        sizeof( int )    * numTestCases ) );
  gpuErrchk ( cudaMalloc ( &tsInEkinConv_d,    sizeof( double ) * numTestCases ) );
  gpuErrchk ( cudaMalloc ( &tsInLogEkinConv_d, sizeof( double ) * numTestCases ) );
  gpuErrchk ( cudaMalloc ( &tsInEkinComp_d,    sizeof( double ) * numTestCases ) );
  gpuErrchk ( cudaMalloc ( &tsInLogEkinComp_d, sizeof( double ) * numTestCases ) );
  gpuErrchk ( cudaMalloc ( &tsOutMXConv_d,     sizeof( double ) * numTestCases ) );
  gpuErrchk ( cudaMalloc ( &tsOutMXComp_d,     sizeof( double ) * numTestCases ) );
  //
  // --- Copy the input data from host to device (test material index, ekin and log-ekin arrays)
  gpuErrchk ( cudaMemcpy ( tsInImat_d,        tsInImat_h,        sizeof( int )    * numTestCases, cudaMemcpyHostToDevice) );
  gpuErrchk ( cudaMemcpy ( tsInEkinConv_d,    tsInEkinConv_h,    sizeof( double ) * numTestCases, cudaMemcpyHostToDevice) );
  gpuErrchk ( cudaMemcpy ( tsInLogEkinConv_d, tsInLogEkinConv_h, sizeof( double ) * numTestCases, cudaMemcpyHostToDevice) );
  gpuErrchk ( cudaMemcpy ( tsInEkinComp_d,    tsInEkinComp_h,    sizeof( double ) * numTestCases, cudaMemcpyHostToDevice) );
  gpuErrchk ( cudaMemcpy ( tsInLogEkinComp_d, tsInLogEkinComp_h, sizeof( double ) * numTestCases, cudaMemcpyHostToDevice) );
  //
  // --- Launch the kernels
  int numThreads = 512;
  int numBlocks  = std::ceil( float(numTestCases)/numThreads );
  //  std::cout << " N = " << numTestCases << " numBlocks = " << numBlocks << " numThreads = " << numThreads << " x = " << numBlocks*numThreads << std::endl;
  const G4HepEmGammaData* theGammaData_d = hepEmData->fTheGammaData_gpu;
  // conversion
  TestMacXSecDataKernel <<< numBlocks, numThreads >>> (theGammaData_d, tsInImat_d, tsInEkinConv_d, tsInLogEkinConv_d, tsOutMXConv_d, 0,  numTestCases );
  // Compton scatteirng
  TestMacXSecDataKernel <<< numBlocks, numThreads >>> (theGammaData_d, tsInImat_d, tsInEkinComp_d, tsInLogEkinComp_d, tsOutMXComp_d, 1, numTestCases );
  //
  // --- Synchronize to make sure that completed on the device
  cudaDeviceSynchronize();
  //
  // --- Copy the results from the device to the host
  gpuErrchk ( cudaMemcpy ( tsOutMXConv_h,     tsOutMXConv_d,     sizeof( double ) * numTestCases, cudaMemcpyDeviceToHost ) );
  gpuErrchk ( cudaMemcpy ( tsOutMXComp_h,     tsOutMXComp_d,     sizeof( double ) * numTestCases, cudaMemcpyDeviceToHost ) );
  //
  // --- Free all dynamically allocated (device side) memory
  cudaFree ( tsInImat_d        );
  cudaFree ( tsInEkinConv_d    );
  cudaFree ( tsInLogEkinConv_d );
  cudaFree ( tsInEkinComp_d    );
  cudaFree ( tsInLogEkinComp_d );
  cudaFree ( tsOutMXConv_d    );
  cudaFree ( tsOutMXComp_d    );
}
