
#include "Declaration.hh"

#include "G4HepEmData.hh"
#include "G4HepEmElectronData.hh"

#include <cuda_runtime.h>
#include "G4HepEmCuUtils.hh"



// Kernel to evaluate the Restricted macroscopic cross section values for the 
// test cases. The required data are stored in the ResMacXSec data part of the 
// G4HepEmElectronData structrue.
//
// Note: both specialisations (needed to be called from the host) are done in 
//  this .cu file below in the TestResMacXSecDataOnDevice function.

template <bool TIsIoni>
__global__
void TestResMacXSecDataKernel ( struct G4HepEmElectronDataOnDevice* theElectronData_d, 
     int* tsInImc_d, double* tsInEkin_d, double* tsInLogEkin_d, double* tsOutRes_d, 
     int numTestCases) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < numTestCases) {
      // the matrial-cut index
      int imc        = tsInImc_d[tid];
      // Compute lower index of the discrete kinetic energy bin:
      //  - index, at which the data for this mat-cut (with index of tsInImc_d[tid]) starts  
      int i0         = ( TIsIoni ) ? theElectronData_d->fResMacXSecIoniDataStart[imc]   : theElectronData_d->fResMacXSecBremDataStart[imc];
      int numData    = ( TIsIoni ) ? theElectronData_d->fResMacXSecNumIoniData[imc]     : theElectronData_d->fResMacXSecNumBremData[imc]   ; 
      // get the kinetic energy, macroscopic xsecion and second derivative grids 
      double* xdata  = ( TIsIoni ) ? &(theElectronData_d->fResMacXSecIoniEData[i0])     : &(theElectronData_d->fResMacXSecBremEData[i0]);
      double* ydata  = ( TIsIoni ) ? &(theElectronData_d->fResMacXSecIoniData[i0])      : &(theElectronData_d->fResMacXSecBremData[i0]);
      double* sdData = ( TIsIoni ) ? &(theElectronData_d->fResMacXSecIoniSDData[i0])    : &(theElectronData_d->fResMacXSecBremSDData[i0]);
      // get the auxiliary data used to compte the lower index of the kinetic energy bin
      double  logE0  = ( TIsIoni ) ? theElectronData_d->fResMacXSecIoniAuxData[4*imc+2] : theElectronData_d->fResMacXSecBremAuxData[4*imc+2];
      double  invLD  = ( TIsIoni ) ? theElectronData_d->fResMacXSecIoniAuxData[4*imc+3] : theElectronData_d->fResMacXSecBremAuxData[4*imc+3];
      // make sure that $x \in  [x[0],x[ndata-1]]$
      double xv      = max( xdata[0], min( xdata[numData-1], tsInEkin_d[tid] ) );
      // compute the lowerindex of the x bin (idx \in [0,N-2] will be guaranted)
      int idxEkin    = __double2int_rz( max( 0.0, min( ( tsInLogEkin_d[tid]-logE0 ) * invLD, numData-2.0 ) ) );
      //
      // perform the spline interpolation
      double x1  = xdata[idxEkin];
      double x2  = xdata[idxEkin+1];
      double dl  = x2-x1;
      double  b  = max( 0., min( 1., (xv - x1) / dl ) );
      //
      double os  = 0.166666666667; // 1./6.
      double  a  = 1.0 - b;
      double c0  = (a*a*a-a)*sdData[idxEkin];
      double c1  = (b*b*b-b)*sdData[idxEkin+1];
      tsOutRes_d[tid] = a*ydata[idxEkin] + b*ydata[idxEkin+1] + (c0+c1)*dl*dl*os; 
  }
}  


void TestResMacXSecDataOnDevice ( const struct G4HepEmData* hepEmData, int* tsInImc_h, 
     double* tsInEkinIoni_h, double* tsInLogEkinIoni_h, double* tsInEkinBrem_h, double* tsInLogEkinBrem_h,
     double* tsOutResMXIoni_h, double* tsOutResMXBrem_h, int numTestCases ) {
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
  TestResMacXSecDataKernel <  true > <<< numBlocks, numThreads >>> (hepEmData->fTheElectronData_gpu, tsInImc_d, tsInEkinIoni_d, tsInLogEkinIoni_d, tsOutResMXIoni_d, numTestCases );
  TestResMacXSecDataKernel < false > <<< numBlocks, numThreads >>> (hepEmData->fTheElectronData_gpu, tsInImc_d, tsInEkinBrem_d, tsInLogEkinBrem_d, tsOutResMXBrem_d, numTestCases );
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




