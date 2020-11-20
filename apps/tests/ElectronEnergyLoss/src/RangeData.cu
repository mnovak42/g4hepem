#include "Declaration.hh"

#include "G4HepEmData.hh"
#include "G4HepEmElectronData.hh"

#include <cuda_runtime.h>
#include "G4HepEmCuUtils.hh"


// Kernel to evaluate the `Restricted Range` and restricted dE/dx values for the 
// test cases. The required data are stored in the EnergyLoss data part of the 
// G4HepEmElectronData structrue. Note: it's only for testing the device side 
// data.
__global__
void TestElossDataRangeDEDXKernel ( struct G4HepEmElectronDataOnDevice* theElectronData_d, 
     int* tsInImc_d, double* tsInEkin_d, double* tsInLogEkin_d,
     double* tsOutRes_d, int numTestCases, const bool isRange ) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < numTestCases) {
      // compute lower index of the discrete kinetic energy bin:
      int numELossData = theElectronData_d->fELossEnergyGridSize; 
      // index, at which the ELossData for this mat-cut (with index of tsInImc_d[tid]) starts  
      int i0           = tsInImc_d[tid] * numELossData;
      // set the kinetic energy, range and second derivative arrays as `x`, `y` and `sd`
      double* xdata    = theElectronData_d->fELossEnergyGrid;
      double* ydata    = (isRange) ? theElectronData_d->fELossDataRange   : theElectronData_d->fELossDataDEDX;
      double* sdData   = (isRange) ? theElectronData_d->fELossDataRangeSD : theElectronData_d->fELossDataDEDXSD;
      // make sure that $x \in  [x[0],x[ndata-1]]$
      double xv        = max ( xdata[0], min( xdata[numELossData-1], tsInEkin_d[tid] ) );
      // compute the lowerindex of the x bin (idx \in [0,N-2] will be guaranted)
      int idxEkin      = __double2int_rz( max( 0.0, min( (tsInLogEkin_d[tid]-theElectronData_d->fELossLogMinEkin)*theElectronData_d->fELossEILDelta, numELossData-2.0) ) );
      int idxVal       = i0 + idxEkin;
      //
      // perform the spline interpolation:
      // NOTE: I could store 1/ElossGrid-delta array to save up the next 3 lines and division
      double x1  = xdata[idxEkin];
      double x2  = xdata[idxEkin+1];
      double dl  = x2-x1;
      double  b  = max(0., min(1., (xv - x1)/dl));
      //
      double os  = 0.166666666667; // 1./6.
      double  a  = 1.0 - b;
      double c0  = (a*a*a-a)*sdData[idxVal];
      double c1  = (b*b*b-b)*sdData[idxVal+1];
      tsOutRes_d[tid] = a*ydata[idxVal] + b*ydata[idxVal+1] + (c0+c1)*dl*dl*os; 
  }
}  



__device__
int   TestElossDataInvRangeFindBinKernel ( double* theRangeArray_d, int itsSize, double theRangeVal ) {
  int ml = -1;
  int mu = itsSize-1;    
  while (abs(mu-ml)>1) {
    int mav = 0.5*(ml+mu);
    if ( theRangeVal < theRangeArray_d[mav] ) { mu = mav; }
    else { ml = mav; }
  }
  return mu-1;  
}


// Kernel to evaluate the inverse `Restricted Range` values for the test cases. 
// The required data are stored in the EnergyLoss data part of the G4HepEmElectronData 
// structrue. Note: it's only for testing the device side data.
__global__
void TestElossDataInvRangeKernel ( struct G4HepEmElectronDataOnDevice* theElectronData_d, 
     int* tsInImc_d, double* tsInRange_d, double* tsOutRes_d, int numTestCases ) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < numTestCases) {
      // Index, at which the ELossData for this mat-cut (with index of tsInImc_d[tid]) starts  
      int numELossData = theElectronData_d->fELossEnergyGridSize; 
      int i0           = tsInImc_d[tid] * numELossData;
      // set the range, kinetic energy and second derivative arrays as `x`, `y` and `sd`
      double* xdata    = &(theElectronData_d->fELossDataRange[i0]);
      double* ydata    = theElectronData_d->fELossEnergyGrid;
      double* sdData   = &(theElectronData_d->fELossDataInvRangeSD[i0]);
      // make sure that $x \in  [x[0],x[ndata-1]]$
      double xv        = max ( xdata[0], min( xdata[numELossData-1]*(1.0-1.0E-15), tsInRange_d[tid] ) );
      // find lower index of the discrete range bin (each threds will perform a small b-search)
      int    ilower    = TestElossDataInvRangeFindBinKernel ( xdata, numELossData, xv );
      // interpolate
      double x1  = xdata[ilower];
      double x2  = xdata[ilower+1];
      double dl  = x2-x1;
      // note: all corner cases of the previous methods are covered and eventually
      //       gives b=0/1 that results in y=y0\y_{N-1} if e<=x[0]/e>=x[N-1] or
      //       y=y_i/y_{i+1} if e<x[i]/e>=x[i+1] due to small numerical errors
      double  b  = max(0., min(1., (xv - x1)/dl));
      //
      double os  = 0.166666666667; // 1./6.
      double  a  = 1.0 - b;
      double c0  = (a*a*a-a)*sdData[ilower];
      double c1  = (b*b*b-b)*sdData[ilower+1];
      tsOutRes_d[tid] = a*ydata[ilower] + b*ydata[ilower+1] + (c0+c1)*dl*dl*os; 
  }
}  



void TestElossDataOnDevice ( const struct G4HepEmData* hepEmData, 
     int* tsInImc_h, double* tsInEkin_h, double* tsInLogEkin_h,
     double* tsOutResRange_h, double* tsOutResDEDX_h, double* tsOutResInvRange_h, 
     int numTestCases ) {
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
  int numThreads = 1024;
  int numBlocks  = std::ceil( float(numTestCases)/numThreads );
//  std::cout << " N = " << numTestCases << " numBlocks = " << numBlocks << " numThreads = " << numThreads << " x = " << numBlocks*numThreads << std::endl;
  TestElossDataRangeDEDXKernel <<< numBlocks, numThreads >>> (hepEmData->fTheElectronData_gpu, tsInImc_d, tsInEkin_d, tsInLogEkin_d, tsOutResRange_d, numTestCases, true  );
  TestElossDataRangeDEDXKernel <<< numBlocks, numThreads >>> (hepEmData->fTheElectronData_gpu, tsInImc_d, tsInEkin_d, tsInLogEkin_d, tsOutResDEDX_d,  numTestCases, false );
  // range data need to be ready before calling the inverse range kernel ==> sync here
  cudaDeviceSynchronize();
  TestElossDataInvRangeKernel  <<< numBlocks, numThreads >>> (hepEmData->fTheElectronData_gpu, tsInImc_d, tsOutResRange_d, tsOutResInvRange_d,  numTestCases );
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




