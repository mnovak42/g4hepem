
#include "Declaration.hh"

#include "G4HepEmData.hh"
#include "G4HepEmElectronData.hh"

#include <cuda_runtime.h>
#include "G4HepEmCuUtils.hh"


//
// Note: both specialisations (needed to be called from the host) are done in 
//  this .cu file below in the TestResMacXSecDataOnDevice function.
template <bool TisSBModel>
__global__
void TestElemSelectorDataBremKernel ( const struct G4HepEmElectronDataOnDevice* theElectronData_d, 
     int* tsInImc_d, double* tsInEkin_d, double* tsInLogEkin_d, double* tsInRngVals_d, 
     int* tsOutRes_d, int numTestCases ) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < numTestCases) {
      // the matrial-cut index
      int imc = tsInImc_d[tid];
      // get start index of the data for this material-cut: 
      // NOTE: start index = -1 in case of single element material, i.e. no selector
      int i0 = TisSBModel ? theElectronData_d->fElemSelectorBremSBDataStart[imc] : theElectronData_d->fElemSelectorBremRBDataStart[imc];
      // NOTE: one should try to avoid to call this kernel for materials with single element !!!
      if ( i0 < 0 ) {
        tsOutRes_d[tid] = 0;
      } else {
        int     numElem = theElectronData_d->fElemSelectorNumElements[imc];
        int     numData;
        double  logE0;
        double  invLD;
        double* xdata;
        if (TisSBModel) {
          numData = theElectronData_d->fElemSelectorNumBremSBData[imc];
          logE0   = theElectronData_d->fElemSelectorBremSBAuxData[2*imc];
          invLD   = theElectronData_d->fElemSelectorBremSBAuxData[2*imc+1];
          xdata   = &(theElectronData_d->fElemSelectorBremSBData[i0]);
        } else {
          numData = theElectronData_d->fElemSelectorNumBremRBData[imc];
          logE0   = theElectronData_d->fElemSelectorBremRBAuxData[2*imc];
          invLD   = theElectronData_d->fElemSelectorBremRBAuxData[2*imc+1];
          xdata   = &(theElectronData_d->fElemSelectorBremRBData[i0]);
        }
        // make sure that $x \in  [x[0],x[ndata-1]]$
        double xv      = max( xdata[0], min( xdata[ numElem * ( numData - 1 ) ], tsInEkin_d[ tid ] ) );
        // compute the lowerindex of the x bin (idx \in [0,N-2] will be guaranted)
        int idxEkin    = __double2int_rz( max( 0.0, min( (tsInLogEkin_d[tid]  -logE0) * invLD, numData - 2.0 ) ) );
        // the real index position is idxEkin x numElem
        int indx0      = idxEkin * numElem;
        int indx1      = indx0 + numElem;
        // linear interpolation
        double x1      = xdata[ indx0++ ];
        double x2      = xdata[ indx1++ ];
        double dl      = x2-x1;
        double  b      = max( 0., min( 1., (xv - x1) / dl ) );
        int  theElemIndex = 0;
        // discrete probabilities, for selecting a given element, are from element index of 0 till #elements-2
        // NOTE: non-deterministic while loop can be turned to deterministic sampling tables for the underlying 
        //       discrete distributions (using Alias table) and combining them with statistical interpolation.
        while ( theElemIndex < numElem-1 && tsInRngVals_d[tid] > xdata[indx0+theElemIndex]+b*(xdata[indx1+theElemIndex]-xdata[indx0+theElemIndex])) { ++theElemIndex; }
        tsOutRes_d[tid] = theElemIndex;   
      }
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
  const struct G4HepEmElectronDataOnDevice* theElectronData = iselectron ? hepEmData->fTheElectronData_gpu : hepEmData->fThePositronData_gpu;
  int numThreads = 512;
  int numBlocks  = std::ceil( float(numTestCases)/numThreads );
  switch (indxModel) {
    case 0: // not used
      break;
    
    case 1:
      
      TestElemSelectorDataBremKernel <  true > <<< numBlocks, numThreads >>> (theElectronData, tsInImc_d, tsInEkin_d, tsInLogEkin_d, tsInRngVals_d, tsOutRes_d, numTestCases );
      break;
      
    case 2:  
      TestElemSelectorDataBremKernel < false > <<< numBlocks, numThreads >>> (theElectronData, tsInImc_d, tsInEkin_d, tsInLogEkin_d, tsInRngVals_d, tsOutRes_d, numTestCases );
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



