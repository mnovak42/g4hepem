
#include "Declaration.hh"

#include "G4HepEmData.hh"
#include "G4HepEmGammaData.hh"

#include "G4HepEmGammaManager.hh"

#include <cuda_runtime.h>
#include "G4HepEmCuUtils.hh"

// Pull in implementation
#include "G4HepEmGammaManager.icc"
#include "G4HepEmRunUtils.icc"

 __global__
 void TestMacXSecDataKernel ( const struct G4HepEmGammaData* gmData_d, const struct G4HepEmMaterialData* matData_d,
                              int* tsInImat_d, double* tsInEkin_d, double* tsInLogEkin_d, double* tsInURand_d,
                              double* tsOutMXTot_d, int* tsOutProcID_d, G4HepEmGammaTrack* gTracks_d,
                              int numTestCases) {
   for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < numTestCases; i += blockDim.x * gridDim.x) {
     tsOutMXTot_d[i] = G4HepEmGammaManager::GetTotalMacXSec (gmData_d, matData_d, tsInImat_d[i], tsInEkin_d[i], tsInLogEkin_d[i], &(gTracks_d[i]));
     double totMFP = (tsOutMXTot_d[i] > 0) ? 1.0/tsOutMXTot_d[i] : 1E+20;
     if (tsOutMXTot_d[i]>0) { // otherwise IMFP would be such that we never call sampling
       gTracks_d[i].GetTrack()->SetMFP(totMFP, 0);
       G4HepEmGammaManager::SampleInteraction(gmData_d, &(gTracks_d[i]), tsInEkin_d[i], tsInLogEkin_d[i], tsInImat_d[i], tsInURand_d[i]); // sample interaction
       tsOutProcID_d[i] = gTracks_d[i].GetTrack()->GetWinnerProcessIndex();
     }
   }
 }

void TestMacXSecDataOnDevice ( const struct G4HepEmData* hepEmData, int* tsInImat_h,
                               double* tsInEkin_h, double* tsInLogEkin_h,
                               double* tsInURand_h, double* tsOutMXTot_h,
                               int* tsOutProcID_h, int numTestCases  ) {
  //
  // --- Allocate device side memory for the input/output data and copy all input
  //     data from host to device
  int*        tsInImat_d = nullptr;
  double*     tsInEkin_d = nullptr;
  double*  tsInLogEkin_d = nullptr;
  double*    tsInURand_d = nullptr;
  double*   tsOutMXTot_d = nullptr;
  int*     tsOutProcID_d = nullptr;

  //
  gpuErrchk ( cudaMalloc ( &tsInImat_d,    sizeof( int )    * numTestCases ) );
  gpuErrchk ( cudaMalloc ( &tsInEkin_d,    sizeof( double ) * numTestCases ) );
  gpuErrchk ( cudaMalloc ( &tsInLogEkin_d, sizeof( double ) * numTestCases ) );
  gpuErrchk ( cudaMalloc ( &tsInURand_d,   sizeof( double ) * numTestCases ) );
  gpuErrchk ( cudaMalloc ( &tsOutMXTot_d,  sizeof( double ) * numTestCases ) );
  gpuErrchk ( cudaMalloc ( &tsOutProcID_d, sizeof( int )    * numTestCases ) );
  //
  // --- Copy the input data from host to device (test material index, ekin and log-ekin arrays)
  gpuErrchk ( cudaMemcpy ( tsInImat_d,    tsInImat_h,    sizeof( int )    * numTestCases, cudaMemcpyHostToDevice) );
  gpuErrchk ( cudaMemcpy ( tsInEkin_d,    tsInEkin_h,    sizeof( double ) * numTestCases, cudaMemcpyHostToDevice) );
  gpuErrchk ( cudaMemcpy ( tsInLogEkin_d, tsInLogEkin_h, sizeof( double ) * numTestCases, cudaMemcpyHostToDevice) );
  gpuErrchk ( cudaMemcpy ( tsInURand_d,   tsInURand_h,   sizeof( double ) * numTestCases, cudaMemcpyHostToDevice) );

  G4HepEmGammaTrack* gTracks_d;
  gpuErrchk ( cudaMalloc ( &gTracks_d, sizeof( G4HepEmGammaTrack ) * numTestCases ) );

  //
  // --- Launch the kernels
  int numThreads = 512;
  int numBlocks  = std::ceil( float(numTestCases)/numThreads );

  //  std::cout << " N = " << numTestCases << " numBlocks = " << numBlocks << " numThreads = " << numThreads << " x = " << numBlocks*numThreads << std::endl;
  const G4HepEmGammaData* theGammaData_d  = hepEmData->fTheGammaData_gpu;
  const G4HepEmMaterialData* theMatData_d = hepEmData->fTheMaterialData_gpu;
  // conversion
  TestMacXSecDataKernel <<< numBlocks, numThreads >>> (theGammaData_d, theMatData_d, tsInImat_d, tsInEkin_d, tsInLogEkin_d, tsInURand_d, tsOutMXTot_d, tsOutProcID_d, gTracks_d, numTestCases );
  //
  // --- Synchronize to make sure that completed on the device
  cudaDeviceSynchronize();
  //
  // --- Copy the results from the device to the host
  gpuErrchk ( cudaMemcpy ( tsOutMXTot_h,  tsOutMXTot_d,  sizeof( double ) * numTestCases, cudaMemcpyDeviceToHost ) );
  gpuErrchk ( cudaMemcpy ( tsOutProcID_h, tsOutProcID_d, sizeof( int ) * numTestCases, cudaMemcpyDeviceToHost ) );

  //
  // --- Free all dynamically allocated (device side) memory
  cudaFree ( tsInImat_d    );
  cudaFree ( tsInEkin_d    );
  cudaFree ( tsInLogEkin_d );
  cudaFree ( tsInURand_d   );
  cudaFree ( tsOutMXTot_d  );
  cudaFree ( tsOutProcID_d );

  cudaFree ( gTracks_d    );
}
