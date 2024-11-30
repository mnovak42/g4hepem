
#include "Declaration.hh"

#include "G4HepEmData.hh"
#include "G4HepEmGammaData.hh"

#include "G4HepEmGammaManager.hh"

#include <cuda_runtime.h>
#include "G4HepEmCuUtils.hh"

// Pull in implementation
#include "G4HepEmGammaManager.icc"
#include "G4HepEmRunUtils.icc"

// a device side G4HepEm data: all its members pointing to host memory will be set to
// overwritten by the device side pointers (i.e. their _gpu correspondance)
__constant__ __device__ struct G4HepEmData hepEmData_d;

 __global__
 void TestMacXSecDataKernel ( G4HepEmGammaTrack* gTracks_d, double* tsInURand_d,
                              double* tsOutMXTot_d, int* tsOutProcID_d, int numTestCases) {
   for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < numTestCases; i += blockDim.x * gridDim.x) {
     tsOutMXTot_d[i] = G4HepEmGammaManager::GetTotalMacXSec (&hepEmData_d, &(gTracks_d[i]));
     double totMFP = (tsOutMXTot_d[i] > 0) ? 1.0/tsOutMXTot_d[i] : 1E+20;
     if (tsOutMXTot_d[i]>0) { // otherwise IMFP would be such that we never call sampling
       gTracks_d[i].GetTrack()->SetMFP(totMFP, 0);
       G4HepEmGammaManager::SampleInteraction(&hepEmData_d, &(gTracks_d[i]), tsInURand_d[i]); // sample interaction
       tsOutProcID_d[i] = gTracks_d[i].GetTrack()->GetWinnerProcessIndex();
     }
   }
 }

void TestMacXSecDataOnDevice( const struct G4HepEmData* hepEmData, int* tsInImat_h,
                               double* tsInEkin_h, double* /*tsInLogEkin_h*/,
                               double* tsInURand_h, double* tsOutMXTot_h,
                               int* tsOutProcID_h, int numTestCases  ) {
  //
  // --- Allocate device side memory for the input/output data and copy all input
  // data from host to device
  double* tsInURand_d = nullptr;
  double* tsOutMXTot_d = nullptr;
  int*    tsOutProcID_d = nullptr;
  G4HepEmGammaTrack* gTracks_d = nullptr;

  // prepare the gamm atracks on host by setting their fields accroding to the set values
  G4HepEmGammaTrack* gTracks_h = new G4HepEmGammaTrack[numTestCases];
  for (int i=0; i<numTestCases; ++i) {
    gTracks_h[i].GetTrack()->SetEKin(tsInEkin_h[i]);
    gTracks_h[i].GetTrack()->SetMCIndex(tsInImat_h[i]); // mat index can be used now as mc index
  }

  // make the HepEmData available on device: all members have their _gpu correspondance
  // that are already available on the device (was copied in the caller). What we do
  // here is to overwrite the memebrs pointing to host by those pointing to the device
  // This makes possible to use all HepEm functions without the _gpu business.
  G4HepEmData tmpHepEmData_d;
  tmpHepEmData_d.fTheMatCutData   = hepEmData->fTheMatCutData_gpu;
  tmpHepEmData_d.fTheMaterialData = hepEmData->fTheMaterialData_gpu;
  tmpHepEmData_d.fTheElementData  = hepEmData->fTheElementData_gpu;
  tmpHepEmData_d.fTheElectronData = hepEmData->fTheElectronData_gpu;
  tmpHepEmData_d.fThePositronData = hepEmData->fThePositronData_gpu;
  tmpHepEmData_d.fTheSBTableData  = hepEmData->fTheSBTableData_gpu;
  tmpHepEmData_d.fTheGammaData    = hepEmData->fTheGammaData_gpu;
  gpuErrchk ( cudaMemcpyToSymbol(hepEmData_d, &tmpHepEmData_d, sizeof(G4HepEmData) ) );

  //
  gpuErrchk ( cudaMalloc ( &tsInURand_d,   sizeof( double ) * numTestCases ) );
  gpuErrchk ( cudaMalloc ( &gTracks_d,     sizeof( G4HepEmGammaTrack ) * numTestCases ) );
  gpuErrchk ( cudaMalloc ( &tsOutMXTot_d,  sizeof( double ) * numTestCases ) );
  gpuErrchk ( cudaMalloc ( &tsOutProcID_d, sizeof( int )    * numTestCases ) );
  // --- Copy the input data from host to device (test material index, ekin and log-ekin arrays)
  gpuErrchk ( cudaMemcpy ( tsInURand_d, tsInURand_h, sizeof( double ) * numTestCases, cudaMemcpyHostToDevice) );
  gpuErrchk ( cudaMemcpy ( gTracks_d,   gTracks_h,   sizeof( G4HepEmGammaTrack ) * numTestCases, cudaMemcpyHostToDevice) );
  //
  // --- Launch the kernels
  int numThreads = 512;
  int numBlocks  = std::ceil( float(numTestCases)/numThreads );

  // conversion
  TestMacXSecDataKernel <<< numBlocks, numThreads >>> (gTracks_d, tsInURand_d, tsOutMXTot_d, tsOutProcID_d, numTestCases );
  //
  // --- Synchronize to make sure that completed on the device
  cudaDeviceSynchronize();
  //
  // --- Copy the results from the device to the host
  gpuErrchk ( cudaMemcpy ( tsOutMXTot_h,  tsOutMXTot_d,  sizeof( double ) * numTestCases, cudaMemcpyDeviceToHost ) );
  gpuErrchk ( cudaMemcpy ( tsOutProcID_h, tsOutProcID_d, sizeof( int ) * numTestCases, cudaMemcpyDeviceToHost ) );

  //
  // --- Free all dynamically allocated (device side) memory
  cudaFree ( tsInURand_d   );
  cudaFree ( gTracks_d     );
  cudaFree ( tsOutMXTot_d  );
  cudaFree ( tsOutProcID_d );
}
