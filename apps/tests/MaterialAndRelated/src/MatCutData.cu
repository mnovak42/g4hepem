#include "Declaration.hh"

#include "G4HepEmData.hh"
#include "G4HepEmMatCutData.hh"

#include <iostream>
#include <random>

#include <cuda_runtime.h>
#include "G4HepEmCuUtils.hh"


// `numTestCases` HepEm material-cuts are selected (by their indices) uniformly
// random from the list of material-cuts used in the geometry. The stored material-
// cuts properties for all these test cases are obtained both on the host and on
// the device using the host and the device side data structures respectively.
// FAILURE is reported in case of any differences, SUCCESS is returned otherwise.


// Kernel to evaluate the G4HepEmMatCutData for the test cases stored on the
// device side main memory
//
__global__
void TestMatCutDataKernel (struct G4HepEmMatCutData* mcData_d, int* mcIndices_d,
                           double* resSecElCut_d, double* resSecGamCut_d, int* resMatIndx_d, int numTestCases) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < numTestCases) {
      int imc = mcIndices_d[tid];
      resMatIndx_d[tid]   = mcData_d->fMatCutData[imc].fHepEmMatIndex;
      resSecElCut_d[tid]  = mcData_d->fMatCutData[imc].fSecElProdCutE;
      resSecGamCut_d[tid] = mcData_d->fMatCutData[imc].fSecGamProdCutE;
    }
}


// Material-cut data test that compares the data stored on the host to that on the device sides
bool TestMatCutDataOnDevice ( const struct G4HepEmData* hepEmData ) {
  // get the G4HepEmMatCutData member of the top level G4HepEmData structure
  const struct G4HepEmMatCutData* mcData = hepEmData->fTheMatCutData;
  //
  // --- Prepare test cases:
  //
  // number of material-cuts (used in the geometry) indices to generate for checking
  int    numTestCases = 1024;
  int  numHepEmMatCut = mcData->fNumMatCutData;
  // set up an rng to get material-cuts indices on [0,numHepEmMatCut)
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(0, numHepEmMatCut-1);
  //
  // --- Allocate memory on the host:
  //
  // for the uniformly random material-cut index values
  int*  mcIndices_h = new int[numTestCases];
  for (int i=0; i<numTestCases; ++i) {
    mcIndices_h[i] = dis(gen);
  }
  // for the scondary e-, gamma production threshold energies and for the material
  // indices results on the host
  double*  resSecElCut_h = new double[numTestCases];
  double* resSecGamCut_h = new double[numTestCases];
  int*      resMatIndx_h = new    int[numTestCases];
  //
  // --- Allocate memory on the device:
  //
  // for the test material indices, the start index of their element composition
  // results, their element composition other results
  int*       mcIndices_d = nullptr;
  int*      resMatIndx_d = nullptr;
  double*  resSecElCut_d = nullptr;
  double* resSecGamCut_d = nullptr;
  //
  gpuErrchk ( cudaMalloc ( &mcIndices_d,    sizeof( int )    * numTestCases ) );
  gpuErrchk ( cudaMalloc ( &resMatIndx_d,   sizeof( int )    * numTestCases ) );
  gpuErrchk ( cudaMalloc ( &resSecElCut_d,  sizeof( double ) * numTestCases ) );
  gpuErrchk ( cudaMalloc ( &resSecGamCut_d, sizeof( double ) * numTestCases ) );
  //
  // --- Copy the input data from host to device (test material-cut index arrays)
  //
  gpuErrchk ( cudaMemcpy ( mcIndices_d, mcIndices_h, sizeof( int ) * numTestCases, cudaMemcpyHostToDevice) );
  //
  // --- Launch the kernel to evaluate the test cases on the device
  //
  TestMatCutDataKernel <<< 1, numTestCases >>> (hepEmData->fTheMatCutData_gpu, mcIndices_d,
                                                resSecElCut_d, resSecGamCut_d, resMatIndx_d, numTestCases);
  //
  // synchronize to make sure that completed on the device
  cudaDeviceSynchronize();
  // copy the results from the device to the host
  gpuErrchk ( cudaMemcpy ( resMatIndx_h,   resMatIndx_d,   sizeof( int )    * numTestCases, cudaMemcpyDeviceToHost ) );
  gpuErrchk ( cudaMemcpy ( resSecElCut_h,  resSecElCut_d,  sizeof( double ) * numTestCases, cudaMemcpyDeviceToHost ) );
  gpuErrchk ( cudaMemcpy ( resSecGamCut_h, resSecGamCut_d, sizeof( double ) * numTestCases, cudaMemcpyDeviceToHost ) );
  //
  // --- Check the results for each test cases by comparing to the corresponding
  //     results obtained on the host side
  //
  bool isPassed = true;
  // loop over all test cases
  for (int i=0; i<numTestCases && isPassed; ++i) {
    // get the host side G4HepEmMCCData structure
    const int imc = mcIndices_h[i];
    const struct G4HepEmMCCData& mcData = hepEmData->fTheMatCutData->fMatCutData[imc];
    // compare the host and device side properties
    if ( mcData.fSecElProdCutE != resSecElCut_h[i] ) {
      isPassed = false;
      std::cerr << "\n*** ERROR:\nMaterialData: HOST v.s. DEVICE mismatch fSecElProdCutE != "    << mcData.fSecElProdCutE    << " != "  <<  resSecElCut_h[i]  << std::endl;
      continue;
    }
    if ( mcData.fSecGamProdCutE != resSecGamCut_h[i] ) {
      isPassed = false;
      std::cerr << "\n*** ERROR:\nMaterialData: HOST v.s. DEVICE mismatch fSecGamProdCutE != "   << mcData.fSecGamProdCutE   << " != "  <<  resSecGamCut_h[i] << std::endl;
      continue;
    }
    if ( mcData.fHepEmMatIndex != resMatIndx_h[i] ) {
      isPassed = false;
      std::cerr << "\n*** ERROR:\nMaterialData: HOST v.s. DEVICE mismatch fHepEmMatIndex != "    << mcData.fHepEmMatIndex    << " != "  <<  resMatIndx_h[i]   << std::endl;
      continue;
    }
  }
  //
  // --- Free all dynamically allocated memory (both host and device)
  //
  delete []   mcIndices_h;
  delete []   resMatIndx_h;
  delete []   resSecElCut_h;
  delete []   resSecGamCut_h;
  cudaFree (  mcIndices_d    );
  cudaFree (  resMatIndx_d   );
  cudaFree (  resSecElCut_d  );
  cudaFree (  resSecGamCut_d );
  //
  return isPassed;
}
