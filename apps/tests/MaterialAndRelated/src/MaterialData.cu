
#include "Declaration.hh"

#include "G4HepEmData.hh"
#include "G4HepEmMaterialData.hh"

#include <iostream>
#include <random>

#include <cuda_runtime.h>
#include "G4HepEmCuUtils.hh"


// `numTestCases` HepEm materials are selected (by their indices) uniformly
// random from the list of materials used in the geometry. The stored material
// properties for all these test cases are obtained both on the host and on the
// device using the host and the device side data structures respectively.
// FAILURE is reported in case of any differences, SUCCESS is returned otherwise.


// Kernel to evaluate the G4HepEmMaterialData for the test cases stored on the
// device side main memory
//
// The element composition data, i.e. the list of elements (their atomic number
// as integer) and the correspondig element density (i.e. number of atoms per unit
// volume) will be returned concatenated into the `resCompX_d (integer/double)
// arrays. The `indxStarts_d` array stores the start indices of these data for
// the individual test cases.
//
// The kernel will be launched with an individual block of threads for each test
// cases and each of the threads within a block will take care of one element
// compositions data. The thread, with zero index, for each block will take care
// of all other (number of element, mass and electron densities) data.
__global__
void TestMaterialDataKernel (struct G4HepEmMaterialData* matData_d, int* matIndices_d, int* indxStarts_d,
                             double* resCompADens_d, int* resCompElems_d,  int* resNumElems_d,
                             double* resMassDens_d, double* resElecDens_d, double* resRadLen_d, int numTestCases) {
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  if (bid < numTestCases) {
    int         imat = matIndices_d[bid];
    int    indxStart = indxStarts_d[bid];
    int numOfElement = matData_d->fMaterialData[imat].fNumOfElement;
    if (tid < numOfElement) {
      int indx = indxStart + tid;
      resCompElems_d[indx] = matData_d->fMaterialData[imat].fElementVect[tid];
      resCompADens_d[indx] = matData_d->fMaterialData[imat].fNumOfAtomsPerVolumeVect[tid];
    }
    if (tid == 0) {
      resNumElems_d[bid] = numOfElement;
      resMassDens_d[bid] = matData_d->fMaterialData[imat].fDensity;
      resElecDens_d[bid] = matData_d->fMaterialData[imat].fElectronDensity;
      resRadLen_d[bid] = matData_d->fMaterialData[imat].fRadiationLength;
    }
  }
}


// Material data test that compares the data stored on the host to that on the device sides
bool TestMaterialDataOnDevice ( const struct G4HepEmData* hepEmData ) {
  // get the G4HepEmMaterialData member of the top level G4HepEmData structure
  const struct G4HepEmMaterialData* matData = hepEmData->fTheMaterialData;
  //
  // --- Prepare test cases:
  //
  // number of material (used in the geometry) indices to generate for checking
  int  numTestCases = 1024;
  int  numHepEmMats = matData->fNumMaterialData;
  // set up an rng to get material indices on [0,numHepEmMats)
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(0, numHepEmMats-1);
  //
  // --- Allocate memory on the host:
  //
  // for the uniformly random material index values as test cases and for recording
  // their start index as the cumulated number of elements
  int*  matIndices_h = new int[numTestCases];
  int*  indxStarts_h = new int[numTestCases];
  int       cumIndx  = 0;
  int    maxNumElems = 0;
  for (int i=0; i<numTestCases; ++i) {
    int hepEmMatIndx = dis(gen);
    matIndices_h[i]  = hepEmMatIndx;
    indxStarts_h[i]  = cumIndx;
    int     numElems = matData->fMaterialData[hepEmMatIndx].fNumOfElement;
    cumIndx         += numElems;
    if ( numElems > maxNumElems ) {
      maxNumElems = numElems;
    }
  }
  // for the (integer and double) material composition data results on the host
  int*    resCompElems_h = new    int[cumIndx];
  double* resCompADens_h = new double[cumIndx];
  // for the number of elements, mass- and electron-density on the host
  int*     resNumElems_h = new    int[numTestCases];
  double*  resMassDens_h = new double[numTestCases];
  double*  resElecDens_h = new double[numTestCases];
  double*    resRadLen_h = new double[numTestCases];
  //
  // --- Allocate memory on the device:
  //
  // for the test material indices, the start index of their element composition
  // results, their element composition other results
  int*      matIndices_d = nullptr;
  int*      indxStarts_d = nullptr;
  int*     resNumElems_d = nullptr;
  int*    resCompElems_d = nullptr;
  double* resCompADens_d = nullptr;
  double*  resMassDens_d = nullptr;
  double*  resElecDens_d = nullptr;
  double*    resRadLen_d = nullptr;
  //
  gpuErrchk ( cudaMalloc ( &matIndices_d,   sizeof( int )    * numTestCases ) );
  gpuErrchk ( cudaMalloc ( &indxStarts_d,   sizeof( int )    * numTestCases ) );
  gpuErrchk ( cudaMalloc ( &resNumElems_d,  sizeof( int )    * numTestCases ) );
  gpuErrchk ( cudaMalloc ( &resCompElems_d, sizeof( int )    * cumIndx ) );
  gpuErrchk ( cudaMalloc ( &resCompADens_d, sizeof( double ) * cumIndx ) );
  gpuErrchk ( cudaMalloc ( &resMassDens_d,  sizeof( double ) * numTestCases ) );
  gpuErrchk ( cudaMalloc ( &resElecDens_d,  sizeof( double ) * numTestCases ) );
  gpuErrchk ( cudaMalloc ( &resRadLen_d,    sizeof( double ) * numTestCases ) );

  //
  // --- Copy the input data from host to device (test material index and start index arrays)
  //
  gpuErrchk ( cudaMemcpy ( matIndices_d, matIndices_h, sizeof( int ) * numTestCases, cudaMemcpyHostToDevice) );
  gpuErrchk ( cudaMemcpy ( indxStarts_d, indxStarts_h, sizeof( int ) * numTestCases, cudaMemcpyHostToDevice) );
  //
  // --- Launch the kernel to evaluate the test cases on the device: one block for each test
  //     cases with 32x threads i.e. at leasr one for each element composition data
  //
  int numThreadsPerBlock = maxNumElems/32 + (maxNumElems % 32 != 0);
  TestMaterialDataKernel <<< numTestCases, 32*numThreadsPerBlock >>> (hepEmData->fTheMaterialData_gpu, matIndices_d, indxStarts_d,
    resCompADens_d, resCompElems_d, resNumElems_d, resMassDens_d, resElecDens_d, resRadLen_d, numTestCases);
  //
  // synchronize to make sure that completed on the device
  cudaDeviceSynchronize();
  // copy the results from the device to the host
  gpuErrchk ( cudaMemcpy ( resNumElems_h,  resNumElems_d,  sizeof( int )    * numTestCases, cudaMemcpyDeviceToHost ) );
  gpuErrchk ( cudaMemcpy ( resCompElems_h, resCompElems_d, sizeof( int )    * cumIndx,      cudaMemcpyDeviceToHost ) );
  gpuErrchk ( cudaMemcpy ( resCompADens_h, resCompADens_d, sizeof( double ) * cumIndx,      cudaMemcpyDeviceToHost ) );
  gpuErrchk ( cudaMemcpy ( resMassDens_h,  resMassDens_d,  sizeof( double ) * numTestCases, cudaMemcpyDeviceToHost ) );
  gpuErrchk ( cudaMemcpy ( resElecDens_h,  resElecDens_d,  sizeof( double ) * numTestCases, cudaMemcpyDeviceToHost ) );
  gpuErrchk ( cudaMemcpy ( resRadLen_h,    resRadLen_d,    sizeof( double ) * numTestCases, cudaMemcpyDeviceToHost ) );

  //
  // --- Check the results for each test cases by comparing to the corresponding
  //     results obtained on the host side
  //
  bool isPassed = true;
  // loop over all test cases
  for (int i=0; i<numTestCases && isPassed; ++i) {
    // get the host side G4HepEmMatData structure
    const int            hepEmMatIndex = hepEmData->fTheMaterialData->fG4MatIndexToHepEmMatIndex[matIndices_h[i]];
    const struct G4HepEmMatData& heMat = hepEmData->fTheMaterialData->fMaterialData[hepEmMatIndex];
    // compare the host and device side properties: mass density, electron density, #elements
    if ( heMat.fDensity != resMassDens_h[i] ) {
      isPassed = false;
      std::cerr << "\n*** ERROR:\nMaterialData: HOST v.s. DEVICE mismatch fDensity != "         << heMat.fDensity         << " != "  << resMassDens_h[i] << std::endl;
      continue;
    }
    if ( heMat.fElectronDensity != resElecDens_h[i] ) {
      isPassed = false;
      std::cerr << "\n*** ERROR:\nMaterialData: HOST v.s. DEVICE mismatch fElectronDensity != " << heMat.fElectronDensity << " != "  << resElecDens_h[i] << std::endl;
      continue;
    }
    if ( heMat.fNumOfElement != resNumElems_h[i] ) {
      isPassed = false;
      std::cerr << "\n*** ERROR:\nMaterialData: HOST v.s. DEVICE mismatch fNumOfElement != "    << heMat.fNumOfElement    << " != "  << resNumElems_h[i] << std::endl;
      continue;
    }
    if ( heMat.fRadiationLength != resRadLen_h[i] ) {
      isPassed = false;
      std::cerr << "\n*** ERROR:\nMaterialData: HOST v.s. DEVICE mismatch fRadiationLength != " << heMat.fRadiationLength << " != "  << resRadLen_h[i] << std::endl;
      continue;
    }
    // obtain the element composition of the host side HepEm material data and comare to that obtained from the device
    const int          indxStart = indxStarts_h[i];
    const int       numOfElement = heMat.fNumOfElement;
    for (size_t ie=0; ie<numOfElement && isPassed; ++ie) {
      const int izet = resCompElems_h[indxStart+ie];
      if ( heMat.fElementVect[ie] != izet ) {
        isPassed = false;
        std::cerr << "\n*** ERROR:\nMaterialData:  HOST v.s. DEVICE mismatch heMat.fElementVect[ " << ie << "] = " << heMat.fElementVect[ie] << " != " << izet << std::endl;
        break;
      }
      const double atomDensity = resCompADens_h[indxStart+ie];
      if ( heMat.fNumOfAtomsPerVolumeVect[ie] != atomDensity ) {
        isPassed = false;
        std::cerr << "\n*** ERROR:\nMaterialData: HOST v.s. DEVICE mismatch heMat.fNumOfAtomsPerVolumeVect[ " << ie << "] = " << heMat.fNumOfAtomsPerVolumeVect[ie] << " != " << atomDensity << std::endl;
        break;
      }
    }
  }
  //
  // --- Free all dynamically allocated memory (both host and device)
  //
  delete []   matIndices_h;
  delete []   indxStarts_h;
  delete []   resNumElems_h;
  delete []   resCompElems_h;
  delete []   resCompADens_h;
  delete []   resMassDens_h;
  delete []   resElecDens_h;
  delete []   resRadLen_h;
  cudaFree (  matIndices_d   );
  cudaFree (  indxStarts_d   );
  cudaFree (  resNumElems_d  );
  cudaFree (  resCompElems_d );
  cudaFree (  resCompADens_d );
  cudaFree (  resMassDens_d  );
  cudaFree (  resElecDens_d  );
  cudaFree (  resRadLen_d    );
  //
  return isPassed;
}
