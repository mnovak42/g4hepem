
#include "Declaration.hh"

#include "G4HepEmData.hh"
#include "G4HepEmElementData.hh"

#include <iostream>
#include <vector>
#include <cmath>
#include <random>

#include <cuda_runtime.h>
#include "G4HepEmCuUtils.hh"


// The list of elements used in the geometry are obtained, `numTestCases`
// elements are selected uniformly random from this list, the stored element
// properties for all these test cases are obtained both on the host and on the
// device using the host and the device side data structures respectively.
// FAILURE is reported in case of any differences, SUCCESS is returned otherwise.


// Kernel to evaluate the G4HepEmElementData for the test cases stored on the
// device main memory
__global__
void TestElementDataKernel (struct G4HepEmElementData* elemData_d, int* elemIndices_d, double* resZet_d, double* resZet13_d, int numTestCases) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < numTestCases) {
    // the atomic number of the element
    int izet = elemIndices_d[tid];
    resZet_d[tid]   = elemData_d->fElementData[izet].fZet;
    resZet13_d[tid] = elemData_d->fElementData[izet].fZet13;
  }
}

// Element data test that compares the data stored on the host v.s. device sides
bool TestElementDataOnDevice ( const struct G4HepEmData* hepEmData ) {
  // get the G4HepEmElementData member of the top level G4HepEmData structure
  const struct G4HepEmElementData* elemData = hepEmData->fTheElementData;
  //
  // --- Prepare test cases:
  //
  // number of (valid i.e. used in the geometry) element indices to generate for checking
  int  numTestCases = 1024;
  // collect all valid Z values
  std::vector<int> validZets;
  for ( int iz=0; iz<elemData->fMaxZet; ++iz ) {
    // check if this Z element data has been set
    int izet = (int)elemData->fElementData[iz].fZet;
    if ( elemData->fElementData[iz].fZet > 0 ) {
      validZets.push_back(izet);
    }
  }
  int numValidZet = (int)validZets.size();
  // set up an rng to get element indices on [0,numValidZet)
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(0, numValidZet-1);
  //
  // --- Allocate memory on the host:
  //
  // for the uniformly random valid element index values as test cases
  int* theElemIndices_h = new int[numTestCases];
  for (int i=0; i<numTestCases; ++i) {
    theElemIndices_h[i] = validZets[ dis(gen) ];
  }
  // for the results i.e. for all fZet, fZet13 values
  double*   theResZet_h = new double[numTestCases];
  double* theResZet13_h = new double[numTestCases];
  //
  // --- Allocate memory on the device:
  //
  // for the input elemement indices, for the resulted Z and Z^{1/3} values
  int* theElemIndices_d = nullptr;
  double*   theResZet_d = nullptr;
  double* theResZet13_d = nullptr;
  gpuErrchk ( cudaMalloc ( &theElemIndices_d, sizeof( int )    * numTestCases ) );
  gpuErrchk ( cudaMalloc ( &theResZet_d,      sizeof( double ) * numTestCases ) );
  gpuErrchk ( cudaMalloc ( &theResZet13_d,    sizeof( double ) * numTestCases ) );
  // and copy the input test element indices from the host to the device memory
  gpuErrchk ( cudaMemcpy ( theElemIndices_d, theElemIndices_h, sizeof( int ) * numTestCases, cudaMemcpyHostToDevice ) );
  //
  // --- Launch the kernel (one thread for each test element cases), synchronize cop the results from device to host
  //
  TestElementDataKernel <<<1, numTestCases>>> ( hepEmData->fTheElementData_gpu, theElemIndices_d, theResZet_d, theResZet13_d, numTestCases);
  // synchronize to make sure that completed on the device
  cudaDeviceSynchronize();
  // copy the resulted Z and Z^{1/3} from the device to the host
  gpuErrchk ( cudaMemcpy ( theResZet_h,     theResZet_d, sizeof( double ) * numTestCases, cudaMemcpyDeviceToHost ) );
  gpuErrchk ( cudaMemcpy ( theResZet13_h, theResZet13_d, sizeof( double ) * numTestCases, cudaMemcpyDeviceToHost ) );
  //
  // --- Check the results for each test cases by comparing to the corresponding
  //     results obtained on the host side
  //
  bool isPassed = true;
  // loop over all test cases
  for (int i=0; i<numTestCases && isPassed; ++i) {
    // obtain the correspondig host side G4HepEmElemData
    const int izet = theElemIndices_h[i];
    const struct G4HepEmElemData& theElemData_h = hepEmData->fTheElementData->fElementData[izet];
    // compare the host data structure values to those in the results obtained on the device
    const double zet = theElemData_h.fZet;
    if ( zet != theResZet_h[i] ) {
      isPassed = false;
      std::cerr << "\n*** ERROR:\nG4HepEmElementData: HOST v.s. DEVICE mismatch: fZet = " << zet << " != " << (double)theResZet_h[i] << std::endl;
      break;
    }
    const double zet13 = theElemData_h.fZet13;
    if ( zet13 != theResZet13_h[i] ) {
      isPassed = false;
      std::cerr << "\n*** ERROR:\nG4HepEmElementData: HOST v.s. DEVICE mismatch: fZet13 = " << zet13 << " != " << (double)theResZet13_h[i] << std::endl;
      break;
    }
  }
  //
  // --- Free all dynamically allocated memory  (both host and device)
  //
  delete []  theElemIndices_h;
  delete []  theResZet_h;
  delete []  theResZet13_h;
  cudaFree ( theElemIndices_d );
  cudaFree ( theResZet_d      );
  cudaFree ( theResZet13_d    );
  //
  return isPassed;
}
