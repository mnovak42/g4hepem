#include "Declaration.hh"

#include "G4HepEmData.hh"
#include "G4HepEmSBTableData.hh"

#include <cuda_runtime.h>
#include "G4HepEmCuUtils.hh"

#include <cstdio>

__global__
void TestSBTableDataKernel(struct G4HepEmSBTableData* theSBTables_d, int* theIData1_d, int* theIData2_d, int* theIData3_d, int* theIData4_d,
                           double* theDData1_d, double* theDData2_d, double* theDData3_d, double* theDData4_d, double* theDData5_d) {
   // one thread will get all data
   if (blockIdx.x * blockDim.x + threadIdx.x == 0) {
     // fMaxZet, fNumElEnergy, fNumKappa, fNumHepEmMatCuts, fNumElemsInMatCuts, fNumSBTableData
     theIData1_d[0] = theSBTables_d->fMaxZet;
     theIData1_d[1] = theSBTables_d->fNumElEnergy;
     theIData1_d[2] = theSBTables_d->fNumKappa;
     theIData1_d[3] = theSBTables_d->fNumHepEmMatCuts;
     theIData1_d[4] = theSBTables_d->fNumElemsInMatCuts;
     theIData1_d[5] = theSBTables_d->fNumSBTableData;
     //
     for (int i=0; i<121; ++i) {
       theIData2_d[i] = theSBTables_d->fSBTablesStartPerZ[i];
     }
     for (int i=0; i<theSBTables_d->fNumHepEmMatCuts; ++i) {
       theIData3_d[i] = theSBTables_d->fGammaCutIndxStartIndexPerMC[i];
     }
     for (int i=0; i<theSBTables_d->fNumElemsInMatCuts; ++i) {
       theIData4_d[i] = theSBTables_d->fGammaCutIndices[i];
     }
     //
     //
     for (int i=0; i<65; ++i) {
       theDData1_d[i] = theSBTables_d->fElEnergyVect[i];
     }
     for (int i=0; i<65; ++i) {
       theDData2_d[i] = theSBTables_d->fLElEnergyVect[i];
     }
     for (int i=0; i<54; ++i) {
       theDData3_d[i] = theSBTables_d->fKappaVect[i];
     }
     for (int i=0; i<54; ++i) {
       theDData4_d[i] = theSBTables_d->fKappaVect[i];
     }
     for (int i=0; i<theSBTables_d->fNumSBTableData; ++i) {
       theDData5_d[i] = theSBTables_d->fSBTableData[i];
     }
   }
}

bool TestSBTableData(const struct G4HepEmData* hepEmData) {
  const struct G4HepEmSBTableData* theSBTables = hepEmData->fTheSBTableData;
  // allocate arrays to store results of device side evaluation of data as:
  // integers in theIData1: fMaxZet, fNumElEnergy, fNumKappa, fNumHepEmMatCuts, fNumElemsInMatCuts, fNumSBTableData
  int  theIData1[6];
  int* theIData1_d;
  gpuErrchk ( cudaMalloc ( &theIData1_d, sizeof( int ) * 6 ) );
  // integers in theIData2: fSBTablesStartPerZ[121]
  int  theIData2[121];
  int* theIData2_d;
  gpuErrchk ( cudaMalloc ( &theIData2_d, sizeof( int ) * 121 ) );
  // integers in theIData3: fGammaCutIndxStartIndexPerMC[fNumHepEmMatCuts]
  int* theIData3 = new int[ theSBTables->fNumHepEmMatCuts ];
  int* theIData3_d;
  gpuErrchk ( cudaMalloc ( &theIData3_d, sizeof( int ) * theSBTables->fNumHepEmMatCuts ) );
  // integers in theIData4: fGammaCutIndices[fNumElemsInMatCuts]
  int* theIData4 = new int[ theSBTables->fNumElemsInMatCuts ];
  int* theIData4_d;
  gpuErrchk ( cudaMalloc ( &theIData4_d, sizeof( int ) * theSBTables->fNumElemsInMatCuts ) );
  // doubles  in theDData1: fElEnergyVect[65]
  double  theDData1[65];
  double* theDData1_d;
  gpuErrchk ( cudaMalloc ( &theDData1_d, sizeof( double ) * 65 ) );
  // doubles  in theDData2: fLElEnergyVect[65]
  double  theDData2[65];
  double* theDData2_d;
  gpuErrchk ( cudaMalloc ( &theDData2_d, sizeof( double ) * 65 ) );
  // doubles  in theDData3: fKappaVect[54]
  double  theDData3[54];
  double* theDData3_d;
  gpuErrchk ( cudaMalloc ( &theDData3_d, sizeof( double ) * 54 ) );
  // doubles  in theDData4: fLKappaVect[54]
  double  theDData4[54];
  double* theDData4_d;
  gpuErrchk ( cudaMalloc ( &theDData4_d, sizeof( double ) * 54 ) );
  // doubles  in theDData5: fSBTableData[fNumSBTableData]
  double* theDData5 = new double[ theSBTables->fNumSBTableData ];
  double* theDData5_d;
  gpuErrchk ( cudaMalloc ( &theDData5_d, sizeof( double ) * theSBTables->fNumSBTableData ) );
  //
  // --- Launch the kernel to obtain the data on device
  int numThreads = 32;
  int numBlocks  =  1;
  TestSBTableDataKernel <<< numBlocks, numThreads >>> (hepEmData->fTheSBTableData_gpu,
                                                       theIData1_d, theIData2_d, theIData3_d, theIData4_d,
                                                       theDData1_d, theDData2_d, theDData3_d, theDData4_d, theDData5_d);
  // --- Synchronize to make sure that completed on the device
  cudaDeviceSynchronize();
  // copy results from device to host
  gpuErrchk ( cudaMemcpy ( theIData1, theIData1_d, sizeof( int )    *   6, cudaMemcpyDeviceToHost) );
  gpuErrchk ( cudaMemcpy ( theIData2, theIData2_d, sizeof( int )    * 121, cudaMemcpyDeviceToHost) );
  gpuErrchk ( cudaMemcpy ( theIData3, theIData3_d, sizeof( int )    * theSBTables->fNumHepEmMatCuts, cudaMemcpyDeviceToHost) );
  gpuErrchk ( cudaMemcpy ( theIData4, theIData4_d, sizeof( int )    * theSBTables->fNumElemsInMatCuts, cudaMemcpyDeviceToHost) );
  gpuErrchk ( cudaMemcpy ( theDData1, theDData1_d, sizeof( double ) *  65, cudaMemcpyDeviceToHost) );
  gpuErrchk ( cudaMemcpy ( theDData2, theDData2_d, sizeof( double ) *  65, cudaMemcpyDeviceToHost) );
  gpuErrchk ( cudaMemcpy ( theDData3, theDData3_d, sizeof( double ) *  54, cudaMemcpyDeviceToHost) );
  gpuErrchk ( cudaMemcpy ( theDData4, theDData4_d, sizeof( double ) *  54, cudaMemcpyDeviceToHost) );
  gpuErrchk ( cudaMemcpy ( theDData5, theDData5_d, sizeof( double ) *  theSBTables->fNumSBTableData, cudaMemcpyDeviceToHost) );
  //
  // --- Check results
  bool isPassing = true;
  int faildAt = -1;
  while (true) {
    isPassing = ( theIData1[0] == theSBTables->fMaxZet);
    if ( !isPassing ) { faildAt = 0; break; }
    isPassing = ( theIData1[1] == theSBTables->fNumElEnergy);
    if ( !isPassing ) { faildAt = 1; break; }
    isPassing = ( theIData1[2] == theSBTables->fNumKappa);
    if ( !isPassing ) { faildAt = 2; break; }
    isPassing = ( theIData1[3] == theSBTables->fNumHepEmMatCuts);
    if ( !isPassing ) { faildAt = 3; break; }
    isPassing = ( theIData1[4] == theSBTables->fNumElemsInMatCuts);
    if ( !isPassing ) { faildAt = 4; break; }
    isPassing = ( theIData1[5] == theSBTables->fNumSBTableData);
    if ( !isPassing ) { faildAt = 5; break; }
    for (int i=0; i<121; ++i) {
      isPassing = ( theIData2[i] == theSBTables->fSBTablesStartPerZ[i]);
      if ( !isPassing ) { faildAt = 6; std::cout << theIData2[i] << " <-> "<< theSBTables->fSBTablesStartPerZ[i]<< std::endl; break; }
    }
    if ( !isPassing ) break;
    for (int i=0; i<theSBTables->fNumHepEmMatCuts; ++i) {
      isPassing = ( theIData3[i] == theSBTables->fGammaCutIndxStartIndexPerMC[i]);
      if ( !isPassing ) { faildAt = 7; break; }
    }
    if ( !isPassing ) break;
    for (int i=0; i<theSBTables->fNumElemsInMatCuts; ++i) {
      isPassing = ( theIData4[i] == theSBTables->fGammaCutIndices[i]);
      if ( !isPassing ) { faildAt = 8; break; }
    }
    break;
  }
  if (!isPassing) {
    std::cerr << "      .... Faild At  = " << faildAt << std::endl;
  }

  // Free all dynamically allocated mamaory
  delete [] theIData3;
  delete [] theIData4;
  delete [] theDData5;
  cudaFree ( theIData1_d );
  cudaFree ( theIData2_d );
  cudaFree ( theIData3_d );
  cudaFree ( theIData4_d );
  cudaFree ( theDData1_d );
  cudaFree ( theDData2_d );
  cudaFree ( theDData3_d );
  cudaFree ( theDData4_d );
  cudaFree ( theDData5_d );

  return isPassing;
}

