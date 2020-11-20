
#include "G4HepEmElectronData.hh"
#include <iostream>

#include <cuda_runtime.h>
#include "G4HepEmCuUtils.hh"


void CopyElectronDataToDevice(struct G4HepEmElectronData* onHOST, struct G4HepEmElectronDataOnDevice** onDEVICE) {
  if ( !onHOST ) return;
  // clean away previous (if any)
  if ( *onDEVICE ) {
    FreeElectronDataOnDevice ( onDEVICE );
  }
  // Create a G4HepEmElectronDataOnDevice structure to store pointers to _d 
  // side arrays on the _h side.
  struct G4HepEmElectronDataOnDevice* elDataHTo_d = new G4HepEmElectronDataOnDevice;
  elDataHTo_d->fNumMatCuts = onHOST->fNumMatCuts;
  int numHepEmMatCuts      = elDataHTo_d->fNumMatCuts;
  // 
  // === ELoss data:
  // 
  // set non-pointer members  of the host side strcuture
  elDataHTo_d->fELossEnergyGridSize = onHOST->fELossEnergyGridSize;
  elDataHTo_d->fELossLogMinEkin     = onHOST->fELossLogMinEkin; 
  elDataHTo_d->fELossEILDelta       = onHOST->fELossEILDelta; 
  // allocate memory on _d for the ELoss energy grid and copy form _h
  int numELossData = onHOST->fELossEnergyGridSize;
  gpuErrchk ( cudaMalloc ( &(elDataHTo_d->fELossEnergyGrid),                        sizeof( double ) *numELossData ) );
  gpuErrchk ( cudaMemcpy ( elDataHTo_d->fELossEnergyGrid, onHOST->fELossEnergyGrid, sizeof( double ) *numELossData, cudaMemcpyHostToDevice ) );
  //
  // allocate data on the host for the re-aranged ELossData arrays
  double* elDataRange_h      = new double[numELossData*numHepEmMatCuts];
  double* elDataRangeSD_h    = new double[numELossData*numHepEmMatCuts];
  double* elDataDEDX_h       = new double[numELossData*numHepEmMatCuts];
  double* elDataDEDXSD_h     = new double[numELossData*numHepEmMatCuts];
  double* elDataInvRangeSD_h = new double[numELossData*numHepEmMatCuts];
  int indxCont = 0;
  for (int imc=0; imc<numHepEmMatCuts; ++imc) {
    int iRangeStart   = imc*5*numELossData;
    int iDEDXStarts   = iRangeStart+2*numELossData;
    int iIRanSDStarts = iRangeStart+4*numELossData;
    for (int i=0; i<numELossData; ++i) {
      elDataRange_h[indxCont]        = onHOST->fELossData[iRangeStart  +2*i  ]; // Range
      elDataRangeSD_h[indxCont]      = onHOST->fELossData[iRangeStart  +2*i+1]; // its SD 
      elDataDEDX_h[indxCont]         = onHOST->fELossData[iDEDXStarts  +2*i  ]; // DEDX
      elDataDEDXSD_h[indxCont]       = onHOST->fELossData[iDEDXStarts  +2*i+1]; // its SD
      elDataInvRangeSD_h[indxCont++] = onHOST->fELossData[iIRanSDStarts+  i  ]; // inv. range SD
    }    
  }
  // allocate memory on the device and copy the loss data arrays to _d
  std::size_t theELossDataSize = sizeof( double )*numELossData*numHepEmMatCuts;
  gpuErrchk ( cudaMalloc ( &(elDataHTo_d->fELossDataRange),      theELossDataSize ) );
  gpuErrchk ( cudaMalloc ( &(elDataHTo_d->fELossDataRangeSD),    theELossDataSize ) );
  gpuErrchk ( cudaMalloc ( &(elDataHTo_d->fELossDataDEDX),       theELossDataSize ) );
  gpuErrchk ( cudaMalloc ( &(elDataHTo_d->fELossDataDEDXSD),     theELossDataSize ) );
  gpuErrchk ( cudaMalloc ( &(elDataHTo_d->fELossDataInvRangeSD), theELossDataSize ) );
  //
  gpuErrchk ( cudaMemcpy (   elDataHTo_d->fELossDataRange,       elDataRange_h,      theELossDataSize, cudaMemcpyHostToDevice ) );
  gpuErrchk ( cudaMemcpy (   elDataHTo_d->fELossDataRangeSD,     elDataRangeSD_h,    theELossDataSize, cudaMemcpyHostToDevice ) );
  gpuErrchk ( cudaMemcpy (   elDataHTo_d->fELossDataDEDX,        elDataDEDX_h,       theELossDataSize, cudaMemcpyHostToDevice ) );
  gpuErrchk ( cudaMemcpy (   elDataHTo_d->fELossDataDEDXSD,      elDataDEDXSD_h,     theELossDataSize, cudaMemcpyHostToDevice ) );
  gpuErrchk ( cudaMemcpy (   elDataHTo_d->fELossDataInvRangeSD,  elDataInvRangeSD_h, theELossDataSize, cudaMemcpyHostToDevice ) );

  // free auxilary memorys allocated on host
  delete[] elDataRange_h;
  delete[] elDataRangeSD_h;
  delete[] elDataDEDX_h;
  delete[] elDataDEDXSD_h;
  delete[] elDataInvRangeSD_h; 
  // 
  // === Restricted macroscopic scross section data:
  //
  // allocate memory for all arrays on _d
  int*       ioniDataStart_h = new int[numHepEmMatCuts];
  int*       ioniNumData_h   = new int[numHepEmMatCuts];
  int*       bremDataStart_h = new int[numHepEmMatCuts];
  int*       bremNumData_h   = new int[numHepEmMatCuts];
  double*    ioniAuxData_h   = new double[4*numHepEmMatCuts];
  double*    bremAuxData_h   = new double[4*numHepEmMatCuts];

  gpuErrchk ( cudaMalloc ( &(elDataHTo_d->fResMacXSecIoniDataStart), sizeof( int ) *numHepEmMatCuts ) );
  gpuErrchk ( cudaMalloc ( &(elDataHTo_d->fResMacXSecNumIoniData),   sizeof( int ) *numHepEmMatCuts ) );
  gpuErrchk ( cudaMalloc ( &(elDataHTo_d->fResMacXSecBremDataStart), sizeof( int ) *numHepEmMatCuts ) );
  gpuErrchk ( cudaMalloc ( &(elDataHTo_d->fResMacXSecNumBremData),   sizeof( int ) *numHepEmMatCuts ) );
  
  gpuErrchk ( cudaMalloc ( &(elDataHTo_d->fResMacXSecIoniAuxData),   sizeof( double ) *4*numHepEmMatCuts ) );
  gpuErrchk ( cudaMalloc ( &(elDataHTo_d->fResMacXSecBremAuxData),   sizeof( double ) *4*numHepEmMatCuts ) );
  
  // run through the fResMacXSecData and count the size of the sum ioni/brem data
  int sumIoniData = 0;
  int sumBremData = 0;
  for (int imc=0; imc<numHepEmMatCuts; ++imc) {
    int      is  = onHOST->fResMacXSecStartIndexPerMatCut[imc];
    int numIoni  = (int)onHOST->fResMacXSecData[is];
    int numBrem  = (int)onHOST->fResMacXSecData[is+5+3*numIoni];
    sumIoniData += numIoni;
    sumBremData += numBrem;
  }
  double* ioniEData_h  = new double[sumIoniData];
  double* ioniData_h   = new double[sumIoniData];
  double* ioniSDData_h = new double[sumIoniData];
  double* bremEData_h  = new double[sumBremData];
  double* bremData_h   = new double[sumBremData];
  double* bremSDData_h = new double[sumBremData];
  //
  gpuErrchk ( cudaMalloc ( &(elDataHTo_d->fResMacXSecIoniEData),  sizeof( double ) *sumIoniData ) );
  gpuErrchk ( cudaMalloc ( &(elDataHTo_d->fResMacXSecIoniData),   sizeof( double ) *sumIoniData ) );
  gpuErrchk ( cudaMalloc ( &(elDataHTo_d->fResMacXSecIoniSDData), sizeof( double ) *sumIoniData ) );
  //
  gpuErrchk ( cudaMalloc ( &(elDataHTo_d->fResMacXSecBremEData),  sizeof( double ) *sumBremData ) );
  gpuErrchk ( cudaMalloc ( &(elDataHTo_d->fResMacXSecBremData),   sizeof( double ) *sumBremData ) );
  gpuErrchk ( cudaMalloc ( &(elDataHTo_d->fResMacXSecBremSDData), sizeof( double ) *sumBremData ) );
  // 
  // populate all host side arrays
  int indxContIoni = 0;
  int indxContBrem = 0;
  for (int imc=0; imc<numHepEmMatCuts; ++imc) {
    int is = onHOST->fResMacXSecStartIndexPerMatCut[imc];
    int ioniStrt = is;
    int numIoni  = (int)onHOST->fResMacXSecData[ioniStrt];
    int bremStrt = ioniStrt+5+3*numIoni;
    int numBrem  = (int)onHOST->fResMacXSecData[bremStrt];
    // fill in where the ioni/brem data starts for this (index = imc) material 
    // cuts couple in the continuous data arrays
    ioniDataStart_h[imc]   = indxContIoni;
    bremDataStart_h[imc]   = indxContBrem;
    // the number of ioni/brem data for this mat-cuts couple (auxilary data stored
    // separately on the device)
    ioniNumData_h[imc]     = numIoni;
    bremNumData_h[imc]     = numBrem;
    // fill in the 4 remaining (the first was the #data) auxilary data
    ioniAuxData_h[4*imc+0] = onHOST->fResMacXSecData[ioniStrt+1]; // max-E
    ioniAuxData_h[4*imc+1] = onHOST->fResMacXSecData[ioniStrt+2]; // max-Val 
    ioniAuxData_h[4*imc+2] = onHOST->fResMacXSecData[ioniStrt+3]; // log(E_0)
    ioniAuxData_h[4*imc+3] = onHOST->fResMacXSecData[ioniStrt+4]; // 1/log-delta
    //
    bremAuxData_h[4*imc+0] = onHOST->fResMacXSecData[bremStrt+1]; // max-E
    bremAuxData_h[4*imc+1] = onHOST->fResMacXSecData[bremStrt+2]; // max-Val 
    bremAuxData_h[4*imc+2] = onHOST->fResMacXSecData[bremStrt+3]; // log(E_0)
    bremAuxData_h[4*imc+3] = onHOST->fResMacXSecData[bremStrt+4]; // 1/log-delta
    // fill in the ioni and brem data
    for (int i=0; i<numIoni; ++i) {
      ioniEData_h[indxContIoni]    = onHOST->fResMacXSecData[ioniStrt+5+3*i];
      ioniData_h[indxContIoni]     = onHOST->fResMacXSecData[ioniStrt+5+3*i+1];
      ioniSDData_h[indxContIoni++] = onHOST->fResMacXSecData[ioniStrt+5+3*i+2];
    }
    for (int i=0; i<numBrem; ++i) {
      bremEData_h[indxContBrem]    = onHOST->fResMacXSecData[bremStrt+5+3*i];
      bremData_h[indxContBrem]     = onHOST->fResMacXSecData[bremStrt+5+3*i+1];
      bremSDData_h[indxContBrem++] = onHOST->fResMacXSecData[bremStrt+5+3*i+2];
    }
  }
  //
  // copy all array data from _h to _d
  //
  // Ioni:
  gpuErrchk ( cudaMemcpy ( elDataHTo_d->fResMacXSecIoniDataStart, ioniDataStart_h, sizeof( int )    *numHepEmMatCuts,   cudaMemcpyHostToDevice ) );
  gpuErrchk ( cudaMemcpy ( elDataHTo_d->fResMacXSecNumIoniData,   ioniNumData_h,   sizeof( int )    *numHepEmMatCuts,   cudaMemcpyHostToDevice ) );
  gpuErrchk ( cudaMemcpy ( elDataHTo_d->fResMacXSecIoniAuxData,   ioniAuxData_h,   sizeof( double ) *4*numHepEmMatCuts, cudaMemcpyHostToDevice ) );
  gpuErrchk ( cudaMemcpy ( elDataHTo_d->fResMacXSecIoniEData,     ioniEData_h,     sizeof( double ) *sumIoniData,       cudaMemcpyHostToDevice ) );
  gpuErrchk ( cudaMemcpy ( elDataHTo_d->fResMacXSecIoniData,      ioniData_h,      sizeof( double ) *sumIoniData,       cudaMemcpyHostToDevice ) );
  gpuErrchk ( cudaMemcpy ( elDataHTo_d->fResMacXSecIoniSDData,    ioniSDData_h,    sizeof( double ) *sumIoniData,       cudaMemcpyHostToDevice ) );
  //
  // brem:
  gpuErrchk ( cudaMemcpy ( elDataHTo_d->fResMacXSecBremDataStart, bremDataStart_h, sizeof( int )    *numHepEmMatCuts,   cudaMemcpyHostToDevice ) );
  gpuErrchk ( cudaMemcpy ( elDataHTo_d->fResMacXSecNumBremData,   bremNumData_h,   sizeof( int )    *numHepEmMatCuts,   cudaMemcpyHostToDevice ) );
  gpuErrchk ( cudaMemcpy ( elDataHTo_d->fResMacXSecBremAuxData,   bremAuxData_h,   sizeof( double ) *4*numHepEmMatCuts, cudaMemcpyHostToDevice ) );
  gpuErrchk ( cudaMemcpy ( elDataHTo_d->fResMacXSecBremEData,     bremEData_h,     sizeof( double ) *sumBremData,       cudaMemcpyHostToDevice ) );
  gpuErrchk ( cudaMemcpy ( elDataHTo_d->fResMacXSecBremData,      bremData_h,      sizeof( double ) *sumBremData,       cudaMemcpyHostToDevice ) );
  gpuErrchk ( cudaMemcpy ( elDataHTo_d->fResMacXSecBremSDData,    bremSDData_h,    sizeof( double ) *sumBremData,       cudaMemcpyHostToDevice ) );
  //
  // free all auxilary memory allocated on the host side
  delete[]  ioniDataStart_h;
  delete[]  ioniNumData_h;
  delete[]  ioniAuxData_h;
  delete[]  ioniEData_h;
  delete[]  ioniData_h;
  delete[]  ioniSDData_h;
  //
  delete[]  bremDataStart_h;
  delete[]  bremNumData_h;
  delete[]  bremAuxData_h;
  delete[]  bremEData_h;
  delete[]  bremData_h;
  delete[]  bremSDData_h;
  //
  //  === Target element selector data (for ioni and brem EM models)
  //
  // allocate data for all arrays on _h and _d
  int* numElements_h      = new int[numHepEmMatCuts];
  int* ioniStart_h        = new int[numHepEmMatCuts];
  int* numIoni_h          = new int[numHepEmMatCuts];
  ioniAuxData_h           = new double[2*numHepEmMatCuts];
  int numIoniData         = onHOST->fElemSelectorIoniNumData;
  ioniData_h              = new double[numIoniData];
  //
  gpuErrchk ( cudaMalloc ( &(elDataHTo_d->fElemSelectorNumElements),   sizeof( int )    *numHepEmMatCuts ) );
  gpuErrchk ( cudaMalloc ( &(elDataHTo_d->fElemSelectorIoniDataStart), sizeof( int )    *numHepEmMatCuts ) );
  gpuErrchk ( cudaMalloc ( &(elDataHTo_d->fElemSelectorNumIoniData),   sizeof( int )    *numHepEmMatCuts ) );
  gpuErrchk ( cudaMalloc ( &(elDataHTo_d->fElemSelectorIoniAuxData),   sizeof( double ) *2*numHepEmMatCuts ) );
  gpuErrchk ( cudaMalloc ( &(elDataHTo_d->fElemSelectorIoniData),      sizeof( double ) *numIoniData     ) );
  //
  int* bremSBStart_h      = new int[numHepEmMatCuts];
  int* numBremSB_h        = new int[numHepEmMatCuts];
  double* bremSBAuxData_h = new double[2*numHepEmMatCuts];
  int numBremSBData       = onHOST->fElemSelectorBremSBNumData;
  double* bremSBData_h    = new double[numBremSBData];
  // 
  gpuErrchk ( cudaMalloc ( &(elDataHTo_d->fElemSelectorBremSBDataStart), sizeof( int )    *numHepEmMatCuts ) );
  gpuErrchk ( cudaMalloc ( &(elDataHTo_d->fElemSelectorNumBremSBData),   sizeof( int )    *numHepEmMatCuts ) );
  gpuErrchk ( cudaMalloc ( &(elDataHTo_d->fElemSelectorBremSBAuxData),   sizeof( double ) *2*numHepEmMatCuts ) );
  gpuErrchk ( cudaMalloc ( &(elDataHTo_d->fElemSelectorBremSBData),      sizeof( double ) *numBremSBData   ) );
  //
  int* bremRBStart_h      = new int[numHepEmMatCuts];
  int* numBremRB_h        = new int[numHepEmMatCuts];
  double* bremRBAuxData_h = new double[2*numHepEmMatCuts];
  int numBremRBData       = onHOST->fElemSelectorBremRBNumData;
  double* bremRBData_h    = new double[numBremRBData];
  //
  gpuErrchk ( cudaMalloc ( &(elDataHTo_d->fElemSelectorBremRBDataStart), sizeof( int )    *numHepEmMatCuts ) );
  gpuErrchk ( cudaMalloc ( &(elDataHTo_d->fElemSelectorNumBremRBData),   sizeof( int )    *numHepEmMatCuts ) );
  gpuErrchk ( cudaMalloc ( &(elDataHTo_d->fElemSelectorBremRBAuxData),   sizeof( double ) *2*numHepEmMatCuts ) );
  gpuErrchk ( cudaMalloc ( &(elDataHTo_d->fElemSelectorBremRBData),      sizeof( double ) *numBremRBData ) );
  //
  // populate the host side arrays with data
  indxContIoni = 0;
  int indxContBremSB = 0;
  int indxContBremRB = 0;
  for (int imc=0; imc<numHepEmMatCuts; ++imc) {
    // ioni: Moller-Bhabha
    int iStart = onHOST->fElemSelectorIoniStartIndexPerMatCut[imc];
    ioniStart_h[imc] = iStart; // might be -1 i.e. in case of single elemnt materials of E_min>=E_max i.e. no selector data
    if (iStart > -1) {
      ioniStart_h[imc]       = indxContIoni;  
      numIoni_h[imc]         = onHOST->fElemSelectorIoniData[iStart];
      numElements_h[imc]     = onHOST->fElemSelectorIoniData[iStart+1];
      ioniAuxData_h[2*imc]   = onHOST->fElemSelectorIoniData[iStart+2];
      ioniAuxData_h[2*imc+1] = onHOST->fElemSelectorIoniData[iStart+3];
      int allData = numIoni_h[imc]*numElements_h[imc];
      for (int i=0; i<allData; ++i) {
        ioniData_h[indxContIoni++] = onHOST->fElemSelectorIoniData[iStart+4+i];
      }
    }
    // brem: Seltzer-Berger
    iStart = onHOST->fElemSelectorBremSBStartIndexPerMatCut[imc];
    bremSBStart_h[imc] = iStart; // might be -1 i.e. in case of single elemnt materials of E_min>=E_max i.e. no selector data
    if (iStart > -1) {
      bremSBStart_h[imc]       = indxContBremSB;
      numBremSB_h[imc]         = onHOST->fElemSelectorBremSBData[iStart];
      numElements_h[imc]       = onHOST->fElemSelectorBremSBData[iStart+1];
      bremSBAuxData_h[2*imc]   = onHOST->fElemSelectorBremSBData[iStart+2];
      bremSBAuxData_h[2*imc+1] = onHOST->fElemSelectorBremSBData[iStart+3];
      int allData = numBremSB_h[imc]*numElements_h[imc];
      for (int i=0; i<allData; ++i) {
        bremSBData_h[indxContBremSB++] = onHOST->fElemSelectorBremSBData[iStart+4+i];
      }
    }
    // brem: relativistic 
    iStart = onHOST->fElemSelectorBremRBStartIndexPerMatCut[imc];
    bremRBStart_h[imc] = iStart; // might be -1 i.e. in case of single elemnt materials of E_min>=E_max i.e. no selector data
    if (iStart > -1) {
      bremRBStart_h[imc]       = indxContBremRB;
      numBremRB_h[imc]         = onHOST->fElemSelectorBremRBData[iStart];
      numElements_h[imc]       = onHOST->fElemSelectorBremRBData[iStart+1];
      bremRBAuxData_h[2*imc]   = onHOST->fElemSelectorBremRBData[iStart+2];
      bremRBAuxData_h[2*imc+1] = onHOST->fElemSelectorBremRBData[iStart+3];
      int allData = numBremRB_h[imc]*numElements_h[imc];
      for (int i=0; i<allData; ++i) {
        bremRBData_h[indxContBremRB++] = onHOST->fElemSelectorBremRBData[iStart+4+i];
      }
    }
  }
  // copy from _h to _d all arrays
  gpuErrchk ( cudaMemcpy ( elDataHTo_d->fElemSelectorNumElements,     numElements_h, sizeof( int )    *numHepEmMatCuts, cudaMemcpyHostToDevice ) );
  // ioni
  gpuErrchk ( cudaMemcpy ( elDataHTo_d->fElemSelectorIoniDataStart,     ioniStart_h, sizeof( int )    *numHepEmMatCuts, cudaMemcpyHostToDevice ) );
  gpuErrchk ( cudaMemcpy ( elDataHTo_d->fElemSelectorNumIoniData,         numIoni_h, sizeof( int )    *numHepEmMatCuts, cudaMemcpyHostToDevice ) );
  gpuErrchk ( cudaMemcpy ( elDataHTo_d->fElemSelectorIoniAuxData,     ioniAuxData_h, sizeof( double ) *2*numHepEmMatCuts, cudaMemcpyHostToDevice ) );
  gpuErrchk ( cudaMemcpy ( elDataHTo_d->fElemSelectorIoniData,           ioniData_h, sizeof( double ) *indxContIoni, cudaMemcpyHostToDevice ) );
  // brem: Seltzer-Berger
  gpuErrchk ( cudaMemcpy ( elDataHTo_d->fElemSelectorBremSBDataStart, bremSBStart_h, sizeof( int )    *numHepEmMatCuts, cudaMemcpyHostToDevice ) );
  gpuErrchk ( cudaMemcpy ( elDataHTo_d->fElemSelectorNumBremSBData,     numBremSB_h, sizeof( int )    *numHepEmMatCuts, cudaMemcpyHostToDevice ) );
  gpuErrchk ( cudaMemcpy ( elDataHTo_d->fElemSelectorBremSBAuxData, bremSBAuxData_h, sizeof( double ) *2*numHepEmMatCuts, cudaMemcpyHostToDevice ) );
  gpuErrchk ( cudaMemcpy ( elDataHTo_d->fElemSelectorBremSBData,       bremSBData_h, sizeof( double ) *indxContBremSB, cudaMemcpyHostToDevice ) );
  // brem: rel. brem
  gpuErrchk ( cudaMemcpy ( elDataHTo_d->fElemSelectorBremRBDataStart, bremRBStart_h, sizeof( int )    *numHepEmMatCuts, cudaMemcpyHostToDevice ) );
  gpuErrchk ( cudaMemcpy ( elDataHTo_d->fElemSelectorNumBremRBData,     numBremRB_h, sizeof( int )    *numHepEmMatCuts, cudaMemcpyHostToDevice ) );
  gpuErrchk ( cudaMemcpy ( elDataHTo_d->fElemSelectorBremRBAuxData, bremRBAuxData_h, sizeof( double ) *2*numHepEmMatCuts, cudaMemcpyHostToDevice ) );
  gpuErrchk ( cudaMemcpy ( elDataHTo_d->fElemSelectorBremRBData,       bremRBData_h, sizeof( double ) *indxContBremRB, cudaMemcpyHostToDevice ) );
  //
  // clean all dynamically allocated auxilary host memory
  delete[] numElements_h;
  //
  delete[] ioniStart_h;
  delete[] numIoni_h;
  delete[] ioniAuxData_h;
  delete[] ioniData_h;
  //
  delete[] bremSBStart_h;
  delete[] numBremSB_h;
  delete[] bremSBAuxData_h;
  delete[] bremSBData_h;
  //
  delete[] bremRBStart_h;
  delete[] numBremRB_h;
  delete[] bremRBAuxData_h;
  delete[] bremRBData_h;   
  //
  // then finaly copy the top level, i.e. the main struct with the already 
  // appropriate pointers to device side memory locations but stored on the host
  gpuErrchk ( cudaMalloc (  onDEVICE,              sizeof(  struct G4HepEmElectronDataOnDevice ) ) );
  gpuErrchk ( cudaMemcpy ( *onDEVICE, elDataHTo_d, sizeof(  struct G4HepEmElectronDataOnDevice ), cudaMemcpyHostToDevice ) );
  // and clean 
  delete elDataHTo_d;  
}


void FreeElectronDataOnDevice(struct G4HepEmElectronDataOnDevice** onDEVICE) {
  if (*onDEVICE) {
    // copy the on-device data bakc to host in order to be able to free the device
    // side dynamically allocated memories
    struct G4HepEmElectronDataOnDevice* onHostTo_d = new G4HepEmElectronDataOnDevice;
    gpuErrchk ( cudaMemcpy( onHostTo_d, onDEVICE, sizeof( struct G4HepEmElectronDataOnDevice ), cudaMemcpyDeviceToHost ) );
    // ELoss data 
    cudaFree( onHostTo_d->fELossEnergyGrid     );
    cudaFree( onHostTo_d->fELossDataRange      );
    cudaFree( onHostTo_d->fELossDataRangeSD    );
    cudaFree( onHostTo_d->fELossDataDEDX       );
    cudaFree( onHostTo_d->fELossDataDEDXSD     );
    cudaFree( onHostTo_d->fELossDataInvRangeSD );
    // Macr. cross sections for ioni/brem
    cudaFree( onHostTo_d->fResMacXSecIoniDataStart );
    cudaFree( onHostTo_d->fResMacXSecNumIoniData   );
    cudaFree( onHostTo_d->fResMacXSecBremDataStart );
    cudaFree( onHostTo_d->fResMacXSecNumBremData   );

    cudaFree( onHostTo_d->fResMacXSecIoniAuxData );
    cudaFree( onHostTo_d->fResMacXSecIoniEData   );
    cudaFree( onHostTo_d->fResMacXSecIoniData    );
    cudaFree( onHostTo_d->fResMacXSecIoniSDData  );

    cudaFree( onHostTo_d->fResMacXSecBremAuxData );
    cudaFree( onHostTo_d->fResMacXSecBremEData   );
    cudaFree( onHostTo_d->fResMacXSecBremData    );
    cudaFree( onHostTo_d->fResMacXSecBremSDData  );
    // Target element selectors for ioni and brem models    
    cudaFree( onHostTo_d->fElemSelectorNumElements     );
    cudaFree( onHostTo_d->fElemSelectorIoniDataStart   );
    cudaFree( onHostTo_d->fElemSelectorNumIoniData     );
    cudaFree( onHostTo_d->fElemSelectorBremSBDataStart );
    cudaFree( onHostTo_d->fElemSelectorNumBremSBData   );
    cudaFree( onHostTo_d->fElemSelectorBremRBDataStart );
    cudaFree( onHostTo_d->fElemSelectorNumBremRBData   );
    
    cudaFree( onHostTo_d->fElemSelectorIoniAuxData   );
    cudaFree( onHostTo_d->fElemSelectorBremSBAuxData );
    cudaFree( onHostTo_d->fElemSelectorBremRBAuxData );

    cudaFree( onHostTo_d->fElemSelectorIoniData   );
    cudaFree( onHostTo_d->fElemSelectorBremSBData );
    cudaFree( onHostTo_d->fElemSelectorBremRBData );
    
    // free the remaining device side electron data and set the host side ptr to null
    cudaFree( *onDEVICE );
    *onDEVICE = nullptr;
  }
}



