

#ifndef G4HepEmGSTableData_HH
#define G4HepEmGSTableData_HH

// tables for sampling angular deflection in MSC according to the Goudsmit -
// Saunderson angular distribution (at least 2 elastic scat. part only) using
// the screened Rutherford DCS for elastic scattering with Molier's screening
// parameters

struct G4HepEmGSTableData {
  // the 2D grid related parameters for both sets of dtr-s.
  const int    fLAMBNUM = 64;        // # L=s/lambda_el in [fLAMBMIN,fLAMBMAX]
  const int    fQNUM1   = 15;        // # Q=s/lambda_el G1 in [fQMIN1,fQMAX1] in the 1-st Q grid
  const int    fQNUM2   = 32;        // # Q=s/lambda_el G1 in [fQMIN2,fQMAX2] in the 2-nd Q grid
//  const int    fNUMSCR1 = 201;       // # of screening parameters in the A(G1) function
//  const int    fNUMSCR2 = 51;        // # of screening parameters in the A(G1) function
  const double fLAMBMIN = 1.0;       // minimum s/lambda_el
  const double fLAMBMAX = 100000.0;  // maximum s/lambda_el
  const double fQMIN1   = 0.001;     // minimum s/lambda_el G1 in the 1-st Q grid
  const double fQMAX1   = 0.99;      // maximum s/lambda_el G1 in the 1-st Q grid
  const double fQMIN2   = 0.99;      // minimum s/lambda_el G1 in the 2-nd Q grid
  const double fQMAX2   = 7.99;      // maximum s/lambda_el G1 in the 2-nd Q grid
  // derived values used to compute bin locations at run-time
  const double fLogLambda0        =  0.0;                // ln(fLAMBMIN)
  const double fLogDeltaLambda    =  0.1827448486503211; // ln(fLAMBMAX/fLAMBMIN)/(fLAMBNUM-1)
  const double fInvLogDeltaLambda =  5.4721104719809726; // 1/[ln(fLAMBMAX/fLAMBMIN)/(fLAMBNUM-1)]
  const double fInvDeltaQ1        = 14.1557128412537917; // 1/[(fQMAX1-fQMIN1)/(fQNUM1-1)]
  const double fDeltaQ2           =  0.2903225806451612; // [(fQMAX2-fQMIN2)/(fQNUM2-1)]
  const double fInvDeltaQ2        =  3.4444444444444455; // 1/[(fQMAX2-fQMIN2)/(fQNUM2-1)]
  // total number of dtr data and the start index of the data for a given iLambda and iQ combination
  int     fNumDtrData1    = 0;
  int     fNumDtrData2    = 0;
  int     fDtrDataStarts1[960];      // [fQNUM1 x fLAMBNUM]
  int     fDtrDataStarts2[2048];     // [fQNUM2 x fLAMBNUM]
  double* fGSDtrData1 = nullptr;
  double* fGSDtrData2 = nullptr;
  // the two sets of dtr data: each contain fLAMBNUM x fQNUM dtr data in a form of
  // ratin sampling tables. Each of these dtr are described by `3xN` data where `N`
  // `N` is stored as the very first value then:
  // [i]   = N
  // [i+1] = u_0, [i+2] = a_0,  [i+3] = b_0
  // ...
  // [i+3xN-2] = u_{N-1}, [i+3xN-1] = a_{N-1}, [i+3xN-0] = b_{N-1},
  // where `i`, for a given `iLambda` and `iQ` index combination is stored
  // at fDtrDataStarts[iLambda x iQ]
  //
  // Moliere's material dependent parameters and number of materials: parameters
  // `bc_i` and `Xc2_i` for the `i`-th material are stored at [i*2+0] and [i*2+1]
  int      fNumMaterials;
  double*  fMoliereParams = nullptr; // [2 x fNumMaterials]
  //
  // DPWA correction factors per materials
  const int     fPWACorNumEkin   = 31;
  const int     fPWACorNumBeta2  = 16;
  const double  fPWACorMinEkin   = 1.0E-3;
  const double  fPWACorMidEkin   = 0.1;
  const double  fPWACorMaxBeta2  = 0.9999;
  //
  const double  fPWACorMaxEkin       =  50.58889209000282;
  const double  fPWACorLogMinEkin    =  -6.907755278982137;
  const double  fPWACorInvLogDelEkin =   3.257208614274389;
  const double  fPWACorMinBeta2      =   0.300546131401874;
  const double  fPWACorInvDelBeta2   =  21.44836923554582;
  //
  // 3xfPWACorNumEkin values for each materials. Data for a give material with
  // HepEm material index of `imat` starts at `imat x 3 x fPWACorNumEkin`. This
  // corresponds to the first energy grid, i.e. the values that corresponds to
  // kinetic energy of `E_0=fPWACorMinEkin` starts at `i=imat x 3 x fPWACorNumEkin`
  // and arrange as [i] = corr-to-screening; [i+1] = corr-to-first-moment;
  // [i+2] = corr-to-fsecond-moment; followed by the 3 values at the next kinetic
  // energy. The complete size of the `fPWACorData` array is stored in `fPWACorDataNum`
  // (that should be equal to #HepEm-materials x 3 x fPWACorNumEkin).
  int       fPWACorDataNum;
  double*   fPWACorDataElectron = nullptr;
  double*   fPWACorDataPositron = nullptr;

};


// Allocates the dynamic parts of the G4HepEmGSTableData structure (done and filled in G4HepEmElectronInit
// with the usage of G4HepEmGSTableBuilder)
void AllocateGSTableData(struct G4HepEmGSTableData** theGSTableData, int numDtrData1, int numDtrData2, int numHepEmMat, int numPWACorData);

// Makes a new instance of G4HepEmGSTableData with the requested sizes for the dynamic components
G4HepEmGSTableData* MakeGSTableData(int numDtrData1, int numDtrData2, int numHepEmMat, int numPWACorData);

// Clears all the dynamic part of the G4HepEmGSTableData structure (filled in G4HepEmElectronInit by using
// G4HepEmGSTableBuilder)
void FreeGSTableData (struct G4HepEmGSTableData** theGSTableData);


#ifdef G4HepEm_CUDA_BUILD
  /**
    * Allocates memory for and copies the G4HepEmGSTableData structure from the
    * host to the device.
    *
    * @param onHost    pointer to the host side, already initialised G4HepEmGSTableData structure.
    * @param onDevice  host side address of a device side G4HepEmGSTableData structure memory pointer.
    *   The pointed memory is cleaned (if not null at input) and points to the device side memory at
    *   termination that stores the copied G4HepEmGSTableDataa structure.
    */
  void CopyGSTableDataToDevice(struct G4HepEmGSTableData* onHost, struct G4HepEmGSTableData** onDevice);

  /**
    * Frees all memory related to the device side G4HepEmGSTableDataa structure referred
    * by the pointer stored on the host side input argument address.
    *
    * @param onDevice host side address of a G4HepEmGSTableData structure located on the device side memory.
    *   The correspondig device memory will be freed and the input argument address will be set to null.
    */
  void FreeGSTableDataOnDevice(struct G4HepEmGSTableData** onDevice);
#endif // DG4HepEm_CUDA_BUILD


#endif // G4HepEmGSTables_HH
