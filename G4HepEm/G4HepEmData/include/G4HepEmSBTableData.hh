

#ifndef G4HepEmSBTableData_HH
#define G4HepEmSBTableData_HH

// tables for sampling energy transfer for the Seltzer-Berger brem model

struct G4HepEmSBTableData {
  // pre-prepared sampling tables are available:
  const int               fMaxZet           = 99; // max Z number
  const int               fNumElEnergy      = 65; // # e- kine (E_k) per Z
  const int               fNumKappa         = 54; // # red. photon eners per E_k
  // min/max electron kinetic energy usage limits
  double                  fLogMinElEnergy = 0.0;
  double                  fILDeltaElEnergy = 0.0;

  // e- kinetic energy and reduced photon energy grids and tehir logarithms
  double                  fElEnergyVect[65];   // [fNumElEnergy]
  double                  fLElEnergyVect[65];  // [fNumElEnergy]
  double                  fKappaVect[54];      // [fNumKappa]
  double                  fLKappaVect[54];     // [fNumKappa]

  int                     fNumHepEmMatCuts = 0;              // #hepEm-MC
  int                     fNumElemsInMatCuts = 0;            // #elements-in-all-hepEm-MC
  int*                    fGammaCutIndxStartIndexPerMC = nullptr;  // [fNumHepEmMatCuts]
  int*                    fGammaCutIndices = nullptr;              // [fNumElemsInMatCuts]
                                                                   // for each mat-cut and for each of their elements in the corresponding elemnt SB-table [ #elements-in-all-hepEm-MC]

  // data starts index for a given Z
  int                     fNumSBTableData = 0;     // # all data stored in fSBTableData
  int                     fSBTablesStartPerZ[121]; // max Z is 99 so all values above 99 will cast to 99 if any
  double*                 fSBTableData = nullptr;  // [fNumSBTableData]
  // for each Z:
  // - [0] #data
  // - [1] minE-grid index for table
  // - [2] maxE-grid index for table
  // - [3] #gamma-cuts i.e. #materia-cuts (with gamma-cut below upper model elenrgy i.e. 1 GeV)
  //       in which this Z appears
  // - then the S-tables for each energy grid i.e. (maxE-grid-index-minE-grid-indx)+1
  //   and each has ()#gamma-cuts + 3x#kappa-values) entries ==> i.e. for a given Z
  //   there are ([2]-[1]+1) x ([3] + 3x54) values stored.
};


// Allocates some of the dynamic part of the G4HepEmSBTableData structure (completed and filled in G4HepEmElectronInit)
void AllocateSBTableData(struct G4HepEmSBTableData** theSBTableData, int numHepEmMatCuts, int numElemsInMC, int numElemsUnique);

// Makes a new instance of G4HepEmSBTableData with the requested sizes for the dynamic components
G4HepEmSBTableData* MakeSBTableData(int numHepEmMatCuts, int numElemsInMC, int numElemsUnique);


// Clears all the dynamic part of the G4HepEmSBTableData structure (filled in G4HepEmElectronInit)
void FreeSBTableData (struct G4HepEmSBTableData** theSBTableData);


#ifdef G4HepEm_CUDA_BUILD
  /**
    * Allocates memory for and copies the G4HepEmSBTableData structure from the
    * host to the device.
    *
    * @param onHost    pointer to the host side, already initialised G4HepEmSBTableData structure.
    * @param onDevice  host side address of a device side G4HepEmSBTableData structure memory pointer.
    *   The pointed memory is cleaned (if not null at input) and points to the device side memory at
    *   termination that stores the copied G4HepEmSBTableDataa structure.
    */
  void CopySBTableDataToDevice(struct G4HepEmSBTableData* onHost, struct G4HepEmSBTableData** onDevice);

  /**
    * Frees all memory related to the device side G4HepEmSBTableDataa structure referred
    * by the pointer stored on the host side input argument address.
    *
    * @param onDevice host side address of a G4HepEmSBTableData structure located on the device side memory.
    *   The correspondig device memory will be freed and the input argument address will be set to null.
    */
  void FreeSBTableDataOnDevice(struct G4HepEmSBTableData** onDevice);
#endif // DG4HepEm_CUDA_BUILD


#endif // G4HepEmSBTables_HH
