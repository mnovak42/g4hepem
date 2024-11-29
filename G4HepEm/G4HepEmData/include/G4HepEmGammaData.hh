#ifndef G4HepEmGammaData_HH
#define G4HepEmGammaData_HH

/**
 * @file    G4HepEmGammaData.hh
 * @struct  G4HepEmGammaData
 * @author  M. Novak
 * @date    2021
 *
 * @brief All energy loss process related data used for \f$e^-/e+\f$ simulations by `G4HepEm`.
 *
 * Covers Gamma conversion itno e-/e+ pairs and Compton scattering at the moment.
 */

struct G4HepEmGammaData {
  /** Number of G4HepEm materials: number of G4HepEmMatData structures stored in the G4HepEmMaterialData::fMaterialData array. */
  int           fNumMaterials = 0;

//// === conversion related data. Grid: 146 bins form 2mc^2 - 100 TeV
  const int     fConvEnergyGridSize = 147;
  double        fConvLogMinEkin = 0.0;    // = 0.021759358706830;  // log(2mc^2)
  double        fConvEILDelta = 0.0;      // = 7.935247775833226;  // 1./[log(emax/emin)/146]
  double*       fConvEnergyGrid = nullptr;    // [fConvEnergyGrid]

//// === compton related data. 84 bins (7 per decades) from 100 eV - 100 TeV
  const int     fCompEnergyGridSize = 85;
  double        fCompLogMinEkin = 0.0;     // = -9.210340371976182; // log(0.0001) i.e. log(100 eV)
  double        fCompEILDelta = 0.0;       // =  3.040061373322763; // 1./[log(emax/emin)/84]
  double*       fCompEnergyGrid = nullptr;     // [fCompEnergyGridSize]

//// === gamma nuclear related data. 255 bins form  2mc^2 - 100 TeV
  const int     fGNucEnergyGridSize = 256;
  double        fGNucLogMinEkin = 0.0;     // =  0.021759358706830;  // log(2mc^2)
  double        fGNucEILDelta = 0.0;       // =  13.85950970842557;  // 1./[log(emax/emin)/255]
  double*       fGNucEnergyGrid = nullptr;     // [fGNucEnergyGridSize]


  // the macroscopic cross sections for all materials and for [conversion,compton,gamma-nuclear]
  // at each material
  double*       fConvCompGNucMacXsecData = nullptr;   // [#materials*2*(fConvEnergyGridSize+fCompEnergyGridSize+fGNucEnergyGridSize)]

//// === Macroscopic cross section related data:
  // The 100 eV 100 TeV kinetic energy range is divided up to 3 kinetic energy window. At a discrete kinetic
  // energy point, the following macroscopic cross section data are stored:
  // - window 0: 100 eV - 150 keV; only 1 data stored at each E_i
  //   1. the Compton scattering mac. xsec. (as PE is not smooth in this region)
  //   note: the total mac. xsec. is the sum of 1. above plus the PE mac. xsec. as Conversion and Gamma-Nuclear
  //         are zero in this energy window
  // - window 1: 150 keV - 2mc^2; 2 data are stored at at each E_i
  //    1. the sum of the Compton and PE mac. xsec
  //    2. and the PE mac. xsec alone
  //    note: conversion is still zero in this energy window and gamma-nuclear is assumed to be zero (very small)
  //          so 1. above is the total mac. xsec.
  // - window 2: 2mc^2 - 100 TeV; 4 data are stored at each E_i
  //    1. the sum of Conversion, Compton, PE, Gamma-Nuclear (GN) mac. xsec.
  //    2. the Conversion mac. xsec.
  //    3. the Compton mac. xsec.
  //    4. the PE mac. xsec.
  //    note: 1. above is the total mac. xsec.
  // NOTE: the total mac. xsec. can be used to determine how far the gamma goes till the next interaction while
  //       the additional mac. xsec. data are sufficient (togeter with the total) to determine the interaction
  //       at that point (if any)
  //
  // these grid densities provide a relativ error less than 0.5 %
  const int     fEGridSize0 =   32;
  const int     fEGridSize1 =   32;
  const int     fEGridSize2 =  256;

  int           fDataPerMat =   0;    // #data for one material in the fMacXsecData array
  int           fNumData0   =   0;    // #data for one material related to the first  (0th) ekin window
  int           fNumData1   =   0;    // #data for one material related to the second (1th) ekinwindow

  double        fEMin0      = 0.0;     // minimum kinetic energy of the first window (100.0*CLHEP::eV)
  double        fEMax0      = 0.0;     // minimum kinetic energy of the second window (150.0*CLHEP::eV)
  double        fLogEMin0   = 0.0;     // =  0.021759358706830;  // log(fEMin0)
  double        fEILDelta0  = 0.0;     // =  13.85950970842557;  // 1./[log(fEMax0/fEMin0)/(fEGridSize0-1)]

  // double        fEMin1  --> fEMax0
  double        fEMax1      = 0.0;
  double        fLogEMin1   = 0.0;
  double        fEILDelta1  = 0.0;

  // double        fEMin2  --> fEMax1
  double        fEMax2      = 0.0;
  double        fLogEMin2   = 0.0;
  double        fEILDelta2  = 0.0;

  double*       fMacXsecData; // [#materials x fDataPerMat]



//// === element selector for conversion (note: KN compton interaction do not know anything about Z)
  int           fElemSelectorConvEgridSize = 0;
  int           fElemSelectorConvNumData = 0;          // total number of data i.e. lenght of fElemSelectorConvData
  double        fElemSelectorConvLogMinEkin = 0.0;
  double        fElemSelectorConvEILDelta = 0.0;         //
  int*          fElemSelectorConvStartIndexPerMat = nullptr; // [fNumMaterials]
  double*       fElemSelectorConvEgrid = nullptr;            // [fElemSelectorConvEgridSize]

  /** Element selector data for all materials */
  double*       fElemSelectorConvData = nullptr;             // [fElemSelectorConvNumData]
};

/**
  * Allocates and pre-initialises the G4HepEmGammaData structure.
  *
  * This method is invoked from the InitGammaData() function declared
  * in the G4HepEmGammaInit header file. The input argument address of the
  * G4HepEmGammaData structure pointer is the one stored in the G4HepEmData
  * member of the `master` G4HepEmRunManager and the initialisation should be
  * done by the master G4HepEmRunManager by invoking the InitGammaData() function
  * for \f$\gamma\f$ particles.
  *
  * @param theGammaData address of a G4HepEmGammaData structure pointer. At termination,
  *   the correspondig pointer will be set to a memory location with a freshly allocated
  *   G4HepEmGammaData structure with all its pointer members set to nullprt.
  *   If the input pointer was not null at input, the pointed memory, including all
  *   dynamic memory members, is freed before the new allocation.
  */
void AllocateGammaData (struct G4HepEmGammaData** theGammaData);

/**
 * Initializes a new @ref G4HepEmGammaData structure
 *
 * This function default constructs an instance of G4HepEmGammaData and returns
 * a pointer to the freshly constructed instance. It is the callees responsibility
 * to free the instance using @ref FreeGammaData.
 *
 * @return Pointer to instance of @ref G4HepEmGammaData
 */
G4HepEmGammaData* MakeGammaData();

/**
  * Frees a G4HepEmGammaData structure.
  *
  * This function deallocates all dynamically allocated memory stored in the
  * input argument related G4HepEmGammaData structure, deallocates the structure
  * itself and sets the input address to store a pointer to null. This makes the
  * corresponding input stucture cleared, freed and ready to be re-initialised.
  * The input argument is supposed to be the address of the corresponding pointer
  * member of the G4HepEmData member of the `master` G4HepEmRunManager.
  *
  * @param theGammaData memory address that stores pointer to a G4HepEmGammaData
  *  structure. The memory is freed and the input address will store a null pointer
  *  at termination.
  */
void FreeGammaData (struct G4HepEmGammaData** theGammaData);



#ifdef G4HepEm_CUDA_BUILD
  /**
    * Allocates memory for and copies the G4HepEmGammaData structure from the
    * host to the device.
    *
    * The input arguments are supposed to be the corresponding members of the
    * G4HepEmData, top level data structure, stored in the `master` G4HepEmRunManager.
    *
    * @param onHOST    pointer to the host side, already initialised G4HepEmGammaData structure.
    * @param onDEVICE  host side address of a pointer to a device side G4HepEmGammaData
    *   structure. The pointed device side memory is cleaned (if not null at input) and
    *   points to the device side memory at termination containing all the copied
    *   G4HepEmGammaData structure members.
    */
  void CopyGammaDataToDevice(struct G4HepEmGammaData* onHOST, struct G4HepEmGammaData** onDEVICE);

  /**
    * Frees all memory related to the device side G4HepEmGammaData structure referred
    * by the pointer stored on the host side input argument address.
    *
    * @param onDEVICE host side address of a G4HepEmGammaDataOnDevice structure located on the device side memory.
    *   The correspondig device memory will be freed and the input argument address will be set to null.
    */
  void FreeGammaDataOnDevice(struct G4HepEmGammaData** onDEVICE);
#endif // DG4HepEm_CUDA_BUILD

#endif // G4HepEmGammaData_HH
