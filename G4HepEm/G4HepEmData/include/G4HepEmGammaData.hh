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

  // the macroscopic cross sections for all materials and for [conversion,compton]
  // at each material
  double*       fConvCompMacXsecData = nullptr;   // [#materials*2*(fConvEnergyGridSize+fCompEnergyGridSize)]

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
