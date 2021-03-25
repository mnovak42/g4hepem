#ifndef G4HepEmMatCutData_HH
#define G4HepEmMatCutData_HH

/**
 * @file    G4HepEmMatCutData.hh
 * @struct  G4HepEmMatCutData
 * @author  M. Novak
 * @date    2020
 *
 * @brief All material - cuts couple related data used by ``G4HepEm``.
 *
 * All material - cuts (secondary production threshold) couple related data,
 * used by ``G4HepEm``, is stored in this simple data structure. The
 * G4HepEmMatCutData structure contains a collection of G4HepEmMCCData for each
 * G4MaterialCutsCouple object that is used in the geometry. Such a G4HepEmMCCData
 * stores a single material-cuts couple realted data required during the simulation.
 *
 * A single instance of this structure is created and stored in the `master`
 * G4HepEmRunManager when its InitializeGlobal() method is invoked by calling
 * the InitMaterialAndCoupleData() function declared in the G4HepEmMaterialInit
 * header file. This method extracts information form the already initialised
 * part of the Geant4 application (i.e. G4MaterialCutsCoupleTable) by reading all
 * G4MaterialCutsCouple objects used in the current geometry, translating them
 * into G4HepEmMCCData strcutrues and store in this single G4HepEmMatCutData sturcture.
 *
 * In case of ``CUDA`` build, the data can be easily copied to the device by either
 * using the CopyMatCutDataToGPU() function or the more general CopyG4HepEmDataToGPU()
 * function: the first will copy only the specified G4HepEmMatCutData structure
 * while the second will (deep copy) all the members (including the G4HepEmMatCutData
 * one) of the G4HepEmData, top level data structure member of the (master)
 * G4HepEmRunManager.
 *
 * @note  only the data members that are needed on the device side are copied to
 * the device memory. This means, that only the G4HepEmMatCutData::fMatCutData
 * collection of the G4HepEmMCCData structures is copied to the device memory
 * together with the G4HepEmMatCutData::fNumMatCutData length of this collection.
 *
 */

/** Data that describes a single matrial-cuts couple in ``G4HepEm``. */
struct G4HepEmMCCData {
  /** Secondary \f$e^-\f$ production threshold energy [MeV]. */
  double  fSecElProdCutE = 0.0;
  /** Secondary \f$\gamma\f$ production threshold energy [MeV]. */
  double  fSecGamProdCutE = 0.0;
  /** Logarithm of the above secondary \f$\gamma\f$ production threshold. */
  double  fLogSecGamCutE = 0.0;
  /** Index of its material realted data: index of its G4HepEmMatData in the G4HepEmMaterialData. */
  int     fHepEmMatIndex = -1;
  /** Index of the corresponding G4MaterialCutsCouple object.*/
  int     fG4MatCutIndex = -1;
};

// Data for all matrial cuts couple that are used by G4HepEm.
struct G4HepEmMatCutData {
  /** Number of G4MaterialCutsCouple objects in ``Geant4`` (irrespectively if used or not).  */
  int      fNumG4MatCuts = 0;
  /** Number of G4HepEmMCCData structure in ``G4HepEm`` (only the used G4MaterialCutsCouple objects are translated). */
  int      fNumMatCutData = 0;
  /** Array that translates a Geant4 G4MaterialCutsCouple object index to the correspondig G4HepEmMCCData index in the collection below.*/
  int*     fG4MCIndexToHepEmMCIndex = nullptr;  // [fNumG4MatCuts]
  /** Collection of G4HepEmMCCData structures for all material-cuts couples used in the current geometry.*/
  struct G4HepEmMCCData* fMatCutData = nullptr; // [fNumMatCutData]
};

/**
  * Allocates and pre-initialises the G4HepEmMatCutData structure.
  *
  * This method is invoked from the InitMaterialAndCoupleData() function declared
  * in the G4HepEmMaterialInit header file. The input argument address of the
  * G4HepEmMatCutData structure pointer is the one stored in the G4HepEmData
  * member of the `master` G4HepEmRunManager and the initialisation should be
  * done by the master G4HepEmRunManager.
  *
  * @param theMatCutData address of a G4HepEmMatCutData structure pointer. At termination,
  *   the correspondig pointer will be set to a memory location with a freshly allocated
  *   G4HepEmMatCutData structure. If the pointer is not null at input, the pointed
  *   memory is freed before the new allocation.
  * @param[in] numG4MatCuts number of Geant4 material-cuts couple objects (irrespectively if used or not).
  *   It determines the maximum value of the G4MaterialCutsCouple object index.
  * @param[in] numUsedG4MatCuts number of Geant4 material-cuts couple objects used in the current geometry.
  *   It determines the number of the G4HepEmMCCData structures.
  */
void AllocateMatCutData(struct G4HepEmMatCutData** theMatCutData, int numG4MatCuts, int numUsedG4MatCuts);


/**
 * Initializes a new @ref G4HepEmMatCutData structure
 *
 * This function constructs and returns an instance of G4HepEmMatCutData to hold a given number of indices
 * to Geant4 material-cuts couple objects and their corresponding G4HepEmMCCData instance.
 * It is the callees responsibility to free the instance using @ref FreeMatCutData.
 *
  * @param[in] numG4MatCuts number of Geant4 material-cuts couple objects
  * @param[in] numUsedG4MatCuts number of Geant4 material-cuts couple objects used in the current geometry.
  * @return Pointer to instance of @ref G4HepEmMatCutData
 */
G4HepEmMatCutData* MakeMatCutData(int numG4MatCuts, int numUsedG4MatCuts);

/**
  * Frees a G4HepEmMatCutData structure.
  *
  * This function deallocates all dynamically allocated memory stored in the
  * input argument related G4HepEmMatCutData structure, deallocates the structure
  * itself and sets the input address to store a pointer to null. This makes the
  * corresponding input stucture cleared, freed and ready to be re-initialised.
  * The input argument is supposed to be the address of the corresponding pointer
  * member of the G4HepEmData member of the `master` G4HepEmRunManager.
  *
  * @param theMatCutData memory address that stores pointer to a G4HepEmMatCutData
  *  structure. The memory is freed and the input address will store a null pointer
  *  at termination.
  */
void FreeMatCutData (struct G4HepEmMatCutData** theMatCutData);


#ifdef G4HepEm_CUDA_BUILD
  /**
    * Allocates memory for and copies the G4HepEmMatCutData structure from the host
    * to the device.
    *
    * Only the G4HepEmMatCutData::fMatCutData collection of the G4HepEmMCCData
    * structures is copied to the device together with the G4HepEmMatCutData::fNumMatCutData
    * length of this collection.
    *
    * The input arguments are supposed to be the corresponding members of the
    * G4HepEmData, top level data structure, stored in the `master` G4HepEmRunManager.
    *
    * @param onHost    pointer to the host side, already initialised G4HepEmMatCutData structure.
    * @param onDevice  host side address of a G4HepEmMatCutData structure memory pointer. The pointed
    *   memory is cleaned (if not null at input) and points to the device side memory at termination
    *   that stores the copied G4HepEmMatCutData structure.
    */
  void CopyMatCutDataToGPU(struct G4HepEmMatCutData* onHost, struct G4HepEmMatCutData** onDevice);

  /**
    * Frees all memory related to the device side G4HepEmMatCutData structure referred
    * by the pointer stored on the host side input argument address.
    *
    * @param onDevice host side address of a G4HepEmMatCutData structure located on the device side memory.
    *   The correspondig device memory will be freed and the input argument address will be set to null.
    */
  void FreeMatCutDataOnGPU(struct G4HepEmMatCutData** onDevice);
#endif // DG4HepEm_CUDA_BUILD

#endif // G4HepEmMatCutData_HH
