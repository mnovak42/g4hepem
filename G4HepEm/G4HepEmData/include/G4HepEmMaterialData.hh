#ifndef G4HepEmMatrialData_HH
#define G4HepEmMatrialData_HH

/**
 * @file    G4HepEmMaterialData.hh
 * @struct  G4HepEmMaterialData
 * @author  M. Novak
 * @date    2020
 *
 * @brief All material related data used by ``G4HepEm``.
 *
 * All material related data, used by ``G4HepEm``, is stored in this simple data
 * structure. The G4HepEmMaterialData structure contains a collection of G4HepEmMatData
 * for each G4Material object that is used in the geometry. Such a G4HepEmMatData
 * stores a single material realted data required during the simulation.
 *
 * A single instance of this structure is created and stored in the `master`
 * G4HepEmRunManager when its InitializeGlobal() method is invoked by calling
 * the InitMaterialAndCoupleData() function declared in the G4HepEmMaterialInit
 * header file. This method extracts information form the already initialised
 * part of the Geant4 application (i.e. G4MaterialCutsCoupleTable) by reading all
 * G4MaterialCutsCouple objects used in the current geometry, translating their
 * G4Material objects into G4HepEmMatData strcutrues and store in this single
 * G4HepEmMaterialData sturcture.
 *
 * In case of ``CUDA`` build, the data can be easily copied to the device by either
 * using the CopyMaterialDataToGPU() function or the more general CopyG4HepEmDataToGPU()
 * function: the first will copy only the specified G4HepEmMaterialData structure
 * while the second will (deep copy) all the members (including the G4HepEmMaterialData
 * one) of the G4HepEmData, top level data structure member of the (master)
 * G4HepEmRunManager.
 *
 * @note  only the data members that are needed on the device side are copied to
 * the device memory. This means, that only the G4HepEmMaterialData::fMaterialData
 * collection of the G4HepEmMatData structures is copied to the device memory
 * together with the G4HepEmMaterialData::fNumMaterialData length of this collection.
 *
 */

/** Data that describes a single matrial in ``G4HepEm``. */
struct G4HepEmMatData {
  /** The corresponding G4Material object index.*/
  int       fG4MatIndex = -1;
  /** Number of elements this matrial is composed of (size of the arrays below). */
  int       fNumOfElement = 0;
  /** The list of element indices in G4HepEmElemData (their atomic number Z), this material is composed of.*/
  int*      fElementVect = nullptr; // [fNumOfElement]
  /** The list of number-of-atoms-per-unit-volume for each element this material is composed of.*/
  double*   fNumOfAtomsPerVolumeVect = nullptr; // [fNumOfElement]
  /** The mass density (\f$\rho\f$) of the material in Geant4 internal units. */
  double    fDensity = 0.0;
  /** Density correction factor (\f$C_{Mg}\rho_{e^-}\f$) used in the `dielectric suppression`
    * of bremsstrahlung photon emission (\f$C_{Mg}=4\pi r_0 \hbar c/(mc^2)\f$ is the `Migdal constant`
    * and \f$\rho_{e^-}\f$ is the electron density of the material). */
  double    fDensityCorFactor = 0.0;
  /** Electron density (\f$\rho_{e^-}\f$) of the material in Geant4 internal units.*/
  double    fElectronDensity = 0.0;
  /** Radiation length. */
  double    fRadiationLength = 0.0;
};

// Data for all materials used in the current geometry.
struct G4HepEmMaterialData {
  /** Number of Geant4 material objects (irrespectively if used or not).*/
  int       fNumG4Material = 0;
  /** Number of G4HepEmMatData structures in ``G4HepEm`` (only the used G4Material objects are translated).*/
  int       fNumMaterialData = 0;
  /** Array that translates a Geant4 G4Material object index to the correspondig G4HepEmMatData index in the collection below.*/
  int*      fG4MatIndexToHepEmMatIndex = nullptr; // [fNumG4Material]
  /** Collection of G4HepEmMatData structures for all materials used in the current geometry.*/
  struct G4HepEmMatData* fMaterialData = nullptr; // [fNumMaterialData]
};


/**
  * Allocates and pre-initialises the G4HepEmMaterialData structure.
  *
  * This method is invoked from the InitMaterialAndCoupleData() function declared
  * in the G4HepEmMaterialInit header file. The input argument address of the
  * G4HepEmMaterialData structure pointer is the one stored in the G4HepEmData
  * member of the `master` G4HepEmRunManager and the initialisation should be
  * done by the master G4HepEmRunManager.
  *
  * @param theMatData address of a G4HepEmMaterialData structure pointer. At termination,
  *   the correspondig pointer will be set to a memory location with a freshly allocated
  *   G4HepEmMaterialData structure. If the pointer was not null at input, the pointed
  *   memory is freed before the new allocation.
  * @param numG4Mat number of Geant4 material objects (irrespectively if used or not).
  *   It determines the maximum value of the G4Material object index.
  * @param numUsedG4Mat number of Geant4 material objects used in the current geometry.
  *   It determines the number of the G4HepEmMatData structures.
  */
void AllocateMaterialData(struct G4HepEmMaterialData** theMatData, int numG4Mat, int numUsedG4Mat);

/**
 * Initializes a new @ref G4HepEmMaterialData structure
 *
 * This function constructs and returns an instance of G4HepEmMaterialData to hold a given number of indices
 * to Geant4 materials objects and their corresponding G4HepEmMaterialData instance.
 * It is the callees responsibility to free the instance using @ref FreeMaterialData.
 *
  * @param[in] numG4Mat number of Geant4 material objects
  * @param[in] numUsedG4Mat number of Geant4 materials objects used in the current geometry.
  * @return Pointer to instance of @ref G4HepEmMaterialData
 */
G4HepEmMaterialData* MakeMaterialData(int numG4Mat, int numUsedG4Mat);

/**
  * Frees a G4HepEmMaterialData structure.
  *
  * This function deallocates all dynamically allocated memory stored in the
  * input argument related G4HepEmMaterialData structure, deallocates the structure
  * itself and sets the input address to store a pointer to null. This makes the
  * corresponding input stucture cleared, freed and ready to be re-initialised.
  * The input argument is supposed to be the address of the corresponding pointer
  * member of the G4HepEmData member of the `master` G4HepEmRunManager.
  *
  * @param theMatData memory address that stores pointer to a G4HepEmMaterialData
  *  structure. The memory is freed and the input address will store a null pointer
  *  at termination.
  */
void FreeMaterialData (struct G4HepEmMaterialData** theMatData);


#ifdef G4HepEm_CUDA_BUILD
  /**
    * Allocates memory for and copies the G4HepEmMaterialData structure from the
    * host to the device.
    *
    * Only the G4HepEmMaterialData::fMaterialData collection of the G4HepEmMatData
    * structures is copied to the device together with the G4HepEmMaterialData::fNumMaterialData
    * length of this collection.
    *
    * The input arguments are supposed to be the corresponding members of the
    * G4HepEmData, top level data structure, stored in the `master` G4HepEmRunManager.
    *
    * @param onHost    pointer to the host side, already initialised G4HepEmMaterialData structure.
    * @param onDevice  host side address of a G4HepEmMaterialData structure memory pointer. The pointed
    *   memory is cleaned (if not null at input) and points to the device side memory at termination
    *   that stores the copied G4HepEmMaterialData structure.
    */
  void CopyMaterialDataToGPU(struct G4HepEmMaterialData* onHost, struct G4HepEmMaterialData** onDevice);

  /**
    * Frees all memory related to the device side G4HepEmMaterialData structure referred
    * by the pointer stored on the host side input argument address.
    *
    * @param onDevice host side address of a G4HepEmMaterialData structure located on the device side memory.
    *   The correspondig device memory will be freed and the input argument address will be set to null.
    */
  void FreeMaterialDataOnGPU(struct G4HepEmMaterialData** onDevice);
#endif // DG4HepEm_CUDA_BUILD


#endif // G4HepEmMatrialData_HH
