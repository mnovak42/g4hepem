#ifndef G4HepEmElementData_HH
#define G4HepEmElementData_HH

/**
 * @file    G4HepEmElementData.hh
 * @struct  G4HepEmElementData
 * @author  M. Novak
 * @date    2020
 *
 * @brief All element related data used by ``G4HepEm``.
 *
 * All element related data, used by ``G4HepEm``, is stored in this simple data
 * structure. The G4HepEmElementData structure contains a collection of G4HepEmElemData
 * for each G4Element object that is used in the geometry. Such a G4HepEmElemData
 * stores a single element realted data required during the simulation.
 *
 * A single instance of this structure is created and stored in the `master`
 * G4HepEmRunManager when its InitializeGlobal() method is invoked by calling
 * the InitMaterialAndCoupleData() function declared in the G4HepEmMaterialInit
 * header file. This method extracts information form the already initialised
 * part of the Geant4 application (i.e. G4MaterialCutsCoupleTable) by reading all
 * G4MaterialCutsCouple objects used in the current geometry, translating their
 * G4Element objects into G4HepEmElemData strcutrues and stores in this single
 * G4HepEmElementData sturcture.
 *
 * In case of ``CUDA`` build, the data can be easily copied to the device by either
 * using the CopyElementDataToGPU() function or the more general CopyG4HepEmDataToGPU()
 * function: the first will copy only the specified G4HepEmElementData structure
 * while the second will (deep copy) all the members (including the G4HepEmElementData
 * one) of the G4HepEmData, top level data structure member of the (master)
 * G4HepEmRunManager.
 *
 */

/** Data that describes a single element in ``G4HepEm``. */
struct G4HepEmElemData {

  /** The atomic number (Z) of the element. */
  double  fZet;

  /** \f$Z^{1/3}\f$ */
  double  fZet13;

  /** \f$Z^{2/3}\f$ */
  double  fZet23;

  /** Coulomb correction \f$ f_C \f$ */
  double  fCoulomb;

  /** \f$ \ln(Z) \f$  */
  double  fLogZ;

  /** \f$ F_{\text{el}}-f_c+F_{\text{inel}}/Z \f$  */
  double  fZFactor1;

  /** \f$ \exp \left[ \frac{42.038-F_{\text{low}}}{8.29} \right] -0.958 \f$ with \f$ Z_{\text{low}} = \frac{8}{3}\log(Z) \f$ */
  double  fDeltaMaxLow;

  /** \f$ \exp \left[ \frac{42.038-F_{\text{high}}}{8.29} \right] -0.958 \f$ with \f$ F_{\text{high}} = 8[\log(Z)/3 + f_C] \f$ */
  double  fDeltaMaxHigh;

  /** LPM variable \f$ 1/ln [ \sqrt{2}s1 ] \f$ */
  double  fILVarS1;

  /** LPM variable \f$ 1/ln[s1] \f$ */
  double  fILVarS1Cond;

};

// Data for all elements that are used by G4HepEm.
struct G4HepEmElementData {
  /** Maximum capacity of the below G4HepEmElemData structure container (max Z that can be stored).*/
  int     fMaxZet;
  /** Collection of G4HepEmElemData structures indexed by the atomic number Z. */
  struct G4HepEmElemData* fElementData;  // [fMaxZet]
};

/**
  * Allocates and pre-initialises the G4HepEmMaterialData structure.
  *
  * This method is invoked from the InitMaterialAndCoupleData() function declared
  * in the G4HepEmMaterialInit header file. The input argument address of the
  * G4HepEmElementData structure pointer is the one stored in the G4HepEmData
  * member of the `master` G4HepEmRunManager and the initialisation should be
  * done by the master G4HepEmRunManager.
  *
  * @param theElementData address of a G4HepEmElementData structure pointer. At termination,
  *   the correspondig pointer will be set to a memory location with a freshly allocated
  *   G4HepEmElementData structure. If the pointer was not null at input, the pointed
  *   memory is freed before the new allocation.
  */
void AllocateElementData(struct G4HepEmElementData** theElementData);

/**
  * Frees a G4HepEmElementData structure.
  *
  * This function deallocates all dynamically allocated memory stored in the
  * input argument related G4HepEmElementData structure, deallocates the structure
  * itself and sets the input address to store a pointer to null. This makes the
  * corresponding input stucture cleared, freed and ready to be re-initialised.
  * The input argument is supposed to be the address of the corresponding pointer
  * member of the G4HepEmData member of the `master` G4HepEmRunManager.
  *
  * @param theElementData memory address that stores pointer to a G4HepEmElementData
  *  structure. The memory is freed and the input address will store a null pointer
  *  at termination.
  */
void FreeElementData (struct G4HepEmElementData** theElementData);


#ifdef G4HepEm_CUDA_BUILD
/**
  * Allocates memory for and copies the G4HepEmElementData structure from the
  * host to the device.
  *
  * The input arguments are supposed to be the corresponding members of the
  * G4HepEmData, top level data structure, stored in the `master` G4HepEmRunManager.
  *
  * @param onHost    pointer to the host side, already initialised G4HepEmElementData structure.
  * @param onDevice  host side address of a G4HepEmElementData structure memory pointer. The pointed
  *   memory is cleaned (if not null at input) and points to the device side memory at termination
  *   that stores the copied G4HepEmElementData structure.
  */
  void CopyElementDataToGPU(struct G4HepEmElementData* onHost, struct G4HepEmElementData** onDevice);

  /**
    * Frees all memory related to the device side G4HepEmElementData structure referred
    * by the pointer stored on the host side input argument address.
    *
    * @param onDevice host side address of a G4HepEmElementData structure located on the device side memory.
    *   The correspondig device memory will be freed and the input argument address will be set to null.
    */
  void FreeElementDataOnGPU(struct G4HepEmElementData** onDevice);
#endif // DG4HepEm_CUDA_BUILD


#endif // G4HepEmElementData_HH
