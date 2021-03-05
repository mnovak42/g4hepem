#ifndef G4HepEmData_HH
#define G4HepEmData_HH

struct G4HepEmMatCutData;
struct G4HepEmMaterialData;
struct G4HepEmElementData;

struct G4HepEmElectronData;

struct G4HepEmSBTableData;

struct G4HepEmGammaData;

/**
 * @file    G4HepEmData.hh
 * @struct  G4HepEmData
 * @author  M. Novak
 * @date    2020
 *
 * The top level, global data structure i.e. collection of global data
 * structures used by all physics interactions covered by `G4HepEm`.
 *
 * There supposed to be a single instance of this data structure constructed and
 * stored by the master `G4HepEmRunManager`. That single instance is
 * constructed when calling the `G4HepEmRunManager::InitializeGlobal()` method
 * wehn invoking the master `G4HepEmRunManager::Initialize()` method.
 * Worker `G4HepEmRunManager`-s will have their pointer set to this master
 * run-manager `G4HepEmData` member object.
 *
 * Members of this data structure, represented by their pointers, are created
 * individualy by invoking the dedicated intialisation methods one-by-one.
 *
 * In case of CUDA build, the members, suffixed by `_gpu`, points do device memory
 * locations where the corresponding data structures are located (after copying
 * them). All the corresponding members can be (deep) copied from the host to
 * device by calling the `CopyG4HepEmDataToGPU()` function. This will invoke
 * the dedicated copy methods provided by the individual data structures.
 *
 * The dynamically allocated memory, represented by all the members of this
 * collection (including device side memeory as well in case of CUDA build), can
 * be cleaned by calling the `FreeG4HepEmData()` function. This is done, in the
 * `G4HepEmRunManager::Clear()` method.
*/

struct G4HepEmData {
  /** Global G4HepEmMatCutData i.e. material and scondary production threshold related data.*/
  struct G4HepEmMatCutData*            fTheMatCutData       = nullptr;
  /** Global material and scondary production threshold related data.*/
  struct G4HepEmMaterialData*          fTheMaterialData     = nullptr;
  struct G4HepEmElementData*           fTheElementData      = nullptr;

  struct G4HepEmElectronData*          fTheElectronData     = nullptr;
  struct G4HepEmElectronData*          fThePositronData     = nullptr;

  struct G4HepEmSBTableData*           fTheSBTableData      = nullptr;


  struct G4HepEmGammaData*             fTheGammaData        = nullptr;


#ifdef G4HepEm_CUDA_BUILD
  struct G4HepEmMatCutData*            fTheMatCutData_gpu   = nullptr;
  struct G4HepEmMaterialData*          fTheMaterialData_gpu = nullptr;
  struct G4HepEmElementData*           fTheElementData_gpu  = nullptr;

  struct G4HepEmElectronData*          fTheElectronData_gpu = nullptr;
  struct G4HepEmElectronData*          fThePositronData_gpu = nullptr;

  struct G4HepEmSBTableData*           fTheSBTableData_gpu  = nullptr;

  struct G4HepEmGammaData*             fTheGammaData_gpu    = nullptr;
#endif  // G4HepEm_CUDA_BUILD

};

/** Function that ...*/
void InitG4HepEmData (struct G4HepEmData* theHepEmData);

/** Function that ...*/
void FreeG4HepEmData (struct G4HepEmData* theHepEmData);


#ifdef G4HepEm_CUDA_BUILD
  /** Function that ...*/
  void CopyG4HepEmDataToGPU(struct G4HepEmData* onCPU);

  /** Function that ...*/
  void FreeG4HepEmDataOnGPU(struct G4HepEmData* onCPU);
#endif  // G4HepEm_CUDA_BUILD


#endif  // G4HepEmData_HH
