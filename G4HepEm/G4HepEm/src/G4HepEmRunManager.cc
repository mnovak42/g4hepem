

#include "G4HepEmRunManager.hh"


#include "G4HepEmData.hh"
#include "G4HepEmParameters.hh"
#include "G4HepEmTLData.hh"

#include "G4HepEmParametersInit.hh"
#include "G4HepEmMaterialInit.hh"

#include "G4HepEmElectronInit.hh"
#include "G4HepEmElectronManager.hh"

#include "G4HepEmGammaInit.hh"
#include "G4HepEmGammaManager.hh"

#include "G4HepEmRandomEngine.hh"

#include <iostream>


G4HepEmRunManager* G4HepEmRunManager::gTheG4HepEmRunManagerMaster = nullptr;

G4HepEmRunManager::G4HepEmRunManager(bool ismaster) {
  if (ismaster && !gTheG4HepEmRunManagerMaster) {
    G4HepEmRunManager::gTheG4HepEmRunManagerMaster = this;
    fIsMaster                    = true;
    fIsInitialisedForParticle[0] = false;
    fIsInitialisedForParticle[1] = false;
    fIsInitialisedForParticle[2] = false;
    // all these above initialisation will be done and filled with data at InitializeGlobal()
  } else {
    fIsMaster = false;
  }
  fExternalParameters            = false;

  fTheG4HepEmParameters          = nullptr;
  fTheG4HepEmData                = nullptr;
  fTheG4HepEmTLData              = nullptr;

  fVerbose = 1;
}


G4HepEmRunManager::~G4HepEmRunManager() {
  Clear();
}


G4HepEmRunManager* G4HepEmRunManager::GetMasterRunManager() {
  return gTheG4HepEmRunManagerMaster;
}


// this might be called more than one: as many times as the process is assigned
// to a particle but no way to ensure that is also called at re-init
void G4HepEmRunManager::InitializeGlobal(G4HepEmParameters* hepEmPars) {
//  std::cout << "  ---- InitializeGlobal() is called for fIsMaster = " << fIsMaster << " "
  if (fIsMaster) {// && !fIsMasterGlobalInitialized) {
    //
    // create HepEmParameters structure shared by all workers
//    fTheG4HepEmParameters = new G4HepEmParameters();
    // create the top level HepEmData structure shared by all workers and init
    fTheG4HepEmData       = new G4HepEmData;
    // set all ptr members of the HepEmData structure to null
    InitG4HepEmData(fTheG4HepEmData);
    //
    // === Use the G4HepEmParamatersInit::InitHepEmParameters method for the
    //     initialization of all configuartion parameters by extracting information
    //     from the G4EmParameters.
    if (hepEmPars != nullptr) {
      fTheG4HepEmParameters = hepEmPars;
      fExternalParameters   = true;
    } else {
      fTheG4HepEmParameters = new G4HepEmParameters();
      InitHepEmParameters(fTheG4HepEmParameters);
    }

    // === Use the G4HepEmMaterialInit::InitMaterialAndCoupleData method for the
    //     initialization of all material and secondary production threshold related
    //     data.
    //
    // It will create and init the G4HepEmMatCutData, G4HepEmMaterialData members
    // of the G4HepEmMatData (fTheG4HepEmData) structrue.
    // - translates all G4MaterialCutsCouple, used in the current geometry, to a
    //   G4HepEmMatCutData structure element used by G4HepEm
    // - generates a G4HepEmMaterialData structure that stores material information
    //   for all unique materials, used in the current geometry
    // - builds the G4HepEmElementData structure
    InitMaterialAndCoupleData(fTheG4HepEmData, fTheG4HepEmParameters);
  }
}

void G4HepEmRunManager::Initialize(G4HepEmRandomEngine* theRNGEngine, int hepEmParticleIndx, G4HepEmParameters* hepEmPars) {
  if (fIsMaster) {
    //
    // Build the global data structures (material-cuts, material, element data and
    // configuration paramaters) because this is the first or a new run i.e. the
    // call to `G4VProcess::BuildPhysicsTable()` was triggered by `physics-has-modified`.
    if (!fTheG4HepEmParameters || fIsInitialisedForParticle[hepEmParticleIndx]) {
      // clear all previously created data structures and create the new global data
      Clear();
      if (fVerbose > 1) std::cout << " === G4HepEm global init ... " << std::endl;
      InitializeGlobal(hepEmPars);
    }
    //
    // Build tables: e-loss tables for e/e+, macroscopic cross section and element
    //   selectors that are used/shared by all workers at run time as read-only.
    if (fVerbose > 1) {
      std::cout << " === G4HepEm init for particle index = " << hepEmParticleIndx << " ..."<< std::endl;
    }
    switch (hepEmParticleIndx) {
      // === e- : use the G4HepEmElementInit::InitElectronData() method for e- initialization.
      case 0 : InitElectronData(fTheG4HepEmData, fTheG4HepEmParameters, true, fVerbose);
               fIsInitialisedForParticle[0] = true;
               break;
      // === e+ : use the G4HepEmElementInit::InitElectronData() method for e+ initialization.
      case 1 : InitElectronData(fTheG4HepEmData, fTheG4HepEmParameters, false, fVerbose);
               fIsInitialisedForParticle[1] = true;
               //fTheG4HepEmPositronManager = new G4HepEmElectronManager;
               break;
      // === Gamma: use the G4HepEmGammaInit::InitGammaData() method for gamma initialization.
      case 2 : InitGammaData(fTheG4HepEmData, fTheG4HepEmParameters, fVerbose);
               fIsInitialisedForParticle[2] = true;
               break;
      default: std::cerr << " **** ERROR in G4HepEmRunManager::Initialize: unknown particle " << std::endl;
               exit(-1);
    }
    if  (!fTheG4HepEmTLData) {
      fTheG4HepEmTLData = new G4HepEmTLData;
      fTheG4HepEmTLData->SetRandomEngine(theRNGEngine);
    }
  } else {
    //
    // Worker: 1. copy the pointers to members that are shared by all workers
    //            from the master-RM if it has not been done yet.
    //if (!fTheG4HepEmParameters) {
      fTheG4HepEmParameters = G4HepEmRunManager::GetMasterRunManager()->GetHepEmParameters();
      fTheG4HepEmData       = G4HepEmRunManager::GetMasterRunManager()->GetHepEmData();
    //}
    // Worker: 2. create a worker local data structure for this worker and set
    //            its RNG engine part if it has not been done yet.
    if  (!fTheG4HepEmTLData) {
      fTheG4HepEmTLData = new G4HepEmTLData;
      fTheG4HepEmTLData->SetRandomEngine(theRNGEngine);
    }
  }
}



void G4HepEmRunManager::Clear() {
  if (fIsMaster) {
    if (fTheG4HepEmParameters && !fExternalParameters) {
      delete fTheG4HepEmParameters;
      fTheG4HepEmParameters = nullptr;
    }
    if (fTheG4HepEmData) {
      FreeG4HepEmData(fTheG4HepEmData);
      delete fTheG4HepEmData;
      fTheG4HepEmData = nullptr;
    }
    fIsInitialisedForParticle[0] = false;
    fIsInitialisedForParticle[1] = false;
    fIsInitialisedForParticle[2] = false;
  } else {
    // set shared ptr already cleaned in master
    fTheG4HepEmParameters      = nullptr;
    fTheG4HepEmData            = nullptr;
    // clean local objects and set ptr
    if (fTheG4HepEmTLData)
      delete fTheG4HepEmTLData;
    fTheG4HepEmTLData     = nullptr;
  }
}
