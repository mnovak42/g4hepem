

#include "G4HepEmRunManager.hh"


#include "G4HepEmData.hh"
#include "G4HepEmParameters.hh"
#include "G4HepEmTLData.hh"

#include "G4HepEmParametersInit.hh"
#include "G4HepEmMaterialInit.hh"

#include "G4HepEmElectronInit.hh"
#include "G4HepEmElectronManager.hh"

// g4 includes
#include "Randomize.hh"
#include "CLHEP/Random/RandomEngine.h"


#include <iostream>


G4HepEmRunManager* G4HepEmRunManager::gTheG4HepEmRunManagerMaster = nullptr;
std::vector<G4HepEmRunManager*> G4HepEmRunManager::gTheG4HepEmRunManagers;

G4HepEmRunManager::G4HepEmRunManager(bool ismaster) {
  if (ismaster && !gTheG4HepEmRunManagerMaster) {
    G4HepEmRunManager::gTheG4HepEmRunManagerMaster = this;
    fIsMaster                    = true;
    fIsInitialisedForParticle[0] = false;
    fIsInitialisedForParticle[1] = false;
    fIsInitialisedForParticle[2] = false;
    // all these above initialisation will be done and filled with data at InitializeGlobal()    
    std::cout << " ------- Master is created ismaster = " << ismaster << std::endl;
  } else {        
    fIsMaster = false;
    std::cout << " ------- Worker is created ismaster = " << ismaster << std::endl;
  }
  fTheG4HepEmParameters          = nullptr;
  fTheG4HepEmData                = nullptr;
  fTheG4HepEmTLData              = nullptr;
  //
  fTheG4HepEmElectronManager     = nullptr;
  //
  G4HepEmRunManager::gTheG4HepEmRunManagers.push_back(this);
}


G4HepEmRunManager::~G4HepEmRunManager() {
  ClearAll();
}


G4HepEmRunManager* G4HepEmRunManager::GetMasterRunManager() {
  return gTheG4HepEmRunManagerMaster;
}


// this might be called more than one: as many times as the process is assigned 
// to a particle but no way to ensure that is also called at re-init
void G4HepEmRunManager::InitializeGlobal() {
//  std::cout << "  ---- InitializeGlobal() is called for fIsMaster = " << fIsMaster << " "
  if (fIsMaster) {// && !fIsMasterGlobalInitialized) {
    //
    std::cout << "  **** G4HepEmRunManager::InitializeGlobal() is called for Master" << std::endl;
    // create HepEmParameters structure shared by all workers
    fTheG4HepEmParameters = new G4HepEmParameters();
    // create the top level HepEmData structure shared by all workers and init
    fTheG4HepEmData       = new G4HepEmData;
    // set all ptr members of the HepEmData structure to null
    InitG4HepEmData(fTheG4HepEmData);
    //
    // === Use the G4HepEmParamatersInit::InitHepEmParameters method for the 
    //     initialization of all configuartion parameters by extracting information 
    //     from the G4EmParameters.
    InitHepEmParameters(fTheG4HepEmParameters);

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

void G4HepEmRunManager::Initialize(CLHEP::HepRandomEngine* theRNGEngine, int hepEmParticleIndx) {
  if (fIsMaster) { 
    //
    // Build the global data structures (material-cuts, material, element data and 
    // configuration paramaters) because this is the first or a new run i.e. the
    // call to `G4VProcess::BuildPhysicsTable()` was triggered by `physics-has-modified`.
    if (!fTheG4HepEmParameters || fIsInitialisedForParticle[hepEmParticleIndx]) {
      // clear all previously created data structures and create the new global data
      ClearAll();
      std::cout << "  ******** Cleared "<< std::endl;
      InitializeGlobal();
    }
    //
    std::cout << "  ==== G4HepEmRunManager::Initialize() is called for Master" << std::endl;
    //
    // Build tables: e-loss tables for e/e+, macroscopic cross section and element
    //   selectors that are used/shared by all workers at run time as read-only.
    switch (hepEmParticleIndx) {      
      // === e- : use the G4HepEmElementInit::InitElectronData() method for e- initialization.
      case 0 : InitElectronData(fTheG4HepEmData, fTheG4HepEmParameters, true);
               fTheG4HepEmElectronManager = new G4HepEmElectronManager;
               fIsInitialisedForParticle[0] = true;
               break;
      // === e+ : use the G4HepEmElementInit::InitElectronData() method for e+ initialization.
      case 1 : InitElectronData(fTheG4HepEmData, fTheG4HepEmParameters, false);
               fIsInitialisedForParticle[1] = true;
               //fTheG4HepEmPositronManager = new G4HepEmElectronManager;
               break;
      // === Gamma:                
      case 2 : // InitGammaData();                 // gamma
               fIsInitialisedForParticle[2] = true;
               break;
      default: std::cerr << " **** ERROR in G4HepEmRunManager::Initialize: unknown particle " << std::endl;
               exit(-1);                
    }
    if  (!fTheG4HepEmTLData) {
      fTheG4HepEmTLData = new G4HepEmTLData;
      fTheG4HepEmTLData->SetRandomEngine(theRNGEngine); 
    }
    std::cout << " master = " << fTheG4HepEmData->fTheMatCutData << std::endl;        
  } else {
    //
    std::cout << "  ==== G4HepEmRunManager::Initialize() is called for Worker" << std::endl;  
    //
    // Worker: 1. copy the pointers to members that are shared by all workers 
    //            from the master-RM if it has not been done yet.
    if (!fTheG4HepEmParameters) { 
      std::cout << " --- setting it for workers " << std::endl;
      fTheG4HepEmParameters = G4HepEmRunManager::GetMasterRunManager()->GetHepEmParameters();
      fTheG4HepEmData       = G4HepEmRunManager::GetMasterRunManager()->GetHepEmData(); 
      fTheG4HepEmElectronManager = G4HepEmRunManager::GetMasterRunManager()->GetTheElectronManager(); 
    }
    std::cout << " worker = " << fTheG4HepEmData->fTheMatCutData << std::endl;
    std::cout << " w-master = " << G4HepEmRunManager::GetMasterRunManager()->GetHepEmData()->fTheMatCutData << std::endl;

    // Worker: 2. create a worker local data structure for this worker and set 
    //            its RNG engine part if it has not been done yet.
    if  (!fTheG4HepEmTLData) {
      fTheG4HepEmTLData = new G4HepEmTLData;
      fTheG4HepEmTLData->SetRandomEngine(theRNGEngine); 
    }
  }
  std::cout << " ====== Completed init call ==== " << std::endl;
}



void G4HepEmRunManager::Clear() {
  if (fIsMaster) {
    if (fTheG4HepEmParameters) {
      delete fTheG4HepEmParameters;
      fTheG4HepEmParameters = nullptr;
    }
    if (fTheG4HepEmData) {
      FreeG4HepEmData(fTheG4HepEmData);
      delete fTheG4HepEmData;
      fTheG4HepEmData = nullptr;
    }
    if (fTheG4HepEmElectronManager)
      delete fTheG4HepEmElectronManager;
    fTheG4HepEmElectronManager = nullptr;
    fIsInitialisedForParticle[0] = false;
    fIsInitialisedForParticle[1] = false;
    fIsInitialisedForParticle[2] = false;
    // should call all worker-rm->Clear()
  } else {
    // set shared ptr already cleaned in master
    fTheG4HepEmParameters      = nullptr;
    fTheG4HepEmData            = nullptr;
    fTheG4HepEmElectronManager = nullptr;
    // clean local objects and set ptr
    if (fTheG4HepEmTLData)
      delete fTheG4HepEmTLData;
    fTheG4HepEmTLData     = nullptr;
  } 
}

void G4HepEmRunManager::ClearAll() {
  for (std::size_t i=0; i<G4HepEmRunManager::gTheG4HepEmRunManagers.size(); ++i) {
    G4HepEmRunManager::gTheG4HepEmRunManagers[i]->Clear();
  }
}

