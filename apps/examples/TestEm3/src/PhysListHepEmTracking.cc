#include "PhysListHepEmTracking.hh"

#include "G4HepEmTrackingManager.hh"
#include "G4HepEmConfig.hh"

#include "G4EmParameters.hh"

#include "G4Electron.hh"
#include "G4Gamma.hh"
#include "G4ParticleDefinition.hh"
#include "G4Positron.hh"

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

PhysListHepEmTracking::PhysListHepEmTracking(const G4String& name)
   :  G4VPhysicsConstructor(name)
{
  G4EmParameters* param = G4EmParameters::Instance();
  param->SetDefaults();

  param->SetMscRangeFactor(0.04);
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

PhysListHepEmTracking::~PhysListHepEmTracking()
{}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void PhysListHepEmTracking::ConstructProcess()
{
  // Register custom tracking manager for e-/e+ and gammas.
  auto* trackingManager = new G4HepEmTrackingManager;
  // Configuration of G4HepEm
  // Several paramaters can be configured per detector region. These are:
  //  MSC parameters, continuous energy loss step limit function parameters,
  //  MSC minimal/default step limit, Woodcock tracking of photons, energy loss
  //  fluctuation, multiple steps in the combined MSC with Transportation
  // Here we activate only one: Woodcock tracking in the calorimeter region (Woodcock_Region)
  G4HepEmConfig* config = trackingManager->GetConfig();
  config->SetWoodcockTrackingRegion("Woodcock_Region");

  G4Electron::Definition()->SetTrackingManager(trackingManager);
  G4Positron::Definition()->SetTrackingManager(trackingManager);
  G4Gamma::Definition()->SetTrackingManager(trackingManager);
}
