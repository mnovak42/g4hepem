#include "PhysListG4EmTracking.hh"
#include "G4EmTrackingManager.hh"

#include "G4EmParameters.hh"

#include "G4Electron.hh"
#include "G4Gamma.hh"
#include "G4ParticleDefinition.hh"
#include "G4Positron.hh"

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

PhysListG4EmTracking::PhysListG4EmTracking(const G4String& name)
   :  G4VPhysicsConstructor(name)
{
  G4EmParameters* param = G4EmParameters::Instance();
  param->SetDefaults();

  param->SetMscRangeFactor(0.04);
  param->SetLossFluctuations(false);
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

PhysListG4EmTracking::~PhysListG4EmTracking()
{}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void PhysListG4EmTracking::ConstructProcess()
{
  // Register custom tracking manager for e-/e+ and gammas.
  auto* trackingManager = new G4EmTrackingManager;

  G4Electron::Definition()->SetTrackingManager(trackingManager);
  G4Positron::Definition()->SetTrackingManager(trackingManager);
  G4Gamma::Definition()->SetTrackingManager(trackingManager);
}
