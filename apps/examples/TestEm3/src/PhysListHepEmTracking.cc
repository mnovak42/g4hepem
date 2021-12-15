#include "PhysListHepEmTracking.hh"
#include "HepEmTrackingManager.hh"

#include "G4Electron.hh"
#include "G4Gamma.hh"
#include "G4ParticleDefinition.hh"
#include "G4Positron.hh"

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

PhysListHepEmTracking::PhysListHepEmTracking(const G4String& name)
   :  G4VPhysicsConstructor(name)
{
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

PhysListHepEmTracking::~PhysListHepEmTracking()
{}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void PhysListHepEmTracking::ConstructProcess()
{
  // Register custom tracking manager for e-/e+ and gammas.
  auto* trackingManager = new HepEmTrackingManager;

  G4Electron::Definition()->SetTrackingManager(trackingManager);
  G4Positron::Definition()->SetTrackingManager(trackingManager);
  G4Gamma::Definition()->SetTrackingManager(trackingManager);
}
