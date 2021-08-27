#ifndef PhysListG4EmTracking_h
#define PhysListG4EmTracking_h 1

#include "G4VPhysicsConstructor.hh"
#include "globals.hh"

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

class PhysListG4EmTracking : public G4VPhysicsConstructor
{
  public: 
     PhysListG4EmTracking(const G4String& name = "G4EmTracking");
    ~PhysListG4EmTracking();

  public: 
    void ConstructParticle() override {}

    void ConstructProcess() override;
};

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

#endif
