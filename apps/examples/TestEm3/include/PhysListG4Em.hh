
#ifndef PhysListG4Em_h
#define PhysListG4Em_h 1

#include "G4VPhysicsConstructor.hh"
#include "globals.hh"


class PhysListG4Em : public G4VPhysicsConstructor {
  public:
     PhysListG4Em (const G4String& name = "G4EM-physics-list (like G4HepEm)");
    ~PhysListG4Em();

  public:
    // This method is dummy for physics: particles are constructed in PhysicsList
    void ConstructParticle() override {};

    // This method will be invoked in the Construct() method.
    // each physics process will be instantiated and
    // registered to the process manager of each particle type
    void ConstructProcess() override;
};

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

#endif
