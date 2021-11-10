
#ifndef G4HepEmNoProcess_h
#define G4HepEmNoProcess_h 1

#include "G4VProcess.hh"

/**
 * @file    G4HepEmNoProcess.hh
 * @class   G4HepEmNoProcess
 * @author  M. Novak
 * @date    2021
 *
 * An empty G4VProcess with configurable name.
 *
 * This process should not be assigned to any particles since it's empty. It's
 * used only to provide infomation to the `Geant4` framework regarding the name
 * of the processes that determined the step. 
 */

class G4HepEmNoProcess : public G4VProcess {
  public:

    G4HepEmNoProcess(const G4String& name) : G4VProcess( name, fGeneral ) {};

   ~G4HepEmNoProcess() override {};

    // This process should not be set to any particle
    G4bool IsApplicable(const G4ParticleDefinition&) override { return false; }

    //  no operations in any GPIL or DoIt
    G4double PostStepGetPhysicalInteractionLength(
                             const G4Track&,
                             G4double,
                             G4ForceCondition*
                            ) override { return -1.0; };

    G4VParticleChange* PostStepDoIt(
                             const G4Track& ,
                             const G4Step&
                            ) override {return nullptr;};

    G4double AtRestGetPhysicalInteractionLength(
                             const G4Track& ,
                             G4ForceCondition*
                            ) override { return -1.0; };

    G4VParticleChange* AtRestDoIt(
                             const G4Track& ,
                             const G4Step&
                            ) override {return nullptr;};

    G4double AlongStepGetPhysicalInteractionLength(
                             const G4Track&,
                             G4double  ,
                             G4double  ,
                             G4double& ,
                             G4GPILSelection*
                            ) override { return -1.0; };

    G4VParticleChange* AlongStepDoIt(
                             const G4Track& ,
                             const G4Step&
                            ) override {return nullptr;};

    G4HepEmNoProcess(G4HepEmNoProcess&) = delete;
    G4HepEmNoProcess& operator=(const G4HepEmNoProcess& right) = delete;

};

#endif
