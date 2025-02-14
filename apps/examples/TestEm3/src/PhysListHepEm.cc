
#include "PhysListHepEm.hh"

// include the G4HepEmProcess from the G4HepEm lib.
#include "G4HepEmProcess.hh"

#include "G4ParticleDefinition.hh"
#include "G4ProcessManager.hh"
#include "G4PhysicsListHelper.hh"
#include "G4EmParameters.hh"
#include "G4BuilderType.hh"
#include "G4SystemOfUnits.hh"


PhysListHepEm::PhysListHepEm(const G4String& name)
: G4VPhysicsConstructor(name) {
  G4EmParameters* param = G4EmParameters::Instance();
  param->SetDefaults();

  param->SetMscRangeFactor(0.04);

  SetPhysicsType(bElectromagnetic);
}


PhysListHepEm::~PhysListHepEm() {}


void PhysListHepEm::ConstructProcess() {
  G4PhysicsListHelper* ph = G4PhysicsListHelper::GetPhysicsListHelper();

  // creae the only one G4HepEm process that will be assigned to e-/e+ and gamma
  G4HepEmProcess* hepEmProcess = new G4HepEmProcess();
  hepEmProcess->SetVerboseLevel(verboseLevel);

  // Add standard EM Processes
  //
  auto aParticleIterator = GetParticleIterator();
  aParticleIterator->reset();
  while( (*aParticleIterator)() ){
    G4ParticleDefinition* particle = aParticleIterator->value();
    G4String particleName = particle->GetParticleName();

    if (particleName == "gamma") {

      // Add G4HepEm process to gamma: includes Conversion, Compton and photoelectric effect.
      particle->GetProcessManager()->AddProcess(hepEmProcess, -1, -1, 1);

    } else if (particleName == "e-") {

      // Add G4HepEm process to e-: includes Ionisation, Bremsstrahlung, MSC for e-
     particle->GetProcessManager()->AddProcess(hepEmProcess, -1, -1, 1);

    } else if (particleName == "e+") {

      // Add G4HepEm process to e+: includes Ionisation, Bremsstrahlung, MSC and e+e-
      // annihilation into 2 gamma interactions for e+
      particle->GetProcessManager()->AddProcess(hepEmProcess, -1, -1, 1);

    }
  }

  if (G4Threading::IsMasterThread() && verboseLevel > 0) {
    G4EmParameters::Instance()->Dump();
  }

}
