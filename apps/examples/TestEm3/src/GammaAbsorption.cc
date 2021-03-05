
#include "GammaAbsorption.hh"

#include "G4EmProcessSubType.hh"

GammaAbsorption::GammaAbsorption(const G4String& processName, G4ProcessType type)
: G4VDiscreteProcess (processName, type) {
  SetProcessSubType(fPhotoElectricEffect);
  fAbsorptionThreshold = 0.25; // 250 keV;
}

GammaAbsorption::~GammaAbsorption() {}

// implementation of virtual method, specific for G4VEmProcess
G4double
GammaAbsorption::PostStepGetPhysicalInteractionLength(const G4Track& track,
                                                      G4double,
                                                      G4ForceCondition* condition) {
  *condition = NotForced;
  return (track.GetDynamicParticle()->GetKineticEnergy() > fAbsorptionThreshold) ? DBL_MAX : 0.0;
}

// implementation of virtual method, specific for G4VEmProcess
G4VParticleChange*
GammaAbsorption::PostStepDoIt(const G4Track& track,const G4Step&) {
  fParticleChange.InitializeForPostStep(track);
  fParticleChange.ProposeLocalEnergyDeposit(track.GetDynamicParticle()->GetKineticEnergy());
  fParticleChange.SetProposedKineticEnergy(0.0);
  fParticleChange.ProposeTrackStatus(fStopAndKill);

  return &fParticleChange;
}


G4double
GammaAbsorption::GetMeanFreePath( const G4Track& track, G4double, G4ForceCondition*) {
  return (track.GetDynamicParticle()->GetKineticEnergy() > fAbsorptionThreshold) ? DBL_MAX : 0.0;
}
