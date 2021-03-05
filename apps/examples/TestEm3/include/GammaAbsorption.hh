
#ifndef GammaAbsorption_HH
#define GammaAbsorption_HH

#include "globals.hh"
#include "G4VDiscreteProcess.hh"
#include "G4ParticleChangeForGamma.hh"



class G4ParticleDefinition;

// Simple G4VDiscreteProcess that will absorb gammas when their energy is
// below the fAbsorptionThreshold ()= 250 [keV]) energy.
// NOTE: this process will need to be replaced with G4PhotoElectricEffect when
//       the corresponding interaction will be handled by G4HepEm. Till that,
//       this process corresponds to the G4HepEmGammaInteractionPhotoelectric
//       process.
class GammaAbsorption : public G4VDiscreteProcess {

public:
   GammaAbsorption(const G4String& processName ="gammaAbsorption",
			             G4ProcessType type = fElectromagnetic);

  ~GammaAbsorption() override;

  // implementation of virtual method, specific for G4VEmProcess
  G4double PostStepGetPhysicalInteractionLength(const G4Track&, G4double, G4ForceCondition*) override;

  // implementation of virtual method, specific for G4VEmProcess
  G4VParticleChange* PostStepDoIt(const G4Track&, const G4Step&) override;


protected:

  G4double GetMeanFreePath( const G4Track& aTrack,
                            G4double previousStepSize,
                            G4ForceCondition* condition ) override;

private:
  G4ParticleChangeForGamma     fParticleChange;
  G4double                     fAbsorptionThreshold;

};

#endif // GammaAbsorption_HH
