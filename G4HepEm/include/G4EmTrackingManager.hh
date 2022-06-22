#ifndef G4EmTrackingManager_h
#define G4EmTrackingManager_h 1

#include "G4VTrackingManager.hh"
#include "globals.hh"

class G4eMultipleScattering;
class G4CoulombScattering;
class G4eIonisation;
class G4eBremsstrahlung;
class G4eplusAnnihilation;

class G4ComptonScattering;
class G4GammaConversion;
class G4PhotoElectricEffect;

class G4Step;

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

class G4EmTrackingManager : public G4VTrackingManager {
public:
  G4EmTrackingManager();
  ~G4EmTrackingManager();

  void BuildPhysicsTable(const G4ParticleDefinition &) override;

  void PreparePhysicsTable(const G4ParticleDefinition &) override;

  void HandOverOneTrack(G4Track *aTrack) override;

private:
  void TrackElectron(G4Track *aTrack);
  void TrackPositron(G4Track *aTrack);
  void TrackGamma(G4Track *aTrack);

  struct {
    G4eMultipleScattering *msc;
    G4eIonisation *ioni;
    G4eBremsstrahlung *brems;
    G4CoulombScattering *ss;
  } electron;

  struct {
    G4eMultipleScattering *msc;
    G4eIonisation *ioni;
    G4eBremsstrahlung *brems;
    G4eplusAnnihilation *annihilation;
    G4CoulombScattering *ss;
  } positron;

  struct {
    G4PhotoElectricEffect *pe;
    G4ComptonScattering *compton;
    G4GammaConversion *conversion;
  } gamma;

  G4Step *fStep;

  static G4EmTrackingManager *masterTrackingManager;
};

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

#endif
