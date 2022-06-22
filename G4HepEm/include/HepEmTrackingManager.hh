#ifndef HepEmTrackingManager_h
#define HepEmTrackingManager_h 1

#include "G4VTrackingManager.hh"
#include "globals.hh"

class G4HepEmRunManager;
class G4HepEmCLHEPRandomEngine;
class G4SafetyHelper;
class G4Step;

#include <vector>

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

class HepEmTrackingManager : public G4VTrackingManager {
public:
  HepEmTrackingManager();
  ~HepEmTrackingManager();

  void BuildPhysicsTable(const G4ParticleDefinition &) override;

  void PreparePhysicsTable(const G4ParticleDefinition &) override;

  void HandOverOneTrack(G4Track *aTrack) override;

  void SetMultipleSteps(G4bool val) {
    fMultipleSteps = val;
  }
  G4bool MultipleSteps() const {
    return fMultipleSteps;
  }

private:
  void TrackElectron(G4Track *aTrack);
  void TrackPositron(G4Track *aTrack);
  void TrackGamma(G4Track *aTrack);

  G4HepEmRunManager *fRunManager;
  G4HepEmCLHEPRandomEngine *fRandomEngine;
  G4SafetyHelper *fSafetyHelper;
  G4Step *fStep;

  const std::vector<G4double> *theCutsGamma = nullptr;
  const std::vector<G4double> *theCutsElectron = nullptr;
  const std::vector<G4double> *theCutsPositron = nullptr;
  G4bool applyCuts = false;
  G4bool fMultipleSteps = true;
};

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

#endif
