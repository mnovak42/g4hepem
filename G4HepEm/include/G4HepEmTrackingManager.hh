#ifndef G4HepEmTrackingManager_h
#define G4HepEmTrackingManager_h 1

#include "G4VTrackingManager.hh"
#include "globals.hh"

class G4HepEmRunManager;
class G4HepEmRandomEngine;
class G4SafetyHelper;
class G4Step;

#include <vector>

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

class G4HepEmTrackingManager : public G4VTrackingManager {
public:
  G4HepEmTrackingManager();
  ~G4HepEmTrackingManager();

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
  G4HepEmRandomEngine *fRandomEngine;
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
