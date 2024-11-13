#ifndef G4HepEmTrackingManager_h
#define G4HepEmTrackingManager_h 1

#include "G4VTrackingManager.hh"
#include "globals.hh"

class G4HepEmRunManager;
class G4HepEmRandomEngine;
class G4HepEmNoProcess;
class G4SafetyHelper;
class G4Step;
class G4VProcess;

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
  void TrackGamma(G4Track *aTrack);

  // Checks if the particles has fast simulation maanger process attached and
  // stores in the local `fFastSimProcess` array (indexed by HepEm particle ID)
  void InitFastSimRelated(int particleID);

  G4HepEmRunManager *fRunManager;
  G4HepEmRandomEngine *fRandomEngine;
  G4SafetyHelper *fSafetyHelper;
  G4Step *fStep;

  const std::vector<G4double> *theCutsGamma = nullptr;
  const std::vector<G4double> *theCutsElectron = nullptr;
  const std::vector<G4double> *theCutsPositron = nullptr;
  G4bool applyCuts = false;
  G4bool fMultipleSteps = true;

  // A set of empty processes with the correct names and types just to be able
  // to set them as process limiting the step and creating secondaries as some
  // user codes rely on this information.
  std::vector<G4HepEmNoProcess *> fElectronNoProcessVector;
  std::vector<G4HepEmNoProcess *> fGammaNoProcessVector;
  G4HepEmNoProcess *fTransportNoProcess;

  // Pointers to the fast simulation manager processes of the 3 particles if any
  // [0] e-; [1] e+; [2] gamma; nullptr: no fast sim manager process attached
  G4VProcess* fFastSimProcess[3];

};

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

#endif
