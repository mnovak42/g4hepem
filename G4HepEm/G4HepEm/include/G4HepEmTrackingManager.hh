#ifndef G4HepEmTrackingManager_h
#define G4HepEmTrackingManager_h 1

#include "G4VTrackingManager.hh"
#include "globals.hh"

class G4HepEmRunManager;
class G4HepEmRandomEngine;
class G4HepEmNoProcess;
class G4HepEmTLData;
class G4SafetyHelper;
class G4Step;
class G4VProcess;
class G4Region;

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

  // ATLAS XTR RELATED:
  // Set the names of the ATLAS specific transition radiation process and
  // radiator region (only for ATLAS and only if different than init.ed below)
  void SetXTRProcessName(const std::string& name) { fXTRProcessName = name; }
  void SetXTRRegionName(const std::string& name)  { fXTRRegionName  = name; }


private:
  void TrackElectron(G4Track *aTrack);
  void TrackGamma(G4Track *aTrack);

  // Stacks secondaries created by HepEm physics (if any) and returns with the
  // energy deposit while stacking due to applying secondary production cuts
  double StackSecondaries(G4HepEmTLData* aTLData, G4Track* aG4PrimaryTrack,
                          const G4VProcess* aG4CreatorProcess, int aG4IMC);

  // Checks if the particles has fast simulation maanger process attached and
  // stores in the local `fFastSimProcess` array (indexed by HepEm particle ID)
  void InitFastSimRelated(int particleID);

  // ATLAS XTR RELATED:
  // Called at init to find the ATLAS specific,Athena local transition radiation
  // process, detector region pointers and store them in field variables allowing
  // to invoke that process during the e-/e+ tracking
  // NOTE: the fields stays nullptr if no such process/region are found causing
  //       no harm outside Athena.
  void InitXTRRelated();


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

  // ATLAS XTR RELATED:
  // Fields to store ptrs to the ATLAS XTR (transition radiation) process and
  // detector region that contain the tradiator volumes
  G4VProcess* fXTRProcess;
  G4Region*   fXTRRegion;
  // The names that will be used to find the XTR process and detector region.
  std::string fXTRProcessName = {"XTR"};
  std::string fXTRRegionName  = {"TRT_RADIATOR"};
};

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

#endif
