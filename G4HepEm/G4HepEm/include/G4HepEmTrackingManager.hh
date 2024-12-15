#ifndef G4HepEmTrackingManager_h
#define G4HepEmTrackingManager_h 1

#include "G4EventManager.hh"
#include "G4VTrackingManager.hh"
#include "globals.hh"

class G4HepEmRunManager;
class G4HepEmRandomEngine;
class G4HepEmNoProcess;
class G4HepEmTLData;
class G4SafetyHelper;
class G4Step;
class G4VProcess;
class G4VParticleChange;
class G4Region;
class G4HepEmWoodcockHelper;

#include <vector>

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

class G4HepEmTrackingManager : public G4VTrackingManager {
public:
  G4HepEmTrackingManager();
  virtual ~G4HepEmTrackingManager();

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

  void AddWoodcockTrackingRegion(const std::string& regionName) {
    fWDTRegionNames.push_back(regionName);
  }

protected:
  bool TrackElectron(G4Track *aTrack);
  bool TrackGamma(G4Track *aTrack);

  // Pointers to the fast simulation manager processes of the 3 particles if any
  // [0] e-; [1] e+; [2] gamma; nullptr: no fast sim manager process attached
  G4VProcess *fFastSimProcess[3];

private:
  // Stacks secondaries created by HepEm physics (if any) and returns with the
  // energy deposit while stacking due to applying secondary production cuts
  double StackSecondaries(G4HepEmTLData* aTLData, G4Track* aG4PrimaryTrack,
                          const G4VProcess* aG4CreatorProcess, int aG4IMC);

  // Stacks secondaries created by Geant4 physics (if any) and returns with the
  // energy deposit while stacking due to applying secondary production cuts
  double StackG4Secondaries(G4VParticleChange* particleChange,
                            G4Track* aG4PrimaryTrack,
                            const G4VProcess* aG4CreatorProcess, int aG4IMC);

  void InitNuclearProcesses(int particleID);

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

#ifdef G4HepEm_EARLY_TRACKING_EXIT
  // Virtual function to check early tracking exit. This function allows user
  // implementations to intercept the G4HepEm tracking loop based on
  // user-defined conditions, e.g., when entering a GPU region To be implemented
  // in derived classes by users, base implementation does nothing and returns
  // false
  virtual bool CheckEarlyTrackingExit(G4Track *track, G4EventManager *evtMgr,
                                      G4UserTrackingAction *userTrackingAction,
                                      G4TrackVector &secondaries) const {
    return false;
  }
#endif

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

  // Pointers to the Gamma-nuclear process (if any)
  G4VProcess* fGNucProcess;

  // Pointers to the Electron/Positron-nuclear processes (if any)
  G4VProcess* fENucProcess;
  G4VProcess* fPNucProcess;

  // ATLAS XTR RELATED:
  // Fields to store ptrs to the ATLAS XTR (transition radiation) process and
  // detector region that contain the tradiator volumes
  G4VProcess* fXTRProcess;
  G4Region*   fXTRRegion;
  // The names that will be used to find the XTR process and detector region.
  std::string fXTRProcessName = {"XTR"};
  std::string fXTRRegionName  = {"TRT_RADIATOR"};

  // A vector of Woodcock tracking region names (set by user if any) and a helper.
  std::vector<std::string> fWDTRegionNames;
  G4HepEmWoodcockHelper*   fWDTHelper;
};

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

#endif
