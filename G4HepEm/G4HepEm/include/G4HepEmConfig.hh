
#ifndef G4HepEmConfig_h
#define G4HepEmConfig_h 1

#include "globals.hh"

#include <vector>
#include <string>


struct G4HepEmParameters;
struct G4HepEmRegionParmeters;

/**
 * @file    G4HepEmConfig.hh
 * @class   G4HepEmConfig
 * @author  M. Novak
 * @date    2025
 *
 * Stores the configuration of the HepEm simulation.
 *
 * The `G4HepEmTrackingManager` stores an object from this class that allows its
 * configuration.
 *
 * Most of the configurations/parameters are those stored in the
 * `G4HepEmParameters` member of this configuration class. These values are red
 * at run time so they can be set anytime before the `beamON`. Many of these
 * parameters can be configured in G4HepEm per detector regions!
 *
 * Other parameters/configurations are already needed at the initialisation,
 * e.g. list of Woodcock tracking regions. However, the `G4HepEmTrackingManager`
 * is constructed only during the initialisation itself (in the physics list
 * `ConstructProcess` interface). Therefore, these parameters/configurations
 * need to be set in the code right after the `G4HepEmTrackingManager` is
 * constructed and assigned to the particles (i.e. in `ConstructProcess`).
 */


// Class to hold parameters that can be set by the user.
class G4HepEmConfig {
public:
  G4HepEmConfig();
 ~G4HepEmConfig();

  void Dump();

  // Activate/deactivate Woodcock-tracking of photons in a given detector region
  // NOTE: this must be done before the initialisation of the run, i.e. right
  //       after the construction of the `G4HepEmTrackinManager` !!!
  // In case the selected region has sub-regions, those also use Woodcock tracking,
  // at least in the part that is within the selected parent region.
  // Additionally selecting these sub-regions is forbidden, as it would mess up the
  // root bounding volume configuration.
  void SetWoodcockTrackingRegion(const std::string& regionName, G4bool val=true);
  const std::vector<std::string>& GetWoodcockTrackingRegionNames() {
    return fWDTRegionNames;
  }

  // Set/get the kinetic energy limit below which Woodcock tracking should be turned off.*/
  void     SetWDTEnergyLimit(G4double ekin) { fWDTEnergyLimit = ekin; }
  G4double GetWDTEnergyLimit() { return fWDTEnergyLimit; }



  // Set the `fDRoverRange` and `fFinalRange` parameters of the continuous energy
  // loss step limit function (everywhere or in a given detector region)
  void     SetEnergyLossStepLimitFunctionParameters(G4double drRange, G4double finRange, const G4String& nameRegion);
  void     SetEnergyLossStepLimitFunctionParameters(G4double drRange, G4double finRange);
  G4double GetEnergyLossStepLimitFunctionParameters(const G4String& nameRegion, G4bool isDRover=true);
  G4double GetEnergyLossStepLimitFunctionParameters(int indxRegion, G4bool isDRover=true);

  // Set the 'rangeFactor' parameter of the MSC step (everywhere or in a given detector region)
  void     SetMSCRangeFactor(G4double val, const G4String& nameRegion);
  void     SetMSCRangeFactor(G4double val);
  G4double GetMSCRangeFactor(const G4String& nameRegion);
  G4double GetMSCRangeFactor(int indxRegion);
  // Set the 'safetyFactor' parameter of the MSC step (everywhere or in a given detector region)
  void     SetMSCSafetyFactor(G4double val, const G4String& nameRegion);
  void     SetMSCSafetyFactor(G4double val);
  G4double GetMSCSafetyFactor(const G4String& nameRegion);
  G4double GetMSCSafetyFactor(int indxRegion);

  // Activating/deactivating energy loss fluctuation everywhere or in a given
  // detector region (default: true --> active)
  void   SetEnergyLossFluctuation(G4bool val, const G4String& nameRegion);
  void   SetEnergyLossFluctuation(G4bool val);
  G4bool GetEnergyLossFluctuation(const G4String& nameRegion);
  G4bool GetEnergyLossFluctuation(int indxRegion);

  // Activating/deactivating using the `minimal` MSC step limit everywhere or in
  // a given detector region (the default i.e. `fUseSafety` is used otherwise)
  void   SetMinimalMSCStepLimit(G4bool val, const G4String& nameRegion);
  void   SetMinimalMSCStepLimit(G4bool val);
  G4bool GetMinimalMSCStepLimit(const G4String& nameRegion);
  G4bool GetMinimalMSCStepLimit(int indxRegion);

  // Activating/deactivating using the combined MSC + Transportation e-/e+ tracking/stepping
  // through multiple steps (default: true --> active)
  void   SetMultipleStepsInMSCWithTransportation(G4bool val, const G4String& nameRegion);
  void   SetMultipleStepsInMSCWithTransportation(G4bool val);
  G4bool GetMultipleStepsInMSCWithTransportation(const G4String& nameRegion);
  G4bool GetMultipleStepsInMSCWithTransportation(int indxRegion);

  // Activating/deactivating applying cuts on other processes beyond ioni. and brem.
  void   SetApplyCuts(G4bool val, const G4String& nameRegion);
  void   SetApplyCuts(G4bool val);
  G4bool GetApplyCuts(const G4String& nameRegion);
  G4bool GetApplyCuts(int indxRegion);

  // Set the e-/e+ tracking (kinetic) energy cut everywhere or in a given detector
  // region (default: G4EmParameters::LowestElectronEnergy). Unlike
  // `/process/em/lowestElectronEnergy`, it does not change the e- production threshold.
  void     SetElectronTrackingCut(G4double val, const G4String& nameRegion);
  void     SetElectronTrackingCut(G4double val);
  G4double GetElectronTrackingCut(const G4String& nameRegion);
  G4double GetElectronTrackingCut(int indxRegion);

  // Set the gamma tracking energy cut everywhere or in a given detector region (default: 0)
  void     SetGammaTrackingCut(G4double val, const G4String& nameRegion);
  void     SetGammaTrackingCut(G4double val);
  G4double GetGammaTrackingCut(const G4String& nameRegion);
  G4double GetGammaTrackingCut(int indxRegion);

  // Also apply the tracking cuts at the start of each step (default: false), i.e.
  // stop a track at or below the cut of its current region in a zero-length step,
  // as `G4UserSpecialCuts` does (newborn secondaries, tracks entering a region
  // with a higher cut). Otherwise the cuts are applied only after the energy loss
  // (e-/e+) or the interaction (gamma) of a step.
  void   SetTrackingCutsAtStepStart(G4bool val);
  G4bool GetTrackingCutsAtStepStart();


  // NOTE: to see if we set its memebr Parameters or it will have its own
  void SetG4HepEmParameters(G4HepEmParameters* hepEmPars) {
    fG4HepEmParameters = hepEmPars;
  }
  G4HepEmParameters* GetG4HepEmParameters() { return fG4HepEmParameters; }

  // Extends the per-region parameters to regions created after this configuration
  // (e.g. by Geant4 VMC after the physics construction). New regions get the values
  // given to the setters for all regions, or the defaults.
  void UpdateRegions();

private:
  G4int GetRegionIndex(const G4String& nameRegion);
  void  CheckRegionIndex(G4int indxRegion);

private:

  // The G4HepEmParameters of the configuration. Default values are taken from Geant4
  // (mainly from G4EmParameters) but further configurations possible through this
  // G4HepEmConfig (even per detector regions).
  G4HepEmParameters*      fG4HepEmParameters;

  // The per-region parameters given to regions created later.
  G4HepEmRegionParmeters* fDefaultRegionParameters;

  // The list of detector regions that the user requested Woodock-tracking in.
  std::vector<std::string> fWDTRegionNames;
  // Kinetic energy below which Woodcock tracking is turned off.*/
  G4double                 fWDTEnergyLimit;

};

#endif // G4HepEmConfig
