
#ifndef G4HepEmConfig_h
#define G4HepEmConfig_h 1

#include "globals.hh"

#include <vector>
#include <string>


class G4HepEmParameters;

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
  // NOTE: this must be done before the initialissation of the run, i.e. right
  //       after the construction of the `G4HepEmTrackinManager` !!!
  void SetWoodcockTrackingRegion(const std::string& regionName, G4bool val=true);
  std::vector<std::string>& GetWoodcockTrackingRegionNames() {
    return fWDTRegionNames;
  }


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



  // NOTE: to see if we set its memebr Parameters or it will have its own
  void SetG4HepEmParameters(G4HepEmParameters* hepEmPars) {
    fG4HepEmParameters = hepEmPars;
  }
  G4HepEmParameters* GetG4HepEmParameters() { return fG4HepEmParameters; }

private:
  G4int GetRegionIndex(const G4String& nameRegion);
  void  CheckRegionIndex(G4int indxRegion);

private:

  // The G4HepEmParameters of the configuration. Default values are taken from Geant4
  // (mainly from G4EmParameters) but further configurations possible through this
  // G4HepEmConfig (even per detector regions).
  G4HepEmParameters*      fG4HepEmParameters;

  // The list of detector regions that the user requested Woodock-tracking in.
  std::vector<std::string> fWDTRegionNames;

};

#endif // G4HepEmConfig
