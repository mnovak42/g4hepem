
#include "G4HepEmConfig.hh"

#include "G4HepEmParameters.hh"
#include "G4HepEmParametersInit.hh"

#include "G4RegionStore.hh"
#include "G4Region.hh"
#include "G4SystemOfUnits.hh"

G4HepEmConfig::G4HepEmConfig() {
  fG4HepEmParameters = new G4HepEmParameters;
  InitHepEmParameters(fG4HepEmParameters);
}


G4HepEmConfig::~G4HepEmConfig() {
  FreeG4HepEmParameters(fG4HepEmParameters);
  delete fG4HepEmParameters;
}


void G4HepEmConfig::SetWoodcockTrackingRegion(const std::string& regionName, G4bool val) {
  // check if the region name has already been added in order to avoid duplications
  // - remove if it was requested (i.e. `val=false`)
  for (std::vector<std::string>::iterator it=fWDTRegionNames.begin(); it != fWDTRegionNames.end(); ++it) {
    if (*it==regionName) {
      if (!val) { fWDTRegionNames.erase(it); }
      return;
    }
  }
  // add to the list if it was not be there
  if (val) { fWDTRegionNames.push_back(regionName); }
}


void G4HepEmConfig::SetEnergyLossStepLimitFunctionParameters(G4double drRange, G4double finRange, const G4String& nameRegion) {
  if (nameRegion == "all") {
    SetEnergyLossStepLimitFunctionParameters(drRange, finRange);
  } else {
    fG4HepEmParameters->fParametersPerRegion[GetRegionIndex(nameRegion)].fDRoverRange = drRange;
    fG4HepEmParameters->fParametersPerRegion[GetRegionIndex(nameRegion)].fFinalRange  = finRange;
  }
}
void G4HepEmConfig::SetEnergyLossStepLimitFunctionParameters(G4double drRange, G4double finRange) {
  for (int i=0; i<fG4HepEmParameters->fNumRegions; ++i) {
    fG4HepEmParameters->fParametersPerRegion[i].fDRoverRange = drRange;
    fG4HepEmParameters->fParametersPerRegion[i].fFinalRange  = finRange;
  }
}
G4double G4HepEmConfig::GetEnergyLossStepLimitFunctionParameters(const G4String& nameRegion, G4bool isDRover) {
  return GetEnergyLossStepLimitFunctionParameters(GetRegionIndex(nameRegion), isDRover);
}
G4double G4HepEmConfig::GetEnergyLossStepLimitFunctionParameters(G4int indxRegion, G4bool isDRover) {
  CheckRegionIndex(indxRegion);
  return isDRover ? fG4HepEmParameters->fParametersPerRegion[indxRegion].fDRoverRange
                  : fG4HepEmParameters->fParametersPerRegion[indxRegion].fFinalRange;
}


void G4HepEmConfig::SetMSCRangeFactor(G4double val, const G4String& nameRegion) {
  if (nameRegion == "all") {
    SetMSCRangeFactor(val);
  } else {
    fG4HepEmParameters->fParametersPerRegion[GetRegionIndex(nameRegion)].fMSCRangeFactor = val;
  }
}
void G4HepEmConfig::SetMSCRangeFactor(G4double val) {
  for (int i=0; i<fG4HepEmParameters->fNumRegions; ++i)
    fG4HepEmParameters->fParametersPerRegion[i].fMSCRangeFactor = val;
}
G4double G4HepEmConfig::GetMSCRangeFactor(const G4String& nameRegion) {
  return GetMSCRangeFactor(GetRegionIndex(nameRegion));
}
G4double G4HepEmConfig::GetMSCRangeFactor(G4int indxRegion) {
  CheckRegionIndex(indxRegion);
  return fG4HepEmParameters->fParametersPerRegion[indxRegion].fMSCRangeFactor;
}

void G4HepEmConfig::SetMSCSafetyFactor(G4double val, const G4String& nameRegion) {
  if (nameRegion == "all") {
    SetMSCSafetyFactor(val);
  } else {
    fG4HepEmParameters->fParametersPerRegion[GetRegionIndex(nameRegion)].fMSCSafetyFactor = val;
  }
}
void G4HepEmConfig::SetMSCSafetyFactor(G4double val) {
  for (int i=0; i<fG4HepEmParameters->fNumRegions; ++i)
    fG4HepEmParameters->fParametersPerRegion[i].fMSCSafetyFactor = val;
}
G4double G4HepEmConfig::GetMSCSafetyFactor(const G4String& nameRegion) {
  return GetMSCSafetyFactor(GetRegionIndex(nameRegion));
}
G4double G4HepEmConfig::GetMSCSafetyFactor(G4int indxRegion) {
  CheckRegionIndex(indxRegion);
  return fG4HepEmParameters->fParametersPerRegion[indxRegion].fMSCSafetyFactor;
}


void G4HepEmConfig::SetEnergyLossFluctuation(G4bool val, const G4String& nameRegion) {
  if (nameRegion == "all") {
    SetEnergyLossFluctuation(val);
  } else {
    fG4HepEmParameters->fParametersPerRegion[GetRegionIndex(nameRegion)].fIsELossFluctuation = val;
  }
}
void G4HepEmConfig::SetEnergyLossFluctuation(G4bool val) {
  for (int i=0; i<fG4HepEmParameters->fNumRegions; ++i)
    fG4HepEmParameters->fParametersPerRegion[i].fIsELossFluctuation = val;
}
G4bool G4HepEmConfig::GetEnergyLossFluctuation(const G4String& nameRegion) {
  return GetEnergyLossFluctuation(GetRegionIndex(nameRegion));
}
G4bool G4HepEmConfig::GetEnergyLossFluctuation(G4int indxRegion) {
  CheckRegionIndex(indxRegion);
  return fG4HepEmParameters->fParametersPerRegion[indxRegion].fIsELossFluctuation;
}


void G4HepEmConfig::SetMinimalMSCStepLimit(G4bool val, const G4String& nameRegion) {
  if (nameRegion == "all") {
    SetMinimalMSCStepLimit(val);
  } else {
    fG4HepEmParameters->fParametersPerRegion[GetRegionIndex(nameRegion)].fIsMSCMinimalStepLimit = val;
  }
}
void G4HepEmConfig::SetMinimalMSCStepLimit(G4bool val) {
  for (int i=0; i<fG4HepEmParameters->fNumRegions; ++i)
    fG4HepEmParameters->fParametersPerRegion[i].fIsMSCMinimalStepLimit = val;
}
G4bool G4HepEmConfig::GetMinimalMSCStepLimit(const G4String& nameRegion) {
  return GetMinimalMSCStepLimit(GetRegionIndex(nameRegion));
}
G4bool G4HepEmConfig::GetMinimalMSCStepLimit(G4int indxRegion) {
  CheckRegionIndex(indxRegion);
  return fG4HepEmParameters->fParametersPerRegion[indxRegion].fIsMSCMinimalStepLimit;
}


void G4HepEmConfig::SetMultipleStepsInMSCWithTransportation(G4bool val, const G4String& nameRegion) {
  if (nameRegion == "all") {
    SetMultipleStepsInMSCWithTransportation(val);
  } else {
    fG4HepEmParameters->fParametersPerRegion[GetRegionIndex(nameRegion)].fIsMultipleStepsInMSCTrans = val;
  }
}
void G4HepEmConfig::SetMultipleStepsInMSCWithTransportation(G4bool val) {
  for (int i=0; i<fG4HepEmParameters->fNumRegions; ++i)
    fG4HepEmParameters->fParametersPerRegion[i].fIsMultipleStepsInMSCTrans = val;
}
G4bool G4HepEmConfig::GetMultipleStepsInMSCWithTransportation(const G4String& nameRegion) {
  return GetMultipleStepsInMSCWithTransportation(GetRegionIndex(nameRegion));
}
G4bool G4HepEmConfig::GetMultipleStepsInMSCWithTransportation(G4int indxRegion) {
  CheckRegionIndex(indxRegion);
  return fG4HepEmParameters->fParametersPerRegion[indxRegion].fIsMultipleStepsInMSCTrans;
}



void G4HepEmConfig::SetApplyCuts(G4bool val, const G4String& nameRegion) {
  if (nameRegion == "all") {
    SetApplyCuts(val);
  } else {
    fG4HepEmParameters->fParametersPerRegion[GetRegionIndex(nameRegion)].fIsApplyCuts = val;
  }
}
void G4HepEmConfig::SetApplyCuts(G4bool val) {
  for (int i=0; i<fG4HepEmParameters->fNumRegions; ++i)
    fG4HepEmParameters->fParametersPerRegion[i].fIsApplyCuts = val;
}
G4bool G4HepEmConfig::GetApplyCuts(const G4String& nameRegion) {
  return GetApplyCuts(GetRegionIndex(nameRegion));
}
G4bool G4HepEmConfig::GetApplyCuts(G4int indxRegion) {
  CheckRegionIndex(indxRegion);
  return fG4HepEmParameters->fParametersPerRegion[indxRegion].fIsApplyCuts;
}


G4int G4HepEmConfig::GetRegionIndex(const G4String& nameRegion) {
  G4Region* region = G4RegionStore::GetInstance()->GetRegion(nameRegion, false);
  if (region == nullptr) {
    std::cerr << "*** ERROR in G4HepEmConfig::GetRegionIndex :\n"
              << "    Unknown detector region with name = "
              << nameRegion
              << std::endl;
    exit(-1);
  }
  return region->GetInstanceID();
}

void  G4HepEmConfig::CheckRegionIndex(G4int indxRegion) {
  if (indxRegion >= fG4HepEmParameters->fNumRegions) {
    std::cerr << "*** ERROR in G4HepEmConfig::CheckRegionIndex :\n"
              << "    Region index ( = " << indxRegion << " ) out of bounds! "
              << std::endl;
    exit(-1);
  }
}


void G4HepEmConfig::Dump() {
 std::cout << "\n ==================== G4HepEmConfig ==================== " << std::endl;
 std::cout << " GLOBAL Parameters: " << std::endl;
 std::cout << " --------------------------------------------------------  " << std::endl;
 std::cout << std::left <<std::setw(30) << " Electron tracking cut " << " : "
           << std::setw(5) << std::right
           << fG4HepEmParameters->fElectronTrackingCut/CLHEP::keV
           << " [keV] " << std::endl;
 std::cout << std::left << std::setw(30) << " Min loss table energies " << " : "
           << std::setw(5) << std::right
           << fG4HepEmParameters->fMinLossTableEnergy/CLHEP::keV
           << " [keV] " << std::endl;
 std::cout << std::left << std::setw(30) << " Max loss table energies " << " : "
           << std::setw(5) << std::right
           << fG4HepEmParameters->fMaxLossTableEnergy/CLHEP::TeV
           << " [TeV] " << std::endl;
 std::cout << std::left << std::setw(30) << " Number of loss table bins " << " : "
           << std::setw(5) << std::right
           << fG4HepEmParameters->fNumLossTableBins << std::endl;
 std::cout << std::left << std::setw(30) << " Linear loss limit " << " : "
           << std::setw(5) << std::right
           << fG4HepEmParameters->fParametersPerRegion[0].fLinELossLimit*100
           << " [%] " << std::endl << std::endl;
 std::cout << " REGION BASED Parameters: " << std::endl;
 std::cout << " --------------------------------------------------------  " << std::endl;
 std::cout << std::left << std::setw(30) << " Parameter for region " << " : ";
 const int numRegions = fG4HepEmParameters->fNumRegions;
 std::vector<int> lengthNameRegions;
 for (int ir=0; ir<numRegions; ++ir) {
   const G4String& nameRegion = (*G4RegionStore::GetInstance())[ir]->GetName();
   int len = nameRegion.length();
   std::cout << std::setw(len) << (*G4RegionStore::GetInstance())[ir]->GetName() << " | ";
   lengthNameRegions.push_back(len);
 }
 std::cout << std::endl;

 std::vector<G4String> names = {" FinalRange (mm)", " DRoverRange", " Energy loss fluctuation",
        " MSC Range factor",  " MSC Safety factor", " MSC minimal step limit",
        " Multiple steps in MSC+Trans.", " Woodcock-tracking", " Apply cuts"};
 const int numParams  = names.size();

 for (int ip=0; ip<numParams; ++ip) {
   for (int ir=0; ir<numRegions; ++ir) {
     if (ir==0) {
       std::cout << std::left << std::setw(30) << names[ip] << " : ";
     }
     std::cout << std::right << std::setw(lengthNameRegions[ir]);

     bool isWDT = false;
     const G4String& regionName =  (*G4RegionStore::GetInstance())[ir]->GetName();
     if (ip==7) {
       for (std::vector<std::string>::iterator it=fWDTRegionNames.begin(); it != fWDTRegionNames.end(); ++it) {
         if (*it==regionName) {
           isWDT = true;
           break;
         }
       }
     }

     switch (ip) {
       case 0: std::cout << fG4HepEmParameters->fParametersPerRegion[ir].fFinalRange/CLHEP::mm << " | ";
              break;
       case 1: std::cout << fG4HepEmParameters->fParametersPerRegion[ir].fDRoverRange << " | ";
              break;
       case 2: std::cout << fG4HepEmParameters->fParametersPerRegion[ir].fIsELossFluctuation << " | ";
              break;
       case 3: std::cout << fG4HepEmParameters->fParametersPerRegion[ir].fMSCRangeFactor << " | ";
              break;
       case 4: std::cout << fG4HepEmParameters->fParametersPerRegion[ir].fMSCSafetyFactor << " | ";
              break;
       case 5: std::cout << fG4HepEmParameters->fParametersPerRegion[ir].fIsMSCMinimalStepLimit << " | ";
              break;
       case 6: std::cout << fG4HepEmParameters->fParametersPerRegion[ir].fIsMultipleStepsInMSCTrans << " | ";
              break;
       case 7: std::cout << isWDT << " | ";
              break;
       case 8: std::cout << fG4HepEmParameters->fParametersPerRegion[ir].fIsApplyCuts << " | ";
              break;

     }
   }
   std::cout << std::endl;
 }
 std::cout << " ========================================================= " << std::endl << std::endl;
}
