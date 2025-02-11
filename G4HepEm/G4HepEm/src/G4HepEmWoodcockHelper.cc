
#include "G4HepEmWoodcockHelper.hh"

#include "G4VSolid.hh"
#include "G4MaterialCutsCouple.hh"
#include "G4Material.hh"
#include "G4LogicalVolume.hh"
#include "G4VPhysicalVolume.hh"
#include "G4RegionStore.hh"

#include "G4Track.hh"
#include "G4TouchableHistory.hh"
#include "G4NavigationHistory.hh"
#include "G4TransportationManager.hh"

#include "Randomize.hh"

#include "G4HepEmData.hh"
#include "G4HepEmMatCutData.hh"
#include "G4HepEmGammaTrack.hh"

#include "G4HepEmGammaManager.hh"

G4HepEmWoodcockHelper::G4HepEmWoodcockHelper()
: fWDTSolid(nullptr),
  fWDTCouple(nullptr),
  fWDTHepEmIMC(-1),
  fWDTKineticEnergyLimit(0.2) // 200 keV
{ }


G4HepEmWoodcockHelper::~G4HepEmWoodcockHelper() {
  ClearData();
}


G4bool G4HepEmWoodcockHelper::Initialize(std::vector<std::string>& wdtRegionNames, struct G4HepEmMatCutData* hepEmMatCutData, G4VPhysicalVolume* worldVolume) {
  // make sure that all data are cleared
  ClearData();
  // NOTE: I will alway know that a given region is Woodcock region or not by
  // checking if the region ID is in the `fWDTData` map. And do not need to check
  // in each step during the tracking: as long as the region stays the same.
  //
  G4bool oneHasBeenFound = false;
  // Try to find the Woodcock tracking regions (one-by-one) in the store by using
  // their names (given by the user).
  const int numWDTRegions = wdtRegionNames.size();
  for (int ir=0; ir<numWDTRegions; ++ir) {
    const std::string& wdtRegionName = wdtRegionNames[ir];
    G4Region* wdtRegion = G4RegionStore::GetInstance()->GetRegion(wdtRegionName, false);
    if (wdtRegion != nullptr) {
      oneHasBeenFound = true;
      // Found a region with the given name: set Region ID to index map element
      // Create a WDT data for this region and store in the `fWDTData` map
      WDTDataForARegion* wdtDataRegion = new WDTDataForARegion();
      fWDTData[wdtRegion->GetInstanceID()] = wdtDataRegion;
      // Iterate the root logical volumes of this region.
      // Find and store their solid and the heaviest material within each
      int numRootLVolume = wdtRegion->GetNumberOfRootVolumes();
      // std::cout << "G4HepEmWoodcockHelper::Initialize() Woodcock region = "
      //          << wdtRegionName << " was found with "
      //          << numRootLVolume << " root logical volumes."
      //          << std::endl;
      std::vector<G4LogicalVolume*>::const_iterator itrLV = wdtRegion->GetRootLogicalVolumeIterator();
      for (std::size_t ilv = 0; ilv<numRootLVolume; ++ilv) {
        G4LogicalVolume* rootLogVol = (*itrLV);
        // std::cout << " The [ " << ilv << " ]-th root logical volume is "
        //          << rootLogVol->GetName() << std::endl;
        G4double maxDensity = -1.0;
        G4Material*  wdtMat = nullptr;
        // find the material with maximum density in this root logical volume branch
        FindWDTMaterial(rootLogVol, maxDensity, &wdtMat);
        G4MaterialCutsCouple* wdtCouple = wdtRegion->FindCouple(wdtMat);
        // std::cout << " The heaviest material in this branch is " << wdtMat->GetName()
        //          << " wdtCouple indx = " << wdtCouple->GetIndex()
        //          << " wdTMat name = " << wdtMat->GetName()
        //          << std::endl;
        // Create a `WDTDataPerRootLogVol`structure for this root logical volume
        // set all required fields and store in the map indexed(key) by the log. vol. ID
        const int hepEmIMC = hepEmMatCutData->fG4MCIndexToHepEmMCIndex[wdtCouple->GetIndex()];
        WDTDataPerRootLogVol* wdtDataRootLogVol = new WDTDataPerRootLogVol(rootLogVol->GetSolid(), wdtCouple, hepEmIMC);
        wdtDataRegion->fWDTDataRegion[rootLogVol->GetInstanceID()] = wdtDataRootLogVol;
        ++itrLV;
      }
    } else {
      // std::cout << "G4HepEmWoodcockHelper::Initialize() Woodcock region = "
      //          << wdtRegionName << " was NOT found!"
      //          << std::endl;
    }
  }
  //
  if (oneHasBeenFound) {
    // Set the world volume of the WDT navigator
    fWDTNavigator.SetWorldVolume(worldVolume);
  }
  return oneHasBeenFound;
}


G4bool G4HepEmWoodcockHelper::FindWDTVolume(int regionID, const G4Track& aTrack) {
  // Early return if this step is not in a WDT region or kinetic energy is too low
  if ( fWDTData.find(regionID) == fWDTData.end() || aTrack.GetKineticEnergy() < fWDTKineticEnergyLimit) {
    return false;
  }
  // Obtain the Woodcock tracking data stored for the actual region
  WDTDataForARegion* wdtDataRegion = fWDTData[regionID];
  // Find the actual root logical volume within this step is done
  // NOTE: it doesn't change the state of the touchable so I can keep tracking after
  const G4NavigationHistory* navHistory = ((G4TouchableHistory*)(aTrack.GetTouchableHandle()()))->GetHistory();
  int currentDepth = navHistory->GetDepth();
  std::map<G4int, WDTDataPerRootLogVol*>::iterator itr;
  while (currentDepth > -1) {
    itr = wdtDataRegion->fWDTDataRegion.find(navHistory->GetVolume(currentDepth)->GetLogicalVolume()->GetInstanceID());
    if (itr != wdtDataRegion->fWDTDataRegion.end()) {
      break;
    }
    // moving one level up in the geometry tree
    --currentDepth;
  };
  // Don't do WDT if the volume was not found
  if (currentDepth < 0) {
    return false;
  }
  // Get the WDT data for the root logical volume in which the actual tracking happens
  WDTDataPerRootLogVol* wdtDataRootLogVol = itr->second;
  // Obtain the actual transformation: used to transform the actual point/direction to
  // volume local before computing the distance to its boundary.
  fWDTTransform = navHistory->GetTransform(currentDepth);
  fWDTHepEmIMC  = wdtDataRootLogVol->fG4CoupleHepEmIndex;
  fWDTSolid     = wdtDataRootLogVol->fSolid;
  fWDTCouple    = wdtDataRootLogVol->fG4Couple;
  //
  // Check if the current pre-step point is indeed inside. If distance to
  // the root volume boundary is zero (or very small) then just do normal
  // tracking as we are already (logically) outside (might be due to
  // boundary crossing in multiple steps)
  const G4ThreeVector& r0 = aTrack.GetPosition();
  const G4ThreeVector& v0 = aTrack.GetMomentumDirection();
  const G4ThreeVector localPoint     = fWDTTransform.TransformPoint(r0);
  const G4ThreeVector localDirection = fWDTTransform.TransformAxis(v0);
  G4double distToBoundary = std::max(fWDTSolid->DistanceToOut(localPoint, localDirection)-1.0E-3, 0.0 );
  return (distToBoundary < 1.0E-6) ? false : true;
}


G4bool G4HepEmWoodcockHelper::KeepTracking(const struct G4HepEmData* theHepEmData, G4HepEmGammaTrack* theGammaTrack, G4Track& aTrack) {
  // Calculate the distance to boundary and the physics length keep eating up
  // the distance to boundary till: 1. interaction is reached or close to boundary.
  // In both cases, locate the point before going further, then either make
  // a boundary step (case 2.) or let it go to selecting and performing the
  // physics interaction.
  G4StepPoint& preStepPoint  = *(aTrack.GetStep()->GetPreStepPoint());
  G4StepPoint& postStepPoint = *(aTrack.GetStep()->GetPostStepPoint());

  // Compute the distance to the boundary of the actual root volume.
  // - get direction and start position
  const G4ThreeVector& r0 = preStepPoint.GetPosition();
  const G4ThreeVector& v0 = preStepPoint.GetMomentumDirection();
  const G4ThreeVector localPoint     = fWDTTransform.TransformPoint(r0);
  const G4ThreeVector localDirection = fWDTTransform.TransformAxis(v0);
  G4double distToBoundary = std::max( fWDTSolid->DistanceToOut(localPoint, localDirection)-1.0E-3, 0.0 );
  // When `distToBoundary = 0`, we are within 1.0E-3 to the the boundary.
  //
  // Compute the WDT reference mxsec (i.e. maximum total mxsec along this setp).
  // (need to set only the MC index as ekin has alrady been set above).
  G4HepEmTrack* thePrimaryTrack = theGammaTrack->GetTrack();
  thePrimaryTrack->SetMCIndex(fWDTHepEmIMC);
  const G4double wdtMXsec   = G4HepEmGammaManager::GetTotalMacXSec(theHepEmData, theGammaTrack);
  const G4double wdtMFP     = wdtMXsec > 0.0 ? 1.0/wdtMXsec : DBL_MAX;
  const G4double wdtPEmxSec = theGammaTrack->GetPEmxSec();
  // Init some variables before starting Woodcock tracking of the gamma
  const G4Material* wdtMaterial = fWDTCouple->GetMaterial();
  G4double mxsec = 0.0;
  G4int prevHepEmIMC = -1;
  G4bool doStop = false;

  // While either interacts or gets close to the boundary of the actual root volume:
  const G4int* g4MCIndexToHepEmMCIndex = theHepEmData->fTheMatCutData->fG4MCIndexToHepEmMCIndex;
  G4double wdtStepLength = 0.0;
  G4bool isWDTReachedBoundary = false;
  while (!doStop) {
    // Compute the step length till the next interaction in the WDT material
    const G4double pstep = wdtMFP < DBL_MAX ? -G4Log( G4UniformRand() )*wdtMFP : DBL_MAX;
    // Take the minimum of this and the distance to the WDT root volume boundary
    // while checking if this step ends up close to the volume boundary
    if (distToBoundary < pstep) {
      wdtStepLength += distToBoundary;
      isWDTReachedBoundary = true;
      doStop = true;
    } else {
      // Particle will be moved by a step length of `pstep` so we reduce the
      // distance to boundary accordingly.
      wdtStepLength  += pstep;
      distToBoundary -= pstep;
      // Locate the actual post step point in order to get the real material.
      // NOTE: we might start here from a certain depth (i.e. from the depth of
      //       the actual root logical volume)
      const G4VPhysicalVolume* pVol = fWDTNavigator.LocateGlobalPointAndSetup(r0+wdtStepLength*v0, nullptr, true, true);
      const G4LogicalVolume*   lVol = pVol->GetLogicalVolume();
      const G4Material* postStepMat = lVol->GetMaterial();
      // Check if the real material of the post-step point is the WDT one?
      if (wdtMaterial != postStepMat) {
        // Post step point is NOT in the WDT material: need to check if interacts.
        // Compute the total macroscopic cross section for that material.
        const G4MaterialCutsCouple* couple = lVol->GetMaterialCutsCouple();
        const int hepEmIMC = g4MCIndexToHepEmMCIndex[couple->GetIndex()];
        if (hepEmIMC != prevHepEmIMC) {
          // Recompute the total macroscopic cross section only if the material
          // has changed compared to the previous computation (energy stays const.)
          prevHepEmIMC = hepEmIMC;
          thePrimaryTrack->SetMCIndex(hepEmIMC);
          mxsec = G4HepEmGammaManager::GetTotalMacXSec(theHepEmData, theGammaTrack);
        }
        // Sample if interaction happens at this post step point:
        // P(interact) = preStepLambda/wdckMXsec note: preStepLambda <= wdckMXsec
        doStop = (mxsec*wdtMFP > G4UniformRand());
        if (doStop) {
          // Interaction happens: set the track fields required later.
          // Set the total MFP of the track that will be needed when sampling
          // the type of the interaction. The HepEm MC index is already set
          // above while the g4 one is also set here.
          const double mfp = mxsec > 0.0 ? 1.0/mxsec : DBL_MAX;
          thePrimaryTrack->SetMFP(mfp, 0);
          // g4IMC = couple->GetIndex();
          // NOTE: PE mxsec is correct as the last call to `GetTotalMacXSec`
          // was fone above for this material.
        }
      } else {
        // Post step point is in the WDT material: interacts for sure (prob.=1)
        // Set the total MFP and MC index of the track that will be needed when
        // sampling the type of the interaction. The g4 MC index is also set here.
        doStop = true;
        thePrimaryTrack->SetMCIndex(fWDTHepEmIMC);
        thePrimaryTrack->SetMFP(wdtMFP, 0);
        // g4IMC = fWDTHelper->fWDTCouple->GetIndex();
        // Reset the PE mxsec: set in `G4HepEmGammaManager::GetTotalMacXSec`
        // that might have been called for an other material above (without
        // resulting in interaction). Needed for the interaction type sampling.
        theGammaTrack->SetPEmxSec(wdtPEmxSec);
      }
    }

    if (doStop) {
      // Reached the end, i.e. either interaction happens at the current post
      // step point or it is locating close to the WDT volume boundary
      // (wdtRegionBoundary=true in this case).
      // Update the track with its final position and locate the track properly.
      postStepPoint.SetPosition(r0+wdtStepLength*v0);
      G4TouchableHandle touchableHandle = aTrack.GetTouchableHandle();
      G4TransportationManager::GetTransportationManager()->GetNavigatorForTracking()->LocateGlobalPointAndUpdateTouchableHandle(postStepPoint.GetPosition(), postStepPoint.GetMomentumDirection(), touchableHandle, false);
      aTrack.SetTouchableHandle(touchableHandle);
      aTrack.SetNextTouchableHandle(touchableHandle);
      // Set pre/post step point location and touchable to be the same (as we
      // might have moved the track from a far away volume).
      // NOTE: as energy deposit happens only at discrete interactions for gamma,
      // in case of non-zero energy deposit, i.e. when SD codes are invoked,
      // the track is never on boundary. So all SD code should work fine with
      // identical pre- and post-step points.
      postStepPoint.SetTouchableHandle(touchableHandle);
      preStepPoint.SetTouchableHandle(touchableHandle);
      preStepPoint.SetPosition(postStepPoint.GetPosition());
      // NOTE: some (e.g. calibration) SD codes might be called even in case of zero
      // energy deposit so set the SD pointer in all cases.
      const G4LogicalVolume* postStepLogVol = touchableHandle->GetVolume()->GetLogicalVolume();
      preStepPoint.SetSensitiveDetector(postStepLogVol->GetSensitiveDetector());

      // Set all track properteis needed later: all pre-step point information are
      // actually set to be their post step point values!
      // lvol = aTrack.GetTouchable()->GetVolume()->GetLogicalVolume();
      // MCC  = lvol->GetMaterialCutsCouple();
      preStepPoint.SetMaterial(postStepLogVol->GetMaterial());
      preStepPoint.SetMaterialCutsCouple(postStepLogVol->GetMaterialCutsCouple());

      postStepPoint.SetMaterial(postStepLogVol->GetMaterial());
      postStepPoint.SetMaterialCutsCouple(postStepLogVol->GetMaterialCutsCouple());

      // NOTE: the number of interaction length left will be cleared in all
      // cases when WDT tracking happened (see below).
    }
    // If the WDT region boundary has not been reached in this step then delta
    // interaction happend so just keep moving the post-step point toward the
    // WDT (root) volume boundary.
  };  // END OF WHILE ON WDT

  // Update the time based on the accumulated WDT step length
  // step.SetStepLength(wdtStepLength);
  // aTrack->SetStepLength(wdtStepLength);
  const G4double preStepVelocity = preStepPoint.GetVelocity();
  const G4double deltaTime = preStepVelocity > 0 ? wdtStepLength/preStepVelocity : 0;
  postStepPoint.AddGlobalTime(deltaTime);
  postStepPoint.AddLocalTime(deltaTime);

  // Store the total Woodcock tracking step length in the HepEmTrack field and
  // return with the flag that indicates if boundary has been reached.
  thePrimaryTrack->SetGStepLength(wdtStepLength);

  return isWDTReachedBoundary;
}


void G4HepEmWoodcockHelper::ClearData() {
  // iterate over the `fWDTData` map
  std::map<G4int, WDTDataForARegion*>::iterator itrRegion = fWDTData.begin();
  for (; itrRegion != fWDTData.end(); itrRegion++) {
    // iterate over the root logical volume data inside each reagion
    WDTDataForARegion* dataForARegion = itrRegion->second;
    std::map<G4int, WDTDataPerRootLogVol*>::iterator itrRootVol
        = dataForARegion->fWDTDataRegion.begin();
    for (; itrRootVol != dataForARegion->fWDTDataRegion.end(); itrRootVol++) {
      // NOTE: we do not own the solid and the material-cuts couple (only store their ptr)
      delete itrRootVol->second;
    }
    dataForARegion->fWDTDataRegion.clear();
    delete dataForARegion;
  }
  fWDTData.clear();
}


void G4HepEmWoodcockHelper::FindWDTMaterial(G4LogicalVolume* lvol, double& maxDensity, G4Material** maxDensityMat) {
  G4Material*  mat = lvol->GetMaterial();
  G4double density = mat->GetDensity();
  if (density > maxDensity) {
    maxDensity = density;
    *maxDensityMat = mat;
  }
  // recurse
  int numDaughters = lvol->GetNoDaughters();
  for (int id=0; id<numDaughters; ++id) {
      G4LogicalVolume* lv = lvol->GetDaughter(id)->GetLogicalVolume();
      // detect sub-region and stop if found: Woodcock tracking can be used only in leaf regions
      if (lvol->GetRegion() != lv->GetRegion()) {
        std::cerr << "\n *** G4HepEmWoodcockHelper::FindWDTMaterial \n"
                  << "     Woodcock tracking cannot be applied in this region: " << lvol->GetRegion()->GetName() << "\n"
                  << "     A sub-region has been found: " <<  lv->GetRegion()->GetName() << "\n"
                  << "     Note: Woodcock tracking requires leaf region!\n"
                  << std::endl;
        exit(1);
      }
      FindWDTMaterial(lv, maxDensity, maxDensityMat);
  }
}
