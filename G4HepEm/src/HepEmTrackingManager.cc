#include "HepEmTrackingManager.hh"
#include "TrackingManagerHelper.hh"

#include "G4HepEmCLHEPRandomEngine.hh"
#include "G4HepEmData.hh"
#include "G4HepEmMatCutData.hh"
#include "G4HepEmRunManager.hh"
#include "G4HepEmTLData.hh"

#include "G4HepEmElectronManager.hh"
#include "G4HepEmElectronTrack.hh"
#include "G4HepEmPositronInteractionAnnihilation.hh"
#include "G4HepEmGammaManager.hh"
#include "G4HepEmGammaTrack.hh"

#include "G4MaterialCutsCouple.hh"
#include "G4Step.hh"
#include "G4StepPoint.hh"
#include "G4StepStatus.hh"
#include "G4Threading.hh"
#include "G4Track.hh"

#include "G4SafetyHelper.hh"
#include "G4TransportationManager.hh"

#include "G4EmParameters.hh"
#include "G4ProductionCutsTable.hh"

#include "G4Electron.hh"
#include "G4Gamma.hh"
#include "G4Positron.hh"

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

HepEmTrackingManager::HepEmTrackingManager() {
  fRunManager = new G4HepEmRunManager(G4Threading::IsMasterThread());
  fRandomEngine = new G4HepEmCLHEPRandomEngine(G4Random::getTheEngine());
  fSafetyHelper =
      G4TransportationManager::GetTransportationManager()->GetSafetyHelper();
  fSafetyHelper->InitialiseHelper();
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

HepEmTrackingManager::~HepEmTrackingManager() {
  delete fRunManager;
  delete fRandomEngine;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void HepEmTrackingManager::BuildPhysicsTable(const G4ParticleDefinition &part) {
  if (&part == G4Electron::Definition()) {
    fRunManager->Initialize(fRandomEngine, 0);
  } else if (&part == G4Positron::Definition()) {
    fRunManager->Initialize(fRandomEngine, 1);
  } else if (&part == G4Gamma::Definition()) {
    fRunManager->Initialize(fRandomEngine, 2);
  } else {
    std::cerr
        << " **** ERROR in G4HepEmProcess::BuildPhysicsTable: unknown particle "
        << std::endl;
    exit(-1);
  }
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void HepEmTrackingManager::PreparePhysicsTable(
    const G4ParticleDefinition &part) {
  applyCuts = G4EmParameters::Instance()->ApplyCuts();

  if (applyCuts) {
    auto *theCoupleTable = G4ProductionCutsTable::GetProductionCutsTable();
    theCutsGamma = theCoupleTable->GetEnergyCutsVector(idxG4GammaCut);
    theCutsElectron = theCoupleTable->GetEnergyCutsVector(idxG4ElectronCut);
    theCutsPositron = theCoupleTable->GetEnergyCutsVector(idxG4PositronCut);
  }
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void HepEmTrackingManager::TrackElectron(G4Track *aTrack) {
  TrackingManagerHelper::ChargedNavigation navigation;

  // Prepare for calling the user action.
  auto* evtMgr             = G4EventManager::GetEventManager();
  auto* userTrackingAction = evtMgr->GetUserTrackingAction();
  auto* userSteppingAction = evtMgr->GetUserSteppingAction();

  // Locate the track in geometry.
  {
    auto* transMgr        = G4TransportationManager::GetTransportationManager();
    auto* linearNavigator = transMgr->GetNavigatorForTracking();

    const G4ThreeVector& pos = aTrack->GetPosition();
    const G4ThreeVector& dir = aTrack->GetMomentumDirection();

    // Do not assign directly, doesn't work if the handle is empty.
    G4TouchableHandle touchableHandle;
    if(aTrack->GetTouchableHandle())
    {
      touchableHandle = aTrack->GetTouchableHandle();
      // FIXME: This assumes we only ever have G4TouchableHistorys!
      auto* touchableHistory          = (G4TouchableHistory*) touchableHandle();
      G4VPhysicalVolume* oldTopVolume = touchableHandle->GetVolume();
      G4VPhysicalVolume* newTopVolume =
        linearNavigator->ResetHierarchyAndLocate(pos, dir, *touchableHistory);
      // TODO: WHY?!
      if(newTopVolume != oldTopVolume ||
         oldTopVolume->GetRegularStructureId() == 1)
      {
        touchableHandle = linearNavigator->CreateTouchableHistory();
        aTrack->SetTouchableHandle(touchableHandle);
      }
    }
    else
    {
      linearNavigator->LocateGlobalPointAndSetup(pos, &dir, false, false);
      touchableHandle = linearNavigator->CreateTouchableHistory();
      aTrack->SetTouchableHandle(touchableHandle);
    }
    aTrack->SetNextTouchableHandle(touchableHandle);
  }

  // Prepare data structures used while tracking.
  G4Step step;
  step.NewSecondaryVector();
  G4StepPoint& preStepPoint = *step.GetPreStepPoint();
  G4StepPoint& postStepPoint = *step.GetPostStepPoint();
  step.InitializeStep(aTrack);
  aTrack->SetStep(&step);
  G4TrackVector secondaries;

  // Start of tracking: Inform user and processes.
  if(userTrackingAction)
  {
    userTrackingAction->PreUserTrackingAction(aTrack);
  }

  // === StartTracking ===
  G4HepEmTLData *theTLData = fRunManager->GetTheTLData();
  G4HepEmElectronTrack* theElTrack = theTLData->GetPrimaryElectronTrack();
  G4HepEmTrack *thePrimaryTrack = theElTrack->GetTrack();
  theElTrack->ReSet();
  // In principle, we could continue to use the other generated Gaussian
  // number as long as we are in the same event, but play it safe.
  G4HepEmRandomEngine *rnge = theTLData->GetRNGEngine();
  rnge->DiscardGauss();

  // Pull data structures into local variables.
  G4HepEmData *theHepEmData = fRunManager->GetHepEmData();
  G4HepEmParameters *theHepEmPars = fRunManager->GetHepEmParameters();

  const G4DynamicParticle *theG4DPart = aTrack->GetDynamicParticle();
  const G4int trackID = aTrack->GetTrackID();

  // Init state that never changes for a track.
  const double charge = aTrack->GetParticleDefinition()->GetPDGCharge();
  const bool isElectron = (charge < 0.0);
  thePrimaryTrack->SetCharge(charge);
  // === StartTracking ===

  while(aTrack->GetTrackStatus() == fAlive)
  {
    // Beginning of this step: Prepare data structures.
    aTrack->IncrementCurrentStepNumber();

    step.CopyPostToPreStepPoint();
    step.ResetTotalEnergyDeposit();
    const G4TouchableHandle &touchableHandle = aTrack->GetNextTouchableHandle();
    aTrack->SetTouchableHandle(touchableHandle);

    auto* lvol = aTrack->GetTouchable()->GetVolume()->GetLogicalVolume();
    preStepPoint.SetMaterial(lvol->GetMaterial());
    auto* MCC = lvol->GetMaterialCutsCouple();
    preStepPoint.SetMaterialCutsCouple(lvol->GetMaterialCutsCouple());

    // Query step lengths from pyhsics and geometry, decide on limit.
    // === GetPhysicalInteractionLength ===
    thePrimaryTrack->SetEKin(theG4DPart->GetKineticEnergy(),
                             theG4DPart->GetLogKineticEnergy());

    const int g4IMC = MCC->GetIndex();
    const int hepEmIMC =
        theHepEmData->fTheMatCutData->fG4MCIndexToHepEmMCIndex[g4IMC];
    thePrimaryTrack->SetMCIndex(hepEmIMC);
    bool preStepOnBoundary =
        preStepPoint.GetStepStatus() == G4StepStatus::fGeomBoundary;
    thePrimaryTrack->SetOnBoundary(preStepOnBoundary);
    const double preSafety =
        preStepOnBoundary ? 0.
                   : fSafetyHelper->ComputeSafety(aTrack->GetPosition());
    thePrimaryTrack->SetSafety(preSafety);
    // === HowFar ===
    // Sample the `number-of-interaction-left`
    for (int ip=0; ip<3; ++ip) {
      if (thePrimaryTrack->GetNumIALeft(ip)<=0.) {
        thePrimaryTrack->SetNumIALeft(-G4HepEmLog(rnge->flat()), ip);
      }
    }
    // True distance to discrete interaction.
    G4HepEmElectronManager::HowFarToDiscreteInteraction(theHepEmData, theHepEmPars, theElTrack);
    // Possibly true step limit of MSC, and conversion to geometrical step length.
    G4HepEmElectronManager::HowFarToMSC(theHepEmData, theHepEmPars, theElTrack, rnge);
    // === HowFar ===

    // returns with the geometrcal step length: straight line distance to make
    // along the org direction
    G4double physicalStep = thePrimaryTrack->GetGStepLength();
    // === GetPhysicalInteractionLength ===
    G4double geometryStep = navigation.MakeStep(*aTrack, step, physicalStep);

    bool geometryLimitedStep = geometryStep < physicalStep;
    G4double finalStep = geometryLimitedStep ? geometryStep : physicalStep;

    step.SetStepLength(finalStep);
    aTrack->SetStepLength(finalStep);

    // Call AlongStepDoIt in every step.
    // === AlongStepDoIt ===
    step.UpdateTrack();

    if(aTrack->GetTrackStatus() == fAlive &&
       aTrack->GetKineticEnergy() < DBL_MIN)
    {
      aTrack->SetTrackStatus(fStopAndKill);
    }

    navigation.FinishStep(*aTrack, step);

    // Check if the track left the world.
    if(aTrack->GetNextVolume() == nullptr)
    {
      aTrack->SetTrackStatus(fStopAndKill);
    }

    // The check should rather check for == fAlive and avoid calling
    // PostStepDoIt for fStopButAlive, but the generic stepping loop
    // does it like this...
    if(aTrack->GetTrackStatus() != fStopAndKill)
    {
      const bool postStepOnBoundary =
          postStepPoint.GetStepStatus() == G4StepStatus::fGeomBoundary;

      // NOTE: this primary track is the same as in the last call in the
      // HowFar()
      //       But transportation might changed its direction, geomertical step
      //       length, or status ( on boundary or not).
      const G4ThreeVector &primDir = theG4DPart->GetMomentumDirection();
      thePrimaryTrack->SetDirection(primDir[0], primDir[1], primDir[2]);
      thePrimaryTrack->SetGStepLength(finalStep);
      thePrimaryTrack->SetOnBoundary(postStepOnBoundary);
      // invoke the physics interactions (all i.e. all along- and post-step as
      // well as possible at rest)
      // === Perform ===
      // Set default values to cover all early returns due to protection against
      // zero step lengths, conversion errors, etc.
      thePrimaryTrack->SetEnergyDeposit(0);
      theElTrack->SetPStepLength(finalStep);
      if (finalStep > 0) {
        theElTrack->SavePreStepEKin();
        // === PerformContinuous ===
        bool stopped = false;
        do {
          //
          // === 1. MSC should be invoked to obtain the physics step Length
          G4HepEmElectronManager::UpdatePStepLength(theElTrack);
          const double pStepLength = theElTrack->GetPStepLength();

          if (pStepLength<=0.0) {
            break;
          }
          // compute the energy loss first based on the new step length: it will be needed in the
          // MSC scatteirng and displacement computation here as well (that is done only if not
          // the last step with the particle).
          // But update the number of interaction length left before.
          //
          // === 2. The `number-of-interaction-left` needs to be updated based on the actual
          //        physical step Length
          G4HepEmElectronManager::UpdateNumIALeft(theElTrack);
          //
          // === 3. Continuous energy loss needs to be computed
          stopped = G4HepEmElectronManager::ApplyMeanEnergyLoss(theHepEmData, theHepEmPars, theElTrack);
          if (stopped) {
            break;
          }

          // === 4. Sample MSC direction change and displacement.
          G4HepEmElectronManager::SampleMSC(theHepEmData, theHepEmPars, theElTrack, rnge);

          // === 5. Sample loss fluctuations.
          stopped = G4HepEmElectronManager::SampleLossFluctuations(theHepEmData, theHepEmPars, theElTrack, rnge);
        } while (0);
        // === PerformContinuous ===

        if (stopped) {
          // call annihilation for e+ !!!
          if (!isElectron) {
            G4HepEmPositronInteractionAnnihilation::Perform(theTLData, true);
          }
        } else {
          // === 4. Discrete part of the interaction (if any)
          G4HepEmElectronManager::PerformDiscrete(theHepEmData, theHepEmPars, theTLData);
        }
      }
      // === Perform ===
      step.SetStepLength(theElTrack->GetPStepLength());

      // energy, e-depo, momentum direction and status
      const double ekin = thePrimaryTrack->GetEKin();
      double edep = thePrimaryTrack->GetEnergyDeposit();
      postStepPoint.SetKineticEnergy(ekin);
      if (ekin <= 0.0) {
        aTrack->SetTrackStatus(fStopAndKill);
      }
      const double *pdir = thePrimaryTrack->GetDirection();
      postStepPoint.SetMomentumDirection(
          G4ThreeVector(pdir[0], pdir[1], pdir[2]));

      // apply MSC displacement if its length is longer than a minimum and we
      // are not on boundary
      G4ThreeVector position = postStepPoint.GetPosition();
      if (!postStepOnBoundary) {
        const double *displacement =
            theElTrack->GetMSCTrackData()->GetDisplacement();
        const double dLength2 = displacement[0] * displacement[0] +
                                displacement[1] * displacement[1] +
                                displacement[2] * displacement[2];
        const double kGeomMinLength = 5.0e-8; // 0.05 [nm]
        const double kGeomMinLength2 =
            kGeomMinLength * kGeomMinLength; // (0.05 [nm])^2
        if (dLength2 > kGeomMinLength2) {
          // apply displacement
          bool isPositionChanged = true;
          const double dispR = std::sqrt(dLength2);
          const double postSafety =
              0.99 * fSafetyHelper->ComputeSafety(position, dispR);
          const G4ThreeVector theDisplacement(displacement[0], displacement[1],
                                              displacement[2]);
          // far away from geometry boundary
          if (postSafety > 0.0 && dispR <= postSafety) {
            position += theDisplacement;
            // near the boundary
          } else {
            // displaced point is definitely within the volume
            if (dispR < postSafety) {
              position += theDisplacement;
              // reduced displacement
            } else if (postSafety > kGeomMinLength) {
              position += theDisplacement * (postSafety / dispR);
              // very small postSafety
            } else {
              isPositionChanged = false;
            }
          }
          if (isPositionChanged) {
            fSafetyHelper->ReLocateWithinVolume(position);
            postStepPoint.SetPosition(position);
          }
        }
      }

      step.UpdateTrack();

      // secondary: only possible is e- or gamma at the moemnt
      const int numSecElectron = theTLData->GetNumSecondaryElectronTrack();
      const int numSecGamma = theTLData->GetNumSecondaryGammaTrack();
      const int numSecondaries = numSecElectron + numSecGamma;
      if (numSecondaries > 0) {
        const G4ThreeVector &theG4PostStepPointPosition = position;
        const G4double theG4PostStepGlobalTime =
            postStepPoint.GetGlobalTime();
        for (int is = 0; is < numSecElectron; ++is) {
          G4HepEmTrack *secTrack =
              theTLData->GetSecondaryElectronTrack(is)->GetTrack();
          const double secEKin = secTrack->GetEKin();
          const bool isElectron = secTrack->GetCharge() < 0.0;
          if (applyCuts) {
            if (isElectron && secEKin < (*theCutsElectron)[g4IMC]) {
              edep += secEKin;
              continue;
            } else if (!isElectron &&
                       CLHEP::electron_mass_c2 < (*theCutsGamma)[g4IMC] &&
                       secEKin < (*theCutsPositron)[g4IMC]) {
              edep += secEKin + 2 * CLHEP::electron_mass_c2;
              continue;
            }
          }

          const double *dir = secTrack->GetDirection();
          const G4ParticleDefinition *partDef = G4Electron::Definition();
          if (!isElectron) {
            partDef = G4Positron::Definition();
          }
          G4DynamicParticle *dp = new G4DynamicParticle(
              partDef, G4ThreeVector(dir[0], dir[1], dir[2]), secEKin);
          G4Track *aG4Track = new G4Track(dp, theG4PostStepGlobalTime,
                                          theG4PostStepPointPosition);
          aG4Track->SetParentID(trackID);
          aG4Track->SetTouchableHandle(touchableHandle);
          secondaries.push_back(aG4Track);
        }
        theTLData->ResetNumSecondaryElectronTrack();

        for (int is = 0; is < numSecGamma; ++is) {
          G4HepEmTrack *secTrack =
              theTLData->GetSecondaryGammaTrack(is)->GetTrack();
          const double secEKin = secTrack->GetEKin();
          if (applyCuts && secEKin < (*theCutsGamma)[g4IMC]) {
            edep += secEKin;
            continue;
          }

          const double *dir = secTrack->GetDirection();
          G4DynamicParticle *dp = new G4DynamicParticle(
              G4Gamma::Definition(), G4ThreeVector(dir[0], dir[1], dir[2]),
              secEKin);
          G4Track *aG4Track = new G4Track(dp, theG4PostStepGlobalTime,
                                          theG4PostStepPointPosition);
          aG4Track->SetParentID(trackID);
          aG4Track->SetTouchableHandle(touchableHandle);
          secondaries.push_back(aG4Track);
        }
        theTLData->ResetNumSecondaryGammaTrack();
      }

      step.AddTotalEnergyDeposit(edep);
    }

    // Need to get the true step length, not the geometry step length!
    aTrack->AddTrackLength(step.GetStepLength());

    // End of this step: Call sensitive detector and stepping actions.
    if(step.GetControlFlag() != AvoidHitInvocation)
    {
      auto* sensitive = lvol->GetSensitiveDetector();
      if(sensitive)
      {
        sensitive->Hit(&step);
      }
    }

    if(userSteppingAction)
    {
      userSteppingAction->UserSteppingAction(&step);
    }

    auto* regionalAction = lvol->GetRegion()->GetRegionalSteppingAction();
    if(regionalAction)
    {
      regionalAction->UserSteppingAction(&step);
    }
  }

  // End of tracking: Inform processes and user.
  // === EndTracking ===

  if(userTrackingAction)
  {
    userTrackingAction->PostUserTrackingAction(aTrack);
  }

  evtMgr->StackTracks(&secondaries);

  step.DeleteSecondaryVector();
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void HepEmTrackingManager::TrackGamma(G4Track *aTrack) {
  class GammaPhysics final : public TrackingManagerHelper::Physics {
  public:
    GammaPhysics(HepEmTrackingManager &mgr) : fMgr(mgr) {}

    void StartTracking(G4Track *aTrack) override {
      fMgr.fRunManager->GetTheTLData()->GetPrimaryGammaTrack()->ReSet();
      // In principle, we could continue to use the other generated Gaussian
      // number as long as we are in the same event, but play it safe.
      fMgr.fRunManager->GetTheTLData()->GetRNGEngine()->DiscardGauss();
    }

    G4double GetPhysicalInteractionLength(const G4Track &track) override {
      G4HepEmTLData *theTLData = fMgr.fRunManager->GetTheTLData();
      G4HepEmTrack *thePrimaryTrack =
          theTLData->GetPrimaryGammaTrack()->GetTrack();
      G4HepEmData *theHepEmData = fMgr.fRunManager->GetHepEmData();
      thePrimaryTrack->SetCharge(0);
      const G4DynamicParticle *theG4DPart = track.GetDynamicParticle();
      thePrimaryTrack->SetEKin(theG4DPart->GetKineticEnergy(),
                               theG4DPart->GetLogKineticEnergy());

      const int g4IMC = track.GetMaterialCutsCouple()->GetIndex();
      const int hepEmIMC =
          theHepEmData->fTheMatCutData->fG4MCIndexToHepEmMCIndex[g4IMC];
      thePrimaryTrack->SetMCIndex(hepEmIMC);
      const G4StepPoint *theG4PreStepPoint = track.GetStep()->GetPreStepPoint();
      thePrimaryTrack->SetOnBoundary(theG4PreStepPoint->GetStepStatus() ==
                                     G4StepStatus::fGeomBoundary);
      G4HepEmGammaManager::HowFar(
          theHepEmData, fMgr.fRunManager->GetHepEmParameters(), theTLData);
      // returns with the geometrcal step length: straight line distance to make
      // along the org direction
      return thePrimaryTrack->GetGStepLength();
    }

    void AlongStepDoIt(G4Track &track, G4Step &step, G4TrackVector &) override {
      // Nothing to do here!
    }

    void PostStepDoIt(G4Track &track, G4Step &step,
                      G4TrackVector &secondaries) override {
      G4HepEmTLData *theTLData = fMgr.fRunManager->GetTheTLData();
      G4StepPoint *theG4PostStepPoint = step.GetPostStepPoint();
      const bool onBoundary =
          theG4PostStepPoint->GetStepStatus() == G4StepStatus::fGeomBoundary;
      G4HepEmTrack *thePrimaryTrack =
          theTLData->GetPrimaryGammaTrack()->GetTrack();

      if (onBoundary) {
        thePrimaryTrack->SetGStepLength(track.GetStepLength());
        G4HepEmGammaManager::UpdateNumIALeft(thePrimaryTrack);
        return;
      }
      // NOTE: this primary track is the same as in the last call in the
      // HowFar()
      //       But transportation might changed its direction, geomertical step
      //       length, or status ( on boundary or not).
      const G4ThreeVector &primDir =
          track.GetDynamicParticle()->GetMomentumDirection();
      thePrimaryTrack->SetDirection(primDir[0], primDir[1], primDir[2]);
      thePrimaryTrack->SetGStepLength(track.GetStepLength());
      thePrimaryTrack->SetOnBoundary(onBoundary);
      // invoke the physics interactions (all i.e. all along- and post-step as
      // well as possible at rest)
      G4HepEmGammaManager::Perform(fMgr.fRunManager->GetHepEmData(),
                                   fMgr.fRunManager->GetHepEmParameters(),
                                   theTLData);

      // energy, e-depo, momentum direction and status
      const double ekin = thePrimaryTrack->GetEKin();
      double edep = thePrimaryTrack->GetEnergyDeposit();
      theG4PostStepPoint->SetKineticEnergy(ekin);
      if (ekin <= 0.0) {
        track.SetTrackStatus(fStopAndKill);
      }
      const double *pdir = thePrimaryTrack->GetDirection();
      theG4PostStepPoint->SetMomentumDirection(
          G4ThreeVector(pdir[0], pdir[1], pdir[2]));

      step.UpdateTrack();

      const int g4IMC =
          step.GetPreStepPoint()->GetMaterialCutsCouple()->GetIndex();
      // secondary: only possible is e- or gamma at the moemnt
      const int numSecElectron = theTLData->GetNumSecondaryElectronTrack();
      const int numSecGamma = theTLData->GetNumSecondaryGammaTrack();
      const int numSecondaries = numSecElectron + numSecGamma;
      if (numSecondaries > 0) {
        const G4ThreeVector &theG4PostStepPointPosition =
            theG4PostStepPoint->GetPosition();
        const G4double theG4PostStepGlobalTime =
            theG4PostStepPoint->GetGlobalTime();
        const G4TouchableHandle &theG4TouchableHandle =
            track.GetTouchableHandle();
        for (int is = 0; is < numSecElectron; ++is) {
          G4HepEmTrack *secTrack =
              theTLData->GetSecondaryElectronTrack(is)->GetTrack();
          const double secEKin = secTrack->GetEKin();
          const bool isElectron = secTrack->GetCharge() < 0.0;
          if (fMgr.applyCuts) {
            if (isElectron && secEKin < (*fMgr.theCutsElectron)[g4IMC]) {
              edep += secEKin;
              continue;
            } else if (!isElectron &&
                       CLHEP::electron_mass_c2 < (*fMgr.theCutsGamma)[g4IMC] &&
                       secEKin < (*fMgr.theCutsPositron)[g4IMC]) {
              edep += secEKin + 2 * CLHEP::electron_mass_c2;
              continue;
            }
          }

          const double *dir = secTrack->GetDirection();
          const G4ParticleDefinition *partDef = G4Electron::Definition();
          if (!isElectron) {
            partDef = G4Positron::Definition();
          }
          G4DynamicParticle *dp = new G4DynamicParticle(
              partDef, G4ThreeVector(dir[0], dir[1], dir[2]), secEKin);
          G4Track *aG4Track = new G4Track(dp, theG4PostStepGlobalTime,
                                          theG4PostStepPointPosition);
          aG4Track->SetParentID(track.GetTrackID());
          aG4Track->SetTouchableHandle(theG4TouchableHandle);
          secondaries.push_back(aG4Track);
        }
        theTLData->ResetNumSecondaryElectronTrack();

        for (int is = 0; is < numSecGamma; ++is) {
          G4HepEmTrack *secTrack =
              theTLData->GetSecondaryGammaTrack(is)->GetTrack();
          const double secEKin = secTrack->GetEKin();
          if (fMgr.applyCuts && secEKin < (*fMgr.theCutsGamma)[g4IMC]) {
            edep += secEKin;
            continue;
          }

          const double *dir = secTrack->GetDirection();
          G4DynamicParticle *dp = new G4DynamicParticle(
              G4Gamma::Definition(), G4ThreeVector(dir[0], dir[1], dir[2]),
              secEKin);
          G4Track *aG4Track = new G4Track(dp, theG4PostStepGlobalTime,
                                          theG4PostStepPointPosition);
          aG4Track->SetParentID(track.GetTrackID());
          aG4Track->SetTouchableHandle(theG4TouchableHandle);
          secondaries.push_back(aG4Track);
        }
        theTLData->ResetNumSecondaryGammaTrack();
      }

      step.AddTotalEnergyDeposit(edep);
    }

  private:
    HepEmTrackingManager &fMgr;
  };

  GammaPhysics physics(*this);
  TrackingManagerHelper::TrackNeutralParticle(aTrack, physics);
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void HepEmTrackingManager::HandOverOneTrack(G4Track *aTrack) {
  const G4ParticleDefinition *part = aTrack->GetParticleDefinition();

  if (part == G4Electron::Definition() || part == G4Positron::Definition()) {
    TrackElectron(aTrack);
  } else if (part == G4Gamma::Definition()) {
    TrackGamma(aTrack);
  }

  aTrack->SetTrackStatus(fStopAndKill);
  delete aTrack;
}
