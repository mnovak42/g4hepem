#include "G4HepEmTrackingManager.hh"
#include "TrackingManagerHelper.hh"

#include "G4HepEmNoProcess.hh"

#include "G4HepEmRandomEngine.hh"
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
#include "G4TrackingManager.hh"
#include "G4VTrajectory.hh"

#include "G4SafetyHelper.hh"
#include "G4TransportationManager.hh"

#include "G4EmParameters.hh"
#include "G4ProductionCutsTable.hh"

#include "G4VProcess.hh"
#include "G4EmProcessSubType.hh"
#include "G4HadronicProcessType.hh"
#include "G4ProcessType.hh"
#include "G4TransportationProcessType.hh"

#include "G4RegionStore.hh"

#include "G4Electron.hh"
#include "G4Gamma.hh"
#include "G4Positron.hh"


//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

G4HepEmTrackingManager::G4HepEmTrackingManager() {
  fRunManager = new G4HepEmRunManager(G4Threading::IsMasterThread());
  fRandomEngine = new G4HepEmRandomEngine(G4Random::getTheEngine());
  fSafetyHelper =
      G4TransportationManager::GetTransportationManager()->GetSafetyHelper();
  fSafetyHelper->InitialiseHelper();
  fStep = new G4Step;
  fStep->NewSecondaryVector();

  // Construct fake G4VProcess-es with the proper name and indices matching the
  // hepEm process indices
  fElectronNoProcessVector.push_back(
      new G4HepEmNoProcess("eIoni", G4ProcessType::fElectromagnetic,
                           G4EmProcessSubType::fIonisation));
  fElectronNoProcessVector.push_back(
      new G4HepEmNoProcess("eBrem", G4ProcessType::fElectromagnetic,
                           G4EmProcessSubType::fBremsstrahlung));
  fElectronNoProcessVector.push_back(
      new G4HepEmNoProcess("annihl", G4ProcessType::fElectromagnetic,
                           G4EmProcessSubType::fAnnihilation));
  fElectronNoProcessVector.push_back(
      new G4HepEmNoProcess("msc", G4ProcessType::fElectromagnetic,
                           G4EmProcessSubType::fMultipleScattering));
  fElectronNoProcessVector.push_back(
      new G4HepEmNoProcess("electronNuclear", G4ProcessType::fHadronic,
                           G4HadronicProcessType::fHadronInelastic));
  fElectronNoProcessVector.push_back(
      new G4HepEmNoProcess("positronNuclear", G4ProcessType::fHadronic,
                           G4HadronicProcessType::fHadronInelastic));

  fGammaNoProcessVector.push_back(
      new G4HepEmNoProcess("conv", G4ProcessType::fElectromagnetic,
                           G4EmProcessSubType::fGammaConversion));
  fGammaNoProcessVector.push_back(
      new G4HepEmNoProcess("compt", G4ProcessType::fElectromagnetic,
                           G4EmProcessSubType::fComptonScattering));
  fGammaNoProcessVector.push_back(
      new G4HepEmNoProcess("phot", G4ProcessType::fElectromagnetic,
                           G4EmProcessSubType::fPhotoElectricEffect));
  fGammaNoProcessVector.push_back(
      new G4HepEmNoProcess("photonNuclear", G4ProcessType::fHadronic,
                           G4HadronicProcessType::fHadronInelastic));

  fTransportNoProcess = new G4HepEmNoProcess(
      "Transportation", G4ProcessType::fTransportation, TRANSPORTATION);

  // Init the gamma-nuclear process
  fGNucProcess = nullptr;
  // Init the electron/positron-nuclear processes
  fENucProcess = nullptr;
  fPNucProcess = nullptr;

  // Init the fast sim mamanger process ptrs of the 3 particles
  fFastSimProcess[0] = nullptr;
  fFastSimProcess[1] = nullptr;
  fFastSimProcess[2] = nullptr;

  // ATLAS XTR RELATED:
  // Init the ATLAS specific transition radiation process related ptrs
  // NOTE: they stay `nullptr` if used outside ATLAS Athena causing no harm
  fXTRRegion  = nullptr;
  fXTRProcess = nullptr;

}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

G4HepEmTrackingManager::~G4HepEmTrackingManager() {
  // Per behaviour in Physics Constructors, we do not delete the G4HepEmNoProcess
  // instances as these are owned by G4ProcessTable.
  delete fRunManager;
  delete fRandomEngine;
  delete fStep;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void G4HepEmTrackingManager::BuildPhysicsTable(const G4ParticleDefinition &part) {
  if (&part == G4Electron::Definition()) {
    int particleID = 0;
    fRunManager->Initialize(fRandomEngine, particleID);
    // Find the electron-nuclear process if has been attached
    InitNuclearProcesses(particleID);
    // Find the fast simulation manager process for e- (if has been attached)
    InitFastSimRelated(particleID);
    // Find the ATLAS specific trans. rad. (XTR) process (if has been attached)
    InitXTRRelated();
  } else if (&part == G4Positron::Definition()) {
    int particleID = 1;
    fRunManager->Initialize(fRandomEngine, particleID);
    // Find the positron-nuclear process if has been attached
    InitNuclearProcesses(particleID);
    // Find the fast simulation manager process for e+ (if has been attached)
    InitFastSimRelated(particleID);
  } else if (&part == G4Gamma::Definition()) {
    int particleID = 2;
    fRunManager->Initialize(fRandomEngine, particleID);
    // Find the gamma-nuclear process if has been attached
    InitNuclearProcesses(particleID);
    // Find the fast simulation manager process for gamma (if has been attached)
    InitFastSimRelated(particleID);
  } else {
    std::cerr
        << " **** ERROR in G4HepEmProcess::BuildPhysicsTable: unknown particle "
        << std::endl;
    exit(-1);
  }
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void G4HepEmTrackingManager::PreparePhysicsTable(
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

void G4HepEmTrackingManager::TrackElectron(G4Track *aTrack) {
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

  // Set OriginTouchableHandle for primary track(set at stacking for secondaries)
  if (aTrack->GetParentID() == 0) {
    aTrack->SetOriginTouchableHandle(aTrack->GetTouchableHandle());
  }

  // Set vertex information: in normal tracking this is done in
  //  `G4TrackingManager::ProcessOneTrack` when calling
  //  `G4SteppingManager::SetInitialStep`
  if (aTrack->GetCurrentStepNumber() == 0) {
    aTrack->SetVertexPosition(aTrack->GetPosition());
    aTrack->SetVertexMomentumDirection(aTrack->GetMomentumDirection());
    aTrack->SetVertexKineticEnergy(aTrack->GetKineticEnergy());
    aTrack->SetLogicalVolumeAtVertex(aTrack->GetVolume()->GetLogicalVolume());
  }

  // Prepare data structures used while tracking.
  G4Step &step = *fStep;
  G4TrackVector& secondaries = *step.GetfSecondary();
  G4StepPoint& preStepPoint = *step.GetPreStepPoint();
  G4StepPoint& postStepPoint = *step.GetPostStepPoint();
  step.InitializeStep(aTrack);
  aTrack->SetStep(&step);

  // Start of tracking: Inform user and processes.
  if(userTrackingAction)
  {
    userTrackingAction->PreUserTrackingAction(aTrack);
  }

  // Store the trajectory only if the user requested in the G4TrackingManager
  // and set their own trajectory object (usually in the PreUserTrackingAction).
  G4TrackingManager* trMgr = evtMgr->GetTrackingManager();
  G4VTrajectory* theTrajectory = trMgr->GetStoreTrajectory() == 0
                                 ? nullptr : trMgr->GimmeTrajectory();

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

  // Init state that never changes for a track.
  const double charge = aTrack->GetParticleDefinition()->GetPDGCharge();
  const bool isElectron = (charge < 0.0);
  thePrimaryTrack->SetCharge(charge);

  // Invoke the fast simulation manager process StartTracking interface (if any)
  G4VProcess* fFastSimProc = isElectron ? fFastSimProcess[0] : fFastSimProcess[1];
  if (fFastSimProc != nullptr) {
    fFastSimProc->StartTracking(aTrack);
  }

  // Invoke the electron/positron-nuclear process start tracking interace (if any)
  G4VProcess* theNucProcess = isElectron ? fENucProcess : fPNucProcess;
  if (theNucProcess != nullptr) {
    theNucProcess->StartTracking(aTrack);
  }

  // === StartTracking ===

  while(aTrack->GetTrackStatus() == fAlive)
  {
    // Beginning of this step: Prepare data structures.
    aTrack->IncrementCurrentStepNumber();

    step.CopyPostToPreStepPoint();
    step.ResetTotalEnergyDeposit();
    step.SetControlFlag(G4SteppingControl::NormalCondition);
    const G4TouchableHandle &touchableHandle = aTrack->GetNextTouchableHandle();
    aTrack->SetTouchableHandle(touchableHandle);

    auto* lvol = aTrack->GetTouchable()->GetVolume()->GetLogicalVolume();
    preStepPoint.SetMaterial(lvol->GetMaterial());
    auto* MCC = lvol->GetMaterialCutsCouple();
    preStepPoint.SetMaterialCutsCouple(lvol->GetMaterialCutsCouple());

    // Call the fast simulation manager process if any and check if any fast
    // sim models have been triggered (in that case, returns zero proposed
    // step length and `ExclusivelyForced` forced condition i.e. this and only
    // this process is ivnvoked.
    G4ForceCondition fastSimCondition;
    if (fFastSimProc != nullptr && fFastSimProc->PostStepGetPhysicalInteractionLength(*aTrack, 0.0, &fastSimCondition) == 0.0) {
      // Fast simulation active: invoke its PostStepDoIt, update properties and continue
      postStepPoint.SetProcessDefinedStep(fFastSimProc);
      G4VParticleChange* fastSimParticleChange = fFastSimProc->PostStepDoIt(*aTrack, step);
      fastSimParticleChange->UpdateStepForPostStep(&step);
      step.UpdateTrack();
      aTrack->SetTrackStatus(fastSimParticleChange->GetTrackStatus());
      // NOTE: it's assumed that fast simulation models do NOT produce secondaries
      //       (this can be improved and secondaries might be stacked here later if any)
      fastSimParticleChange->Clear();
      // End of this step: call the SD codes and required actions
      // NOTE: fast simulation sets the step control flag to `AvoidHitInvocation`
      //       as the fast sim models invoke their hit processing inside. So usually
      //       there is no hit processing here.
      if (step.GetControlFlag() != AvoidHitInvocation) {
        auto* sensitive = lvol->GetSensitiveDetector();
        if (sensitive) {
          sensitive->Hit(&step);
        }
      }
      if (userSteppingAction) {
        userSteppingAction->UserSteppingAction(&step);
      }
      auto* regionalAction = lvol->GetRegion()->GetRegionalSteppingAction();
      if (regionalAction) {
        regionalAction->UserSteppingAction(&step);
      }
      // Append the trajectory if a trajectory was set by the user.
      if (theTrajectory != nullptr) {
        theTrajectory->AppendStep(&step);
      }
      continue;
    }

    // Query step lengths from pyhsics and geometry, decide on limit.
    const G4double preStepEkin = theG4DPart->GetKineticEnergy();
    const G4double preStepLogEkin = theG4DPart->GetLogKineticEnergy();
    thePrimaryTrack->SetEKin(preStepEkin, preStepLogEkin);

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

    // Sample the `number-of-interaction-left`
    for (int ip=0; ip<4; ++ip) {
      if (thePrimaryTrack->GetNumIALeft(ip)<=0.) {
        thePrimaryTrack->SetNumIALeft(-G4HepEmLog(rnge->flat()), ip);
      }
    }
    // True distance to discrete interaction.
    G4HepEmElectronManager::HowFarToDiscreteInteraction(theHepEmData, theHepEmPars, theElTrack);
    // Remember which process was selected - MSC might limit the sub-steps.
    const int iDProc = thePrimaryTrack->GetWinnerProcessIndex();

    double stepLimitLeft = theElTrack->GetPStepLength();
    double totalTruePathLength = 0, totalEloss = 0;
    bool continueStepping = fMultipleSteps, stopped = false;

    theElTrack->SavePreStepEKin();

    do {
      // Possibly true step limit of MSC, and conversion to geometrical step length.
      G4HepEmElectronManager::HowFarToMSC(theHepEmData, theHepEmPars, theElTrack, rnge);
      if (thePrimaryTrack->GetWinnerProcessIndex() != -2) {
        // If MSC did not limit the step, exit the loop after this iteration.
        continueStepping = false;
      }

      // Get the geometrcal step length: straight line distance to make along the
      // original direction.
      G4double physicalStep = thePrimaryTrack->GetGStepLength();
      G4double geometryStep = navigation.MakeStep(*aTrack, step, physicalStep);

      bool geometryLimitedStep = geometryStep < physicalStep;
      G4double finalStep = geometryLimitedStep ? geometryStep : physicalStep;

      step.UpdateTrack();

      navigation.FinishStep(*aTrack, step);

      if (geometryLimitedStep) {
        continueStepping = false;
        // Check if the track left the world.
        if(aTrack->GetNextVolume() == nullptr)
        {
          aTrack->SetTrackStatus(fStopAndKill);
          break;
        }
      }

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

      if (finalStep > 0) {
        do {
          //
          // === 1. MSC should be invoked to obtain the physics step Length
          G4HepEmElectronManager::UpdatePStepLength(theElTrack);
          const double pStepLength = theElTrack->GetPStepLength();
          totalTruePathLength += pStepLength;

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
          totalEloss += thePrimaryTrack->GetEnergyDeposit();
          if (stopped) {
            continueStepping = false;
            break;
          }

          // === 4. Sample MSC direction change and displacement.
          G4HepEmElectronManager::SampleMSC(theHepEmData, theHepEmPars, theElTrack, rnge);

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

        } while (0);

        if (continueStepping) {
          // Reset the energy deposit, we're accumulating it in totalEloss.
          thePrimaryTrack->SetEnergyDeposit(0);
          // Also reset the selected winner process that MSC replaced.
          thePrimaryTrack->SetWinnerProcessIndex(iDProc);

          // Save the current energy for the next MSC invocation.
          postStepPoint.SetKineticEnergy(thePrimaryTrack->GetEKin());
          theElTrack->SavePreStepEKin();
          // Apply everything to the track, so that navigation sees it.
          step.UpdateTrack();

          // Subtract the path length we just traveled, and set this as the next
          // attempted sub-step.
          const double pStepLength = theElTrack->GetPStepLength();
          stepLimitLeft -= pStepLength;
          theElTrack->SetPStepLength(stepLimitLeft);
          thePrimaryTrack->SetGStepLength(stepLimitLeft);

          // Also reduce the range accordingly.
          const double range = theElTrack->GetRange() - pStepLength;
          theElTrack->SetRange(range);
        }

      }
    } while (continueStepping);

    // Restore the total (mean) energy loss accumulated along the sub-steps.
    thePrimaryTrack->SetEnergyDeposit(totalEloss);

    if (!stopped) {
      // If not already stopped, restore the pre-step energy and sample loss
      // fluctuations.
      theElTrack->SetPreStepEKin(preStepEkin, preStepLogEkin);
      stopped = G4HepEmElectronManager::SampleLossFluctuations(theHepEmData, theHepEmPars, theElTrack, rnge);
    }

    // ATLAS XTR RELATED:
    // For the XTR process (if any)
    G4VParticleChange* particleChangeXTR = nullptr;

    // Set the true (possible accumulated) step length here as some G4 processes
    // (e.g. XTR) might need below in their DoIt
    step.SetStepLength(totalTruePathLength);

    const G4VProcess *proc = nullptr;
    if (stopped) {
      // call annihilation for e+ !!!
      if (!isElectron) {
        G4HepEmPositronInteractionAnnihilation::Perform(theTLData, true);
        proc = fElectronNoProcessVector[2];
      } else {
        // otherwise ionization limited the step
        proc = fElectronNoProcessVector[0];
      }
    } else if (aTrack->GetTrackStatus() != fStopAndKill) {
      // === 4. Discrete part of the interaction (if any)
      const int iDProc = thePrimaryTrack->GetWinnerProcessIndex();
      if (thePrimaryTrack->GetOnBoundary()) {
        // no disrete interaction in this case
        proc = fTransportNoProcess;
      } else if (iDProc != 3) {
        // interactions handled by the HepEm physics: ioni, brem or annihilation (for e+)
        G4HepEmElectronManager::PerformDiscrete(theHepEmData, theHepEmPars, theTLData);
        const double *pdir = thePrimaryTrack->GetDirection();
        postStepPoint.SetMomentumDirection( G4ThreeVector(pdir[0], pdir[1], pdir[2]) );
        // Get the final process defining the step - might still be MSC!
        if (iDProc == -1) {
          // ionization
          proc = fElectronNoProcessVector[0];
        } else if (iDProc == -2) {
          proc = fElectronNoProcessVector[3];
        } else {
          proc = fElectronNoProcessVector[iDProc];
        }
      } else {
        // Electron/positron-nuclear: --> use Geant4 for the interaction:
        proc = isElectron ? fElectronNoProcessVector[4] : fElectronNoProcessVector[5];
        thePrimaryTrack->SetNumIALeft(-1.0, iDProc);
        // check if delta interaction happens
        // Invoke the electron/positron-nuclear interaction using the Geant4 process
        G4VParticleChange* particleChangeNuc = nullptr;
        if (theNucProcess != nullptr && !G4HepEmElectronManager::CheckDelta(theHepEmData, thePrimaryTrack, theTLData->GetRNGEngine()->flat())) {
          // call to set some fields of the process like material, energy etc.. used in its DoIt
          G4ForceCondition forceCondition;
          theNucProcess->PostStepGetPhysicalInteractionLength(*aTrack, 0.0, &forceCondition);
          //
          postStepPoint.SetStepStatus(fPostStepDoItProc);
          particleChangeNuc = theNucProcess->PostStepDoIt(*aTrack, step);
          // update the track and stack according to the result of the interaction
          particleChangeNuc->UpdateStepForPostStep(&step);
          step.UpdateTrack();
          aTrack->SetTrackStatus(particleChangeNuc->GetTrackStatus());
          // need to add secondaries to the secondary vector of the current track
          // NOTE: as we use Geant4, we should care only those changes that are
          //   not included in the above update step and track, i.e. the energy
          //   deposited due to applying the cut when stacking the secondaries
          thePrimaryTrack->AddEnergyDeposit( StackG4Secondaries(particleChangeNuc, aTrack, proc, g4IMC) );
          // done: clear the particle change
          particleChangeNuc->Clear();
          // update the primary track kinetic energy and direction
          thePrimaryTrack->SetEKin(aTrack->GetKineticEnergy());
          const G4ThreeVector& dir = aTrack->GetMomentumDirection();
          thePrimaryTrack->SetDirection(dir.x(), dir.y(), dir.z());
        }
      }

      // ATLAS XTR RELATED:
      // Invoke the TRTTransitionRadiation process for e-/e+ fXTRProcess:
      // But only if the step was done in the Radiator region with energy > 255 MeV (m_EkinMin)
      if (fXTRProcess != nullptr && fXTRRegion == preStepPoint.GetPhysicalVolume()->GetLogicalVolume()->GetRegion() && thePrimaryTrack->GetEKin() > 255.0) {
        // the TRTTransitionRadiation process might create photons as secondary
        // and changes the primary e-/e+ energy only but nothing more than that
        // requires: kinetic energy and momentum direction from the track (dynamic part.) and logical volume
        //           step length and post step point position from the step
        // all these are up-to-date except the kinetic energy and momentum direction of the dynamic particle
        G4DynamicParticle* mutableG4DPart = const_cast<G4DynamicParticle*>(theG4DPart);
        mutableG4DPart->SetKineticEnergy(thePrimaryTrack->GetEKin());
        mutableG4DPart->SetMomentumDirection(postStepPoint.GetMomentumDirection());
        particleChangeXTR = fXTRProcess->PostStepDoIt(*aTrack, step);
        if (particleChangeXTR->GetNumberOfSecondaries() > 0) {
          thePrimaryTrack->SetEKin(static_cast<G4ParticleChange*>(particleChangeXTR)->GetEnergy());
        }
      }
      // === End discrete part of the interaction (if any)

    } else {
      // Else the particle left the world.
      proc = fTransportNoProcess;
    }

    postStepPoint.SetProcessDefinedStep(proc);

    // energy, e-depo and status
    const double ekin = thePrimaryTrack->GetEKin();
    double edep = thePrimaryTrack->GetEnergyDeposit();
    postStepPoint.SetKineticEnergy(ekin);
    if (ekin <= 0.0) {
      aTrack->SetTrackStatus(fStopAndKill);
    }
    step.UpdateTrack();

    // Stack secondaries created by the HepEm physics above
    edep += StackSecondaries(theTLData, aTrack, proc, g4IMC);

    // ATLAS XTR RELATED:
    // Stack XTR secondaries (if any)
    if (particleChangeXTR != nullptr) {
      const int numXTRPhotons = particleChangeXTR->GetNumberOfSecondaries();
      for (int i = 0; i < numXTRPhotons; i++) {
        G4Track *secTrack = particleChangeXTR->GetSecondary(i);
        const double secEKin = secTrack->GetKineticEnergy();
        if (applyCuts && secEKin < (*theCutsGamma)[g4IMC]) {
          edep += secEKin;
          continue;
        }
        secTrack->SetParentID(aTrack->GetTrackID());
        secTrack->SetCreatorProcess(fXTRProcess);
        secTrack->SetTouchableHandle(aTrack->GetTouchableHandle());
        secTrack->SetWeight(aTrack->GetWeight());
        secondaries.push_back(secTrack);
      }
      particleChangeXTR->Clear();
    }

    step.AddTotalEnergyDeposit(edep);

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

    // Append the trajectory if it was requested.
    if (theTrajectory != nullptr) {
      theTrajectory->AppendStep(&step);
    }
  }

  // End of tracking: Inform processes and user.
  // === EndTracking ===

  // Invoke the fast simulation manager process EndTracking interface (if any)
  if (fFastSimProc != nullptr) {
    fFastSimProc->EndTracking();
  }

  if(userTrackingAction)
  {
    userTrackingAction->PostUserTrackingAction(aTrack);
  }

  // Delete the trajectory object (if the user set any)
  if (theTrajectory != nullptr) {
    delete theTrajectory;
  }

  evtMgr->StackTracks(&secondaries);
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void G4HepEmTrackingManager::TrackGamma(G4Track *aTrack) {
  TrackingManagerHelper::NeutralNavigation navigation;

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

  // Set OriginTouchableHandle for primary track
  if (aTrack->GetParentID() == 0) {
    aTrack->SetOriginTouchableHandle(aTrack->GetTouchableHandle());
  }

  // Set vertex information: in normal tracking this is done in
  //  `G4TrackingManager::ProcessOneTrack` when calling
  //  `G4SteppingManager::SetInitialStep`
  if (aTrack->GetCurrentStepNumber() == 0) {
    aTrack->SetVertexPosition(aTrack->GetPosition());
    aTrack->SetVertexMomentumDirection(aTrack->GetMomentumDirection());
    aTrack->SetVertexKineticEnergy(aTrack->GetKineticEnergy());
    aTrack->SetLogicalVolumeAtVertex(aTrack->GetVolume()->GetLogicalVolume());
  }

  // Prepare data structures used while tracking.
  G4Step&        step          = *fStep;
  G4TrackVector& secondaries   = *step.GetfSecondary();
  G4StepPoint&   preStepPoint  = *step.GetPreStepPoint();
  G4StepPoint&   postStepPoint = *step.GetPostStepPoint();
  step.InitializeStep(aTrack);
  aTrack->SetStep(&step);

  // Start of tracking: Inform user and processes.
  if(userTrackingAction) {
    userTrackingAction->PreUserTrackingAction(aTrack);
  }

  // Store the trajectory only if the user requested to store the trajectory in
  // the normal G4TrackingManager by setting their own trajectory object.
  // (usually in the PreUserTrackingAction)
  G4VTrajectory* theTrajectory = evtMgr->GetTrackingManager()->GetStoreTrajectory() == 0
                                 ? nullptr : evtMgr->GetTrackingManager()->GimmeTrajectory();

  // === StartTracking ===
  G4HepEmTLData* theTLData = fRunManager->GetTheTLData();
  G4HepEmGammaTrack* theGammaTrack = theTLData->GetPrimaryGammaTrack();
  G4HepEmTrack* thePrimaryTrack = theGammaTrack->GetTrack();
  theGammaTrack->ReSet();
  // In principle, we could continue to use the other generated Gaussian
  // number as long as we are in the same event, but play it safe.
  G4HepEmRandomEngine *rnge = theTLData->GetRNGEngine();
  rnge->DiscardGauss();

  // Pull data structures into local variables.
  G4HepEmData *theHepEmData = fRunManager->GetHepEmData();
  G4HepEmParameters *theHepEmPars = fRunManager->GetHepEmParameters();

  const G4DynamicParticle *theG4DPart = aTrack->GetDynamicParticle();

  thePrimaryTrack->SetCharge(0);

  // Invoke the fast simulation manager process StartTracking interface (if any)
  G4VProcess* fFastSimProc = fFastSimProcess[2];
  if (fFastSimProc != nullptr) {
    fFastSimProc->StartTracking(aTrack);
  }

  if (fGNucProcess != nullptr) {
    fGNucProcess->StartTracking(aTrack);
  }
  // === StartTracking ===

  while (aTrack->GetTrackStatus() == fAlive) {
    // Beginning of this step: Prepare data structures.
    aTrack->IncrementCurrentStepNumber();

    step.CopyPostToPreStepPoint();
    step.ResetTotalEnergyDeposit();
    step.SetControlFlag(G4SteppingControl::NormalCondition);
    aTrack->SetTouchableHandle(aTrack->GetNextTouchableHandle());

    auto* lvol = aTrack->GetTouchable()->GetVolume()->GetLogicalVolume();
    preStepPoint.SetMaterial(lvol->GetMaterial());
    auto* MCC = lvol->GetMaterialCutsCouple();
    preStepPoint.SetMaterialCutsCouple(lvol->GetMaterialCutsCouple());

    // Call the fast simulation manager process if any and check if any fast
    // sim models have been triggered (in that case, returns zero proposed
    // step length and `ExclusivelyForced` forced condition i.e. this and only
    // this process is ivnvoked.
    G4ForceCondition fastSimCondition;
    if (fFastSimProc != nullptr && fFastSimProc->PostStepGetPhysicalInteractionLength(*aTrack, 0.0, &fastSimCondition) == 0.0) {
      // Fast simulation active: invoke its PostStepDoIt, update properties and continue
      postStepPoint.SetProcessDefinedStep(fFastSimProc);
      G4VParticleChange* fastSimParticleChange = fFastSimProc->PostStepDoIt(*aTrack, step);
      fastSimParticleChange->UpdateStepForPostStep(&step);
      step.UpdateTrack();
      aTrack->SetTrackStatus(fastSimParticleChange->GetTrackStatus());
      // NOTE: it's assumed that fast simulation models do NOT produce secondaries
      //       (this can be improved and secondaries might be stacked here later if any)
      fastSimParticleChange->Clear();
      // End of this step: call the SD codes and required actions
      // NOTE: fast simulation sets the step control flag to `AvoidHitInvocation`
      //       as the fast sim models invoke their hit processing inside. So usually
      //       there is no hit processing here.
      if (step.GetControlFlag() != AvoidHitInvocation) {
        auto* sensitive = lvol->GetSensitiveDetector();
        if (sensitive) {
          sensitive->Hit(&step);
        }
      }
      if (userSteppingAction) {
        userSteppingAction->UserSteppingAction(&step);
      }
      auto* regionalAction = lvol->GetRegion()->GetRegionalSteppingAction();
      if (regionalAction) {
        regionalAction->UserSteppingAction(&step);
      }
      // Append the trajectory if a trajectory was set by the user.
      if (theTrajectory != nullptr) {
        theTrajectory->AppendStep(&step);
      }
      continue;
    }

    // Query step lengths from pyhsics and geometry, decide on limit.
    const G4double preStepEkin    = theG4DPart->GetKineticEnergy();
    const G4double preStepLogEkin = theG4DPart->GetLogKineticEnergy();
    thePrimaryTrack->SetEKin(preStepEkin, preStepLogEkin);
    const G4ThreeVector &primDir = theG4DPart->GetMomentumDirection();
    thePrimaryTrack->SetDirection(primDir[0], primDir[1], primDir[2]);

    const int g4IMC    = MCC->GetIndex();
    const int hepEmIMC = theHepEmData->fTheMatCutData->fG4MCIndexToHepEmMCIndex[g4IMC];
    thePrimaryTrack->SetMCIndex(hepEmIMC);
    bool preStepOnBoundary = preStepPoint.GetStepStatus() == G4StepStatus::fGeomBoundary;
    thePrimaryTrack->SetOnBoundary(preStepOnBoundary);

    G4HepEmGammaManager::HowFar(theHepEmData, theHepEmPars, theTLData);

    G4double physicalStep = thePrimaryTrack->GetGStepLength();
    G4double geometryStep = navigation.MakeStep(*aTrack, step, physicalStep);
    bool geometryLimitedStep = geometryStep < physicalStep;
    G4double finalStep = geometryLimitedStep ? geometryStep : physicalStep;

    step.SetStepLength(finalStep);
    aTrack->SetStepLength(finalStep);

    step.UpdateTrack();

    navigation.FinishStep(*aTrack, step);

    // Check if the track left the world.
    if (aTrack->GetNextVolume() == nullptr) {
      aTrack->SetTrackStatus(fStopAndKill);
    }

    // DoIt
    if (aTrack->GetTrackStatus() != fStopAndKill) {
      // Intercat but only when step was not limited by boundary
      const bool onBoundary = postStepPoint.GetStepStatus() == G4StepStatus::fGeomBoundary;
      thePrimaryTrack->SetOnBoundary(onBoundary);
      const G4VProcess* proc = nullptr;
      if (onBoundary) {
        proc = fTransportNoProcess;
        // Update the number of interaction length left only if on boundary
        thePrimaryTrack->SetGStepLength(aTrack->GetStepLength());
        G4HepEmGammaManager::UpdateNumIALeft(thePrimaryTrack);
      } else {
        double edep = 0.0;
        // NOTE: gamma-nuclear interaction needs to be done here while others in HepEm
        //       so we need to select first the interaction then see if we call HepEm
        //       or Geant4 physics to perform the selected interaction.
        G4HepEmGammaManager::SelectInteraction(theHepEmData, theTLData);
        const int iDProc = thePrimaryTrack->GetWinnerProcessIndex();
        if (iDProc != 3) {
          // Conversion, Compton or photoelectric --> use HepEm for the interaction
          // (NOTE: Ekin, MC-index, step-length, onBoundary have all set)
          G4HepEmGammaManager::Perform(theHepEmData, theHepEmPars, theTLData);
          // energy, e-depo, momentum direction and status
          const double ekin = thePrimaryTrack->GetEKin();
          edep = thePrimaryTrack->GetEnergyDeposit();
          postStepPoint.SetKineticEnergy(ekin);
          if (ekin <= 0.0) {
            aTrack->SetTrackStatus(fStopAndKill);
          }
          const double *pdir = thePrimaryTrack->GetDirection();
          postStepPoint.SetMomentumDirection( G4ThreeVector(pdir[0], pdir[1], pdir[2]) );

          step.UpdateTrack();

          // Stack secondaries created by the HepEm physics above
          edep += StackSecondaries(theTLData, aTrack, proc, g4IMC);

        } else {
          // Gamma-nuclear: --> use Geant4 for the interaction:
          // NOTE: it's destructive i.e. stopps and kills the gammma when the
          //    interaction happens so we do not use anymore the primary HepEm
          //    track thus we should not update the number of interaction length
          //    left and the others done in G4HepEmGammaManager::Perform before
          //    the interaction. HOWEVER, the interaction might not happen at
          //    the end so we do what we need in that case anyway for clarity:
          // - update the number of interaction length left for all process
          // - reset the number of interaction lenght left for the winner
          //   process, i.e. the one that will be invoked (the gamnma nuclear)
          // - reset the energy deposit to zero
          G4HepEmGammaManager::UpdateNumIALeft(thePrimaryTrack);
          thePrimaryTrack->SetNumIALeft(-1.0, iDProc);
          thePrimaryTrack->SetEnergyDeposit(0.0);
          // Invoke the gamma-nuclear interaction using the Geant4 process
          G4VParticleChange* particleChangeGNuc = nullptr;
          if (fGNucProcess != nullptr) {
            // call to set some fields of the process like material, energy etc...
            G4ForceCondition forceCondition;
            fGNucProcess->PostStepGetPhysicalInteractionLength(*aTrack, 0.0, &forceCondition);
            //
            postStepPoint.SetStepStatus(fPostStepDoItProc);
            particleChangeGNuc = fGNucProcess->PostStepDoIt(*aTrack, step);
            // update the track and stack according to the result of the interaction
            particleChangeGNuc->UpdateStepForPostStep(&step);
            step.UpdateTrack();
            aTrack->SetTrackStatus(particleChangeGNuc->GetTrackStatus());
            // need to add secondaries to the secondary vector of the current track
            // NOTE: as we use Geant4, we should care only those changes that are
            //   not included in the above update step and track, i.e. the energy
            //   deposited due to applying the cut when stacking the secondaries
            edep = StackG4Secondaries(particleChangeGNuc, aTrack, fGammaNoProcessVector[iDProc], g4IMC);
            // done: clear the particle change
            particleChangeGNuc->Clear();
          }
        }
        // Set process defined setp and add edep to the step
        proc = fGammaNoProcessVector[iDProc];
        step.AddTotalEnergyDeposit(edep);
      } // END if NOT onBoundary

      postStepPoint.SetProcessDefinedStep(proc);
    } // END status is NOT fStopAndKill

    aTrack->AddTrackLength(step.GetStepLength());

    // End of this step: Call sensitive detector and stepping actions.
    if(step.GetControlFlag() != AvoidHitInvocation) {
      auto* sensitive = lvol->GetSensitiveDetector();
      if(sensitive) {
        sensitive->Hit(&step);
      }
    }
    if(userSteppingAction) {
      userSteppingAction->UserSteppingAction(&step);
    }
    auto* regionalAction = lvol->GetRegion()->GetRegionalSteppingAction();
    if(regionalAction) {
      regionalAction->UserSteppingAction(&step);
    }
    // Append the trajectory if a trajectory was set by the user.
    if (theTrajectory != nullptr) {
      theTrajectory->AppendStep(&step);
    }

  } // END while loop of stepping till status is fAlive

  // End of tracking: Inform processes and user.
  // === EndTracking ===

  // Invoke the fast simulation manager process EndTracking interface (if any)
  if (fFastSimProc != nullptr) {
    fFastSimProc->EndTracking();
  }

  if(userTrackingAction)
  {
    userTrackingAction->PostUserTrackingAction(aTrack);
  }

  // Delete the trajectory object if the user set any
  if (theTrajectory != nullptr) {
    delete theTrajectory;
  }

  evtMgr->StackTracks(&secondaries);
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void G4HepEmTrackingManager::HandOverOneTrack(G4Track *aTrack) {
  const G4ParticleDefinition *part = aTrack->GetParticleDefinition();

  if (part == G4Electron::Definition() || part == G4Positron::Definition()) {
    TrackElectron(aTrack);
  } else if (part == G4Gamma::Definition()) {
    TrackGamma(aTrack);
  }

  aTrack->SetTrackStatus(fStopAndKill);
  delete aTrack;
}


// Helper that can be used to stack secondary e-/e+ and gamma i.e. everything
// that HepEm physics can produce
double G4HepEmTrackingManager::StackSecondaries(G4HepEmTLData* aTLData, G4Track* aG4PrimaryTrack, const G4VProcess* aG4CreatorProcess, int aG4IMC) {
  const int numSecElectron = aTLData->GetNumSecondaryElectronTrack();
  const int numSecGamma    = aTLData->GetNumSecondaryGammaTrack();
  const int numSecondaries = numSecElectron + numSecGamma;
  // return early if there are no secondaries created by HepEm physics
  double edep = 0.0;
  if (numSecondaries == 0) {
    return edep;
  }

  G4Step&        step           = *fStep;
  G4TrackVector& secondaries    = *step.GetfSecondary();
  G4StepPoint&   postStepPoint  = *step.GetPostStepPoint();

  const G4ThreeVector&     theG4PostStepPointPosition = postStepPoint.GetPosition();
  const G4double           theG4PostStepGlobalTime    = postStepPoint.GetGlobalTime();
  const G4TouchableHandle& theG4TouchableHandle       = aG4PrimaryTrack->GetTouchableHandle();
  const double             theG4ParentTrackWeight     = aG4PrimaryTrack->GetWeight();
  const int                theG4ParentTrackID         = aG4PrimaryTrack->GetTrackID();

  for (int is = 0; is < numSecElectron; ++is) {
    G4HepEmTrack *secTrack = aTLData->GetSecondaryElectronTrack(is)->GetTrack();
    const double  secEKin  = secTrack->GetEKin();
    const bool isElectron  = secTrack->GetCharge() < 0.0;
    if (applyCuts) {
      if (isElectron && secEKin < (*theCutsElectron)[aG4IMC]) {
        edep += secEKin;
        continue;
      } else if (!isElectron &&
                 CLHEP::electron_mass_c2 < (*theCutsGamma)[aG4IMC] &&
                 secEKin < (*theCutsPositron)[aG4IMC]) {
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
    aG4Track->SetParentID(theG4ParentTrackID);
    aG4Track->SetCreatorProcess(aG4CreatorProcess);
    aG4Track->SetTouchableHandle(theG4TouchableHandle);
    aG4Track->SetWeight(theG4ParentTrackWeight);
    secondaries.push_back(aG4Track);
  }
  aTLData->ResetNumSecondaryElectronTrack();

  for (int is = 0; is < numSecGamma; ++is) {
    G4HepEmTrack *secTrack = aTLData->GetSecondaryGammaTrack(is)->GetTrack();
    const double secEKin = secTrack->GetEKin();
    if (applyCuts && secEKin < (*theCutsGamma)[aG4IMC]) {
      edep += secEKin;
      continue;
    }

    const double *dir = secTrack->GetDirection();
    G4DynamicParticle *dp = new G4DynamicParticle(
        G4Gamma::Definition(), G4ThreeVector(dir[0], dir[1], dir[2]),
        secEKin);
    G4Track *aG4Track = new G4Track(dp, theG4PostStepGlobalTime,
                                    theG4PostStepPointPosition);
    aG4Track->SetParentID(theG4ParentTrackID);
    aG4Track->SetCreatorProcess(aG4CreatorProcess);
    aG4Track->SetTouchableHandle(theG4TouchableHandle);
    aG4Track->SetWeight(theG4ParentTrackWeight);
    secondaries.push_back(aG4Track);
  }
  aTLData->ResetNumSecondaryGammaTrack();

  return edep;
}


// Helper that can be used to stack secondary e-/e+ and gamma i.e. everything
// that HepEm physics can produce
double G4HepEmTrackingManager::StackG4Secondaries(G4VParticleChange* particleChange, G4Track* aG4PrimaryTrack, const G4VProcess* aG4CreatorProcess, int aG4IMC) {
  const int numSecondaries = particleChange->GetNumberOfSecondaries();
  // return early if there are no secondaries created by the physics interaction
  double edep = 0.0;
  if (numSecondaries == 0) {
    return edep;
  }

  G4Step&        step           = *fStep;
  G4TrackVector& secondaries    = *step.GetfSecondary();
  G4StepPoint&   postStepPoint  = *step.GetPostStepPoint();

  const G4ThreeVector&     theG4PostStepPointPosition = postStepPoint.GetPosition();
  const G4double           theG4PostStepGlobalTime    = postStepPoint.GetGlobalTime();
  const G4TouchableHandle& theG4TouchableHandle       = aG4PrimaryTrack->GetTouchableHandle();
  const double             theG4ParentTrackWeight     = aG4PrimaryTrack->GetWeight();
  const int                theG4ParentTrackID         = aG4PrimaryTrack->GetTrackID();

  for (int isec=0; isec<particleChange->GetNumberOfSecondaries(); ++isec) {
    G4Track *secTrack = particleChange->GetSecondary(isec);
    double   secEKin  = secTrack->GetKineticEnergy();
    const G4ParticleDefinition* secPartDef = secTrack->GetParticleDefinition();
//    std::cout << "applycutS = " << applyCuts << " Ekin = " << secEKin  << " imc = " << aG4IMC << " par = " << secPartDef->GetParticleName()<<std::endl;

    if (applyCuts) {
      if (secPartDef == G4Gamma::Definition() && secEKin < (*theCutsGamma)[aG4IMC]) {
        edep += secEKin;
        continue;
      } else if (secPartDef == G4Electron::Definition() && secEKin < (*theCutsElectron)[aG4IMC]) {
        edep += secEKin;
        continue;
      } else if (secPartDef == G4Positron::Definition() && CLHEP::electron_mass_c2 < (*theCutsGamma)[aG4IMC]
                 && secEKin < (*theCutsPositron)[aG4IMC]) {
        edep += secEKin + 2 * CLHEP::electron_mass_c2;
        continue;
      }
    }
    secTrack->SetParentID(theG4ParentTrackID);
    secTrack->SetCreatorProcess(aG4CreatorProcess);
    secTrack->SetTouchableHandle(theG4TouchableHandle);
    secTrack->SetWeight(theG4ParentTrackWeight);
    secondaries.push_back(secTrack);
  }

  return edep;
}


// Try to get the nuclear process pointer from the process manager of the particle
void G4HepEmTrackingManager::InitNuclearProcesses(int particleID) {
  G4ParticleDefinition* particleDef = nullptr;
  std::string nameNuclearProcess = "";
  G4VProcess** proc = nullptr;
  switch(particleID) {
    case 0: particleDef = G4Electron::Definition();
            nameNuclearProcess = "electronNuclear";
            proc = &fENucProcess;
            break;
    case 1: particleDef = G4Positron::Definition();
            nameNuclearProcess = "positronNuclear";
            proc = &fPNucProcess;
            break;
    case 2: particleDef = G4Gamma::Definition();
            nameNuclearProcess = "photonNuclear";
            proc = &fGNucProcess;
            break;
  }
  if (particleDef == nullptr) {
    std::cerr << " *** Unknown particle in G4HepEmTrackingManager::InitNuclearProcesses with ID = "
              << particleID
              << std::endl;
    exit(-1);
  }
  //
  const G4ProcessVector* processVector = particleDef->GetProcessManager()->GetProcessList();
  for (std::size_t ip=0; ip<processVector->entries(); ip++) {
    if( (*processVector)[ip]->GetProcessName()==nameNuclearProcess) {
      *proc = (*processVector)[ip];
      // make sure the process is initialised (element selectors needs to be built)
      (*proc)->PreparePhysicsTable(*particleDef);
      (*proc)->BuildPhysicsTable(*particleDef);
      break;
    }
  }
}


// Try to get the fast sim mamanger process for the particle (e- 0, e+ 1, gamma 2)
void G4HepEmTrackingManager::InitFastSimRelated(int particleID) {
  G4ParticleDefinition* particleDef = nullptr;
  switch(particleID) {
    case 0: particleDef = G4Electron::Definition();
            break;
    case 1: particleDef = G4Positron::Definition();
            break;
    case 2: particleDef = G4Gamma::Definition();
            break;
  }
  if (particleDef == nullptr) {
    std::cerr << " *** Unknown particle in G4HepEmTrackingManager::InitFastSimRelated with ID = "
              << particleID
              << std::endl;
    exit(-1);
  }
  const G4ProcessVector* processVector = particleDef->GetProcessManager()->GetProcessList();
  for (std::size_t ip=0; ip<processVector->entries(); ip++) {
    if( (*processVector)[ip]->GetProcessType()==G4ProcessType::fParameterisation) {
      fFastSimProcess[particleID] = (*processVector)[ip];
      break;
    }
  }
}

// ATLAS XTR RELATED:
void G4HepEmTrackingManager::InitXTRRelated() {
  // Try to get the pointer to the detector region that contains the TRT radiators
  // NOTE: becomes `nullptr` if there is no detector region with the name ensuring
  //       that everything works fine also outside ATLAS Athena
  // NOTE: after the suggested changes in Athena, the region name should be
  //       `TRT_RADIATOR` but till that it's `DefaultRegionForTheWorld`
  fXTRRegion = G4RegionStore::GetInstance()->GetRegion(fXTRRegionName);
  // Try to get the pointer to the TRTTransitionRadiation process (same for e-/e+)
  // NOTE: stays `nullptr` if gamma dosen't have process with the name ensuring
  //       that everything works fine also outside ATLAS Athena
  const G4ProcessVector* processVector = G4Electron::Definition()->GetProcessManager()->GetProcessList();
  for (std::size_t ip=0; ip<processVector->entries(); ip++) {
    if( (*processVector)[ip]->GetProcessName()==fXTRProcessName) {
      fXTRProcess = (*processVector)[ip];
      break;
    }
  }
  // Print information if the XTR process was found (enable to check)
  if (fXTRProcess != nullptr) {
    std::cout << " G4HepEmTrackingManager: found the ATLAS specific "
              << fXTRProcess->GetProcessName() << " process";
    if (fXTRRegion != nullptr) {
      std::cout << " with the " << fXTRRegion->GetName() << " region.";
    }
    std::cout << std::endl;
  }
}
