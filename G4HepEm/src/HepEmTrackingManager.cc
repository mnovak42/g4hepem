#include "HepEmTrackingManager.hh"
#include "TrackingManagerHelper.hh"

#include "G4HepEmCLHEPRandomEngine.hh"
#include "G4HepEmData.hh"
#include "G4HepEmMatCutData.hh"
#include "G4HepEmRunManager.hh"
#include "G4HepEmTLData.hh"

#include "G4HepEmElectronManager.hh"
#include "G4HepEmElectronTrack.hh"
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
  class ElectronPhysics final : public TrackingManagerHelper::Physics {
  public:
    ElectronPhysics(HepEmTrackingManager &mgr) : fMgr(mgr) {}

    void StartTracking(G4Track *aTrack) override {
      fMgr.fRunManager->GetTheTLData()->GetPrimaryElectronTrack()->ReSet();
      // In principle, we could continue to use the other generated Gaussian
      // number as long as we are in the same event, but play it safe.
      fMgr.fRunManager->GetTheTLData()->GetRNGEngine()->DiscardGauss();
    }

    G4double GetPhysicalInteractionLength(const G4Track &track) override {
      G4HepEmTLData *theTLData = fMgr.fRunManager->GetTheTLData();
      G4HepEmTrack *thePrimaryTrack =
          theTLData->GetPrimaryElectronTrack()->GetTrack();
      G4HepEmData *theHepEmData = fMgr.fRunManager->GetHepEmData();
      thePrimaryTrack->SetCharge(track.GetParticleDefinition()->GetPDGCharge());
      const G4DynamicParticle *theG4DPart = track.GetDynamicParticle();
      thePrimaryTrack->SetEKin(theG4DPart->GetKineticEnergy(),
                               theG4DPart->GetLogKineticEnergy());

      const int g4IMC = track.GetMaterialCutsCouple()->GetIndex();
      const int hepEmIMC =
          theHepEmData->fTheMatCutData->fG4MCIndexToHepEmMCIndex[g4IMC];
      thePrimaryTrack->SetMCIndex(hepEmIMC);
      const G4StepPoint *theG4PreStepPoint = track.GetStep()->GetPreStepPoint();
      bool onBoundary =
          theG4PreStepPoint->GetStepStatus() == G4StepStatus::fGeomBoundary;
      thePrimaryTrack->SetOnBoundary(onBoundary);
      const double preSafety =
          onBoundary ? 0.
                     : fMgr.fSafetyHelper->ComputeSafety(track.GetPosition());
      thePrimaryTrack->SetSafety(preSafety);
      G4HepEmElectronManager::HowFar(
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
      G4HepEmElectronTrack *theElTrack = theTLData->GetPrimaryElectronTrack();
      G4HepEmTrack *thePrimaryTrack = theElTrack->GetTrack();

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
      G4HepEmElectronManager::Perform(fMgr.fRunManager->GetHepEmData(),
                                      fMgr.fRunManager->GetHepEmParameters(),
                                      theTLData);
      step.SetStepLength(theElTrack->GetPStepLength());

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

      // apply MSC displacement if its length is longer than a minimum and we
      // are not on boundary
      G4ThreeVector position = theG4PostStepPoint->GetPosition();
      if (!onBoundary) {
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
              0.99 * fMgr.fSafetyHelper->ComputeSafety(position, dispR);
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
            fMgr.fSafetyHelper->ReLocateWithinVolume(position);
            theG4PostStepPoint->SetPosition(position);
          }
        }
      }

      step.UpdateTrack();

      const int g4IMC =
          step.GetPreStepPoint()->GetMaterialCutsCouple()->GetIndex();
      // secondary: only possible is e- or gamma at the moemnt
      const int numSecElectron = theTLData->GetNumSecondaryElectronTrack();
      const int numSecGamma = theTLData->GetNumSecondaryGammaTrack();
      const int numSecondaries = numSecElectron + numSecGamma;
      if (numSecondaries > 0) {
        const G4ThreeVector &theG4PostStepPointPosition = position;
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

  ElectronPhysics physics(*this);
  TrackingManagerHelper::TrackChargedParticle(aTrack, physics);
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
