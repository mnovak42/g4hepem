#include "G4EmTrackingManager.hh"
#include "TrackingManagerHelper.hh"

#include "G4eBremsstrahlung.hh"
#include "G4eIonisation.hh"
#include "G4eMultipleScattering.hh"
#include "G4eplusAnnihilation.hh"

#include "G4ComptonScattering.hh"
#include "G4GammaConversion.hh"
#include "G4PhotoElectricEffect.hh"

#include "G4Electron.hh"
#include "G4Gamma.hh"
#include "G4Positron.hh"

#include "G4Step.hh"

G4EmTrackingManager *G4EmTrackingManager::masterTrackingManager = nullptr;

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

G4EmTrackingManager::G4EmTrackingManager() {
  // e-
  {
    electron.msc = new G4eMultipleScattering;
    electron.ioni = new G4eIonisation;
    electron.brems = new G4eBremsstrahlung;
  }

  // e+
  {
    positron.msc = new G4eMultipleScattering;
    positron.ioni = new G4eIonisation;
    positron.brems = new G4eBremsstrahlung;
    positron.annihilation = new G4eplusAnnihilation;
  }

  {
    gamma.pe = new G4PhotoElectricEffect;
    gamma.compton = new G4ComptonScattering;
    gamma.conversion = new G4GammaConversion;
  }

  if (masterTrackingManager == nullptr) {
    masterTrackingManager = this;
  } else {
    electron.msc->SetMasterProcess(masterTrackingManager->electron.msc);
    electron.ioni->SetMasterProcess(masterTrackingManager->electron.ioni);
    electron.brems->SetMasterProcess(masterTrackingManager->electron.brems);

    positron.msc->SetMasterProcess(masterTrackingManager->positron.msc);
    positron.ioni->SetMasterProcess(masterTrackingManager->positron.ioni);
    positron.brems->SetMasterProcess(masterTrackingManager->positron.brems);
    positron.annihilation->SetMasterProcess(
        masterTrackingManager->positron.annihilation);

    gamma.pe->SetMasterProcess(masterTrackingManager->gamma.pe);
    gamma.compton->SetMasterProcess(masterTrackingManager->gamma.compton);
    gamma.conversion->SetMasterProcess(masterTrackingManager->gamma.conversion);
  }

  fStep = new G4Step;
  fStep->NewSecondaryVector();
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

G4EmTrackingManager::~G4EmTrackingManager() {
  delete fStep;
  if (masterTrackingManager == this) {
    masterTrackingManager = nullptr;
  }
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void G4EmTrackingManager::BuildPhysicsTable(const G4ParticleDefinition &part) {
  if (&part == G4Electron::Definition()) {
    electron.msc->BuildPhysicsTable(part);
    electron.ioni->BuildPhysicsTable(part);
    electron.brems->BuildPhysicsTable(part);
  } else if (&part == G4Positron::Definition()) {
    positron.msc->BuildPhysicsTable(part);
    positron.ioni->BuildPhysicsTable(part);
    positron.brems->BuildPhysicsTable(part);
    positron.annihilation->BuildPhysicsTable(part);
  } else if (&part == G4Gamma::Definition()) {
    gamma.pe->BuildPhysicsTable(part);
    gamma.compton->BuildPhysicsTable(part);
    gamma.conversion->BuildPhysicsTable(part);
  }
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void G4EmTrackingManager::PreparePhysicsTable(
    const G4ParticleDefinition &part) {
  if (&part == G4Electron::Definition()) {
    electron.msc->PreparePhysicsTable(part);
    electron.ioni->PreparePhysicsTable(part);
    electron.brems->PreparePhysicsTable(part);
  } else if (&part == G4Positron::Definition()) {
    positron.msc->PreparePhysicsTable(part);
    positron.ioni->PreparePhysicsTable(part);
    positron.brems->PreparePhysicsTable(part);
    positron.annihilation->PreparePhysicsTable(part);
  } else if (&part == G4Gamma::Definition()) {
    gamma.pe->PreparePhysicsTable(part);
    gamma.compton->PreparePhysicsTable(part);
    gamma.conversion->PreparePhysicsTable(part);
  }
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void G4EmTrackingManager::TrackElectron(G4Track *aTrack) {
  class ElectronPhysics final : public TrackingManagerHelper::Physics {
  public:
    ElectronPhysics(G4EmTrackingManager &mgr) : fMgr(mgr) {}

    void StartTracking(G4Track *aTrack) override {
      auto &electron = fMgr.electron;

      electron.msc->StartTracking(aTrack);
      electron.ioni->StartTracking(aTrack);
      electron.brems->StartTracking(aTrack);

      fPreviousStepLength = 0;
    }
    void EndTracking() override {
      auto &electron = fMgr.electron;

      electron.msc->EndTracking();
      electron.ioni->EndTracking();
      electron.brems->EndTracking();
    }

    G4double GetPhysicalInteractionLength(const G4Track &track) override {
      auto &electron = fMgr.electron;
      G4double physIntLength, proposedSafety = DBL_MAX;
      G4ForceCondition condition;
      G4GPILSelection selection;

      fProposedStep = DBL_MAX;
      fSelected = -1;

      physIntLength =
          electron.brems->PostStepGPIL(track, fPreviousStepLength, &condition);
      if (physIntLength < fProposedStep) {
        fProposedStep = physIntLength;
        fSelected = 0;
      }

      physIntLength =
          electron.ioni->PostStepGPIL(track, fPreviousStepLength, &condition);
      if (physIntLength < fProposedStep) {
        fProposedStep = physIntLength;
        fSelected = 1;
      }

      physIntLength = electron.ioni->AlongStepGPIL(track, fPreviousStepLength,
                                                   fProposedStep,
                                                   proposedSafety, &selection);
      if (physIntLength < fProposedStep) {
        fProposedStep = physIntLength;
        fSelected = -1;
      }

      physIntLength =
          electron.msc->AlongStepGPIL(track, fPreviousStepLength, fProposedStep,
                                      proposedSafety, &selection);
      if (physIntLength < fProposedStep) {
        fProposedStep = physIntLength;
        // Check if MSC actually wants to win, in most cases it only limits the
        // step size.
        if (selection == CandidateForSelection) {
          fSelected = -1;
        }
      }

      return fProposedStep;
    }

    void AlongStepDoIt(G4Track &track, G4Step &step, G4TrackVector &) override {
      if (step.GetStepLength() == fProposedStep) {
        step.GetPostStepPoint()->SetStepStatus(fAlongStepDoItProc);
      } else {
        // Remember that the step was limited by geometry.
        fSelected = -1;
      }
      auto &electron = fMgr.electron;
      G4VParticleChange *particleChange;

      particleChange = electron.msc->AlongStepDoIt(track, step);
      particleChange->UpdateStepForAlongStep(&step);
      track.SetTrackStatus(particleChange->GetTrackStatus());
      particleChange->Clear();

      particleChange = electron.ioni->AlongStepDoIt(track, step);
      particleChange->UpdateStepForAlongStep(&step);
      track.SetTrackStatus(particleChange->GetTrackStatus());
      particleChange->Clear();

      fPreviousStepLength = step.GetStepLength();
    }

    void PostStepDoIt(G4Track &track, G4Step &step,
                      G4TrackVector &secondaries) override {
      if (fSelected < 0) {
        return;
      }
      step.GetPostStepPoint()->SetStepStatus(fPostStepDoItProc);

      auto &electron = fMgr.electron;
      G4VProcess *process;
      G4VParticleChange *particleChange;

      switch (fSelected) {
      case 0:
        process = electron.brems;
        particleChange = electron.brems->PostStepDoIt(track, step);
        break;
      case 1:
        process = electron.ioni;
        particleChange = electron.ioni->PostStepDoIt(track, step);
        break;
      }

      particleChange->UpdateStepForPostStep(&step);
      step.UpdateTrack();

      int numSecondaries = particleChange->GetNumberOfSecondaries();
      for (int i = 0; i < numSecondaries; i++) {
        G4Track *secondary = particleChange->GetSecondary(i);
        secondary->SetParentID(track.GetTrackID());
        secondary->SetCreatorProcess(process);
        secondaries.push_back(secondary);
      }

      track.SetTrackStatus(particleChange->GetTrackStatus());
      particleChange->Clear();
    }

  private:
    G4EmTrackingManager &fMgr;
    G4double fPreviousStepLength;
    G4double fProposedStep;
    G4int fSelected;
  };

  ElectronPhysics physics(*this);
  TrackingManagerHelper::TrackChargedParticle(aTrack, fStep, physics);
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void G4EmTrackingManager::TrackPositron(G4Track *aTrack) {
  class PositronPhysics final : public TrackingManagerHelper::Physics {
  public:
    PositronPhysics(G4EmTrackingManager &mgr) : fMgr(mgr) {}

    void StartTracking(G4Track *aTrack) override {
      auto &positron = fMgr.positron;

      positron.msc->StartTracking(aTrack);
      positron.ioni->StartTracking(aTrack);
      positron.brems->StartTracking(aTrack);
      positron.annihilation->StartTracking(aTrack);

      fPreviousStepLength = 0;
    }
    void EndTracking() override {
      auto &positron = fMgr.positron;

      positron.msc->EndTracking();
      positron.ioni->EndTracking();
      positron.brems->EndTracking();
      positron.annihilation->EndTracking();
    }

    G4double GetPhysicalInteractionLength(const G4Track &track) override {
      auto &positron = fMgr.positron;
      G4double physIntLength, proposedSafety = DBL_MAX;
      G4ForceCondition condition;
      G4GPILSelection selection;

      fProposedStep = DBL_MAX;
      fSelected = -1;

      physIntLength = positron.annihilation->PostStepGPIL(
          track, fPreviousStepLength, &condition);
      if (physIntLength < fProposedStep) {
        fProposedStep = physIntLength;
        fSelected = 0;
      }

      physIntLength =
          positron.brems->PostStepGPIL(track, fPreviousStepLength, &condition);
      if (physIntLength < fProposedStep) {
        fProposedStep = physIntLength;
        fSelected = 1;
      }

      physIntLength =
          positron.ioni->PostStepGPIL(track, fPreviousStepLength, &condition);
      if (physIntLength < fProposedStep) {
        fProposedStep = physIntLength;
        fSelected = 2;
      }

      physIntLength = positron.ioni->AlongStepGPIL(track, fPreviousStepLength,
                                                   fProposedStep,
                                                   proposedSafety, &selection);
      if (physIntLength < fProposedStep) {
        fProposedStep = physIntLength;
        fSelected = -1;
      }

      physIntLength =
          positron.msc->AlongStepGPIL(track, fPreviousStepLength, fProposedStep,
                                      proposedSafety, &selection);
      if (physIntLength < fProposedStep) {
        fProposedStep = physIntLength;
        // Check if MSC actually wants to win, in most cases it only limits the
        // step size.
        if (selection == CandidateForSelection) {
          fSelected = -1;
        }
      }

      return fProposedStep;
    }

    void AlongStepDoIt(G4Track &track, G4Step &step, G4TrackVector &) override {
      if (step.GetStepLength() == fProposedStep) {
        step.GetPostStepPoint()->SetStepStatus(fAlongStepDoItProc);
      } else {
        // Remember that the step was limited by geometry.
        fSelected = -1;
      }
      auto &positron = fMgr.positron;
      G4VParticleChange *particleChange;

      particleChange = positron.msc->AlongStepDoIt(track, step);
      particleChange->UpdateStepForAlongStep(&step);
      track.SetTrackStatus(particleChange->GetTrackStatus());
      particleChange->Clear();

      particleChange = positron.ioni->AlongStepDoIt(track, step);
      particleChange->UpdateStepForAlongStep(&step);
      track.SetTrackStatus(particleChange->GetTrackStatus());
      particleChange->Clear();

      fPreviousStepLength = step.GetStepLength();
    }

    void PostStepDoIt(G4Track &track, G4Step &step,
                      G4TrackVector &secondaries) override {
      if (fSelected < 0) {
        return;
      }
      step.GetPostStepPoint()->SetStepStatus(fPostStepDoItProc);

      auto &positron = fMgr.positron;
      G4VProcess *process;
      G4VParticleChange *particleChange;

      switch (fSelected) {
      case 0:
        process = positron.annihilation;
        particleChange = positron.annihilation->PostStepDoIt(track, step);
        break;
      case 1:
        process = positron.brems;
        particleChange = positron.brems->PostStepDoIt(track, step);
        break;
      case 2:
        process = positron.ioni;
        particleChange = positron.ioni->PostStepDoIt(track, step);
        break;
      }

      particleChange->UpdateStepForPostStep(&step);
      step.UpdateTrack();

      int numSecondaries = particleChange->GetNumberOfSecondaries();
      for (int i = 0; i < numSecondaries; i++) {
        G4Track *secondary = particleChange->GetSecondary(i);
        secondary->SetParentID(track.GetTrackID());
        secondary->SetCreatorProcess(process);
        secondaries.push_back(secondary);
      }

      track.SetTrackStatus(particleChange->GetTrackStatus());
      particleChange->Clear();
    }

    G4bool HasAtRestProcesses() override { return true; }

    void AtRestDoIt(G4Track &track, G4Step &step,
                    G4TrackVector &secondaries) override {
      auto &positron = fMgr.positron;
      // Annihilate the positron at rest.
      G4VParticleChange *particleChange =
          positron.annihilation->AtRestDoIt(track, step);
      particleChange->UpdateStepForAtRest(&step);
      step.UpdateTrack();

      int numSecondaries = particleChange->GetNumberOfSecondaries();
      for (int i = 0; i < numSecondaries; i++) {
        G4Track *secondary = particleChange->GetSecondary(i);
        secondary->SetParentID(track.GetTrackID());
        secondary->SetCreatorProcess(positron.annihilation);
        secondaries.push_back(secondary);
      }

      track.SetTrackStatus(particleChange->GetTrackStatus());
      particleChange->Clear();
    }

  private:
    G4EmTrackingManager &fMgr;
    G4double fPreviousStepLength;
    G4double fProposedStep;
    G4int fSelected;
  };

  PositronPhysics physics(*this);
  TrackingManagerHelper::TrackChargedParticle(aTrack, fStep, physics);
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void G4EmTrackingManager::TrackGamma(G4Track *aTrack) {
  class GammaPhysics final : public TrackingManagerHelper::Physics {
  public:
    GammaPhysics(G4EmTrackingManager &mgr) : fMgr(mgr) {}

    void StartTracking(G4Track *aTrack) override {
      auto &gamma = fMgr.gamma;

      gamma.pe->StartTracking(aTrack);
      gamma.compton->StartTracking(aTrack);
      gamma.conversion->StartTracking(aTrack);

      fPreviousStepLength = 0;
    }
    void EndTracking() override {
      auto &gamma = fMgr.gamma;

      gamma.pe->EndTracking();
      gamma.compton->EndTracking();
      gamma.conversion->EndTracking();
    }

    G4double GetPhysicalInteractionLength(const G4Track &track) override {
      auto &gamma = fMgr.gamma;
      G4double physIntLength;
      G4ForceCondition condition;

      fProposedStep = DBL_MAX;
      fSelected = -1;

      physIntLength = gamma.conversion->PostStepGPIL(track, fPreviousStepLength,
                                                     &condition);
      if (physIntLength < fProposedStep) {
        fProposedStep = physIntLength;
        fSelected = 0;
      }

      physIntLength =
          gamma.compton->PostStepGPIL(track, fPreviousStepLength, &condition);
      if (physIntLength < fProposedStep) {
        fProposedStep = physIntLength;
        fSelected = 1;
      }

      physIntLength =
          gamma.pe->PostStepGPIL(track, fPreviousStepLength, &condition);
      if (physIntLength < fProposedStep) {
        fProposedStep = physIntLength;
        fSelected = 2;
      }

      return fProposedStep;
    }

    void AlongStepDoIt(G4Track &, G4Step &step, G4TrackVector &) override {
      if (step.GetStepLength() == fProposedStep) {
        step.GetPostStepPoint()->SetStepStatus(fAlongStepDoItProc);
      } else {
        // Remember that the step was limited by geometry.
        fSelected = -1;
      }
      fPreviousStepLength = step.GetStepLength();
    }

    void PostStepDoIt(G4Track &track, G4Step &step,
                      G4TrackVector &secondaries) override {
      if (fSelected < 0) {
        return;
      }
      step.GetPostStepPoint()->SetStepStatus(fPostStepDoItProc);

      auto &gamma = fMgr.gamma;
      G4VProcess *process;
      G4VParticleChange *particleChange;

      switch (fSelected) {
      case 0:
        process = gamma.conversion;
        particleChange = gamma.conversion->PostStepDoIt(track, step);
        break;
      case 1:
        process = gamma.compton;
        particleChange = gamma.compton->PostStepDoIt(track, step);
        break;
      case 2:
        process = gamma.pe;
        particleChange = gamma.pe->PostStepDoIt(track, step);
        break;
      }

      particleChange->UpdateStepForPostStep(&step);
      step.UpdateTrack();

      int numSecondaries = particleChange->GetNumberOfSecondaries();
      for (int i = 0; i < numSecondaries; i++) {
        G4Track *secondary = particleChange->GetSecondary(i);
        secondary->SetParentID(track.GetTrackID());
        secondary->SetCreatorProcess(process);
        secondaries.push_back(secondary);
      }

      track.SetTrackStatus(particleChange->GetTrackStatus());
      particleChange->Clear();
    }

  private:
    G4EmTrackingManager &fMgr;
    G4double fPreviousStepLength;
    G4double fProposedStep;
    G4int fSelected;
  };

  GammaPhysics physics(*this);
  TrackingManagerHelper::TrackNeutralParticle(aTrack, fStep, physics);
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void G4EmTrackingManager::HandOverOneTrack(G4Track *aTrack) {
  const G4ParticleDefinition *part = aTrack->GetParticleDefinition();

  if (part == G4Electron::Definition()) {
    TrackElectron(aTrack);
  } else if (part == G4Positron::Definition()) {
    TrackPositron(aTrack);
  } else if (part == G4Gamma::Definition()) {
    TrackGamma(aTrack);
  }

  aTrack->SetTrackStatus(fStopAndKill);
  delete aTrack;
}
