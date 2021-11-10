
#include "G4HepEmProcess.hh"


#include "G4HepEmRunManager.hh"
#include "G4HepEmData.hh"
#include "G4HepEmMatCutData.hh"
#include "G4HepEmTLData.hh"
#include "G4HepEmRunManager.hh"
#include "G4HepEmCLHEPRandomEngine.hh"
#include "G4HepEmNoProcess.hh"

#include "G4HepEmElectronTrack.hh"
#include "G4HepEmGammaTrack.hh"
#include "G4HepEmElectronManager.hh"
#include "G4HepEmGammaManager.hh"

#include "G4Threading.hh"
#include "G4Track.hh"
#include "G4Step.hh"
#include "G4StepPoint.hh"
#include "G4StepStatus.hh"
#include "G4MaterialCutsCouple.hh"
#include "G4ParticleChange.hh"

#include "G4SafetyHelper.hh"
#include "G4TransportationManager.hh"

#include "G4EmParameters.hh"
#include "G4ProductionCutsTable.hh"

#include "G4Electron.hh"
#include "G4Positron.hh"
#include "G4Gamma.hh"

G4HepEmProcess::G4HepEmProcess()
: G4VProcess("hepEm", fElectromagnetic),
  fTheG4HepEmRunManager(nullptr) {
  enableAtRestDoIt    = false;
  enableAlongStepDoIt = false;
  enablePostStepDoIt  = true;

  fTheG4HepEmRunManager   = new G4HepEmRunManager(G4Threading::IsMasterThread());
  fTheG4HepEmRandomEngine = new G4HepEmCLHEPRandomEngine(G4Random::getTheEngine());
  fParticleChange  = new G4ParticleChange();

  fSafetyHelper = G4TransportationManager::GetTransportationManager()->GetSafetyHelper();
  fSafetyHelper->InitialiseHelper();

  fElectronNoProcessVector.resize(4, nullptr);
  fGammaNoProcessVector.resize(4, nullptr);
}

G4HepEmProcess::~G4HepEmProcess() {
  delete fTheG4HepEmRunManager;
  delete fTheG4HepEmRandomEngine;
}


void G4HepEmProcess::BuildPhysicsTable(const G4ParticleDefinition& partDef) {
  G4cout << " G4HepEmProcess::BuildPhysicsTable for Particle = " << partDef.GetParticleName() << G4endl;
  StreamInfo(G4cout, partDef);

  // The ptr-s to global data structures created and filled in InitializeGlobal()
  // will be copied to the workers and the TL-data structure will be created.
  //fTheG4HepEmRunManager->Initialize(G4Random::getTheEngine());

  if (partDef.GetPDGEncoding()==11) {          // e-
    fTheG4HepEmRunManager->Initialize(fTheG4HepEmRandomEngine, 0);
    // construct fake G4VProcess-es with the proper name and indices matching the hepEm process indices
    fElectronNoProcessVector[0] = new G4HepEmNoProcess("eIoni");
    fElectronNoProcessVector[1] = new G4HepEmNoProcess("eBrem");
    fElectronNoProcessVector[3] = new G4HepEmNoProcess("msc");
  } else if (partDef.GetPDGEncoding()==-11) {  // e+
    fTheG4HepEmRunManager->Initialize(fTheG4HepEmRandomEngine, 1);
    fElectronNoProcessVector[0] = new G4HepEmNoProcess("eIoni");
    fElectronNoProcessVector[1] = new G4HepEmNoProcess("eBrem");
    fElectronNoProcessVector[2] = new G4HepEmNoProcess("annihl");
    fElectronNoProcessVector[3] = new G4HepEmNoProcess("msc");
  } else if (partDef.GetPDGEncoding()==22) {   // gamma
    fTheG4HepEmRunManager->Initialize(fTheG4HepEmRandomEngine, 2);
    fGammaNoProcessVector[0]    = new G4HepEmNoProcess("phot");
    fGammaNoProcessVector[1]    = new G4HepEmNoProcess("compt");
    fGammaNoProcessVector[2]    = new G4HepEmNoProcess("conv");
  } else {
    std::cerr << " **** ERROR in G4HepEmProcess::BuildPhysicsTable: unknown particle " << std::endl;
    exit(-1);
  }
}

void G4HepEmProcess::PreparePhysicsTable(const G4ParticleDefinition&) {
  applyCuts = G4EmParameters::Instance()->ApplyCuts();

  if (applyCuts) {
    auto* theCoupleTable= G4ProductionCutsTable::GetProductionCutsTable();
    theCutsGamma        = theCoupleTable->GetEnergyCutsVector(idxG4GammaCut);
    theCutsElectron     = theCoupleTable->GetEnergyCutsVector(idxG4ElectronCut);
    theCutsPositron     = theCoupleTable->GetEnergyCutsVector(idxG4PositronCut);
  }
}

void     G4HepEmProcess::StartTracking(G4Track* track) {
    // reset number of interaction length left to -1
  const G4ParticleDefinition* partDef = track->GetParticleDefinition();
  if (std::abs(partDef->GetPDGEncoding())==11) {          // e- and e+
    fTheG4HepEmRunManager->GetTheTLData()->GetPrimaryElectronTrack()->ReSet();
  } else if (partDef->GetPDGEncoding()==22) {   // gamma
    fTheG4HepEmRunManager->GetTheTLData()->GetPrimaryGammaTrack()->ReSet();
  }
  // In principle, we could continue to use the other generated Gaussian number
  // as long as we are in the same event, but play it safe.
  fTheG4HepEmRandomEngine->DiscardGauss();
}

G4double G4HepEmProcess::PostStepGetPhysicalInteractionLength ( const G4Track& track,
                                                                G4double previousStepSize,
                                                                G4ForceCondition* condition ) {
  G4HepEmTLData*            theTLData = fTheG4HepEmRunManager->GetTheTLData();
  const G4ParticleDefinition* partDef = track.GetParticleDefinition();
  const bool                 isGamma  = (partDef->GetPDGEncoding()==22);
  G4HepEmTrack*       thePrimaryTrack = isGamma
                                        ? theTLData->GetPrimaryGammaTrack()->GetTrack()
                                        : theTLData->GetPrimaryElectronTrack()->GetTrack();
  // forced the DoIt to be called in all cases
  *condition = G4ForceCondition::Forced;
  thePrimaryTrack->SetCharge(partDef->GetPDGCharge());
  const G4DynamicParticle* theG4DPart = track.GetDynamicParticle();
  thePrimaryTrack->SetEKin(theG4DPart->GetKineticEnergy(), theG4DPart->GetLogKineticEnergy());
  const int    g4IMC = track.GetMaterialCutsCouple()->GetIndex();
  const int hepEmIMC = fTheG4HepEmRunManager->GetHepEmData()->fTheMatCutData->fG4MCIndexToHepEmMCIndex[g4IMC];
  thePrimaryTrack->SetMCIndex(hepEmIMC);
  const G4StepPoint* theG4PreStepPoint = track.GetStep()->GetPreStepPoint();
  const bool onBoundary = theG4PreStepPoint->GetStepStatus()==G4StepStatus::fGeomBoundary;
  thePrimaryTrack->SetOnBoundary(onBoundary);
  //
  G4StepPoint* theG4PostStepPoint = track.GetStep()->GetPostStepPoint();
  if (isGamma) {
    // note: safety is used only for e-/e+ in the MCS step limit)
    //thePrimaryTrack->SetSafety(theG4PreStepPoint->GetSafety());
    G4HepEmGammaManager::HowFar(fTheG4HepEmRunManager->GetHepEmData(), fTheG4HepEmRunManager->GetHepEmParameters(), theTLData);
  } else {
    const double preSafety = onBoundary ? 0. : fSafetyHelper->ComputeSafety(track.GetPosition());
    thePrimaryTrack->SetSafety(preSafety);
    G4HepEmElectronManager::HowFar(fTheG4HepEmRunManager->GetHepEmData(), fTheG4HepEmRunManager->GetHepEmParameters(), theTLData);
  }
  // returns with the geometrcal step length: straight line distance to make along the org direction
  return thePrimaryTrack->GetGStepLength();
}


G4VParticleChange* G4HepEmProcess::PostStepDoIt( const G4Track& track, const G4Step& step) {
  // init particle change: it might be more special we need to see later
  fParticleChange->Initialize(track);

  G4HepEmTLData*              theTLData = fTheG4HepEmRunManager->GetTheTLData();
  const G4ParticleDefinition*   partDef = track.GetParticleDefinition();
  const bool                    isGamma = (partDef->GetPDGEncoding()==22);
  G4StepPoint* theG4PostStepPoint       = step.GetPostStepPoint();
  const bool               onBoundary   = theG4PostStepPoint->GetStepStatus()==G4StepStatus::fGeomBoundary;
  G4HepEmTrack*       thePrimaryTrack = isGamma
                                        ? theTLData->GetPrimaryGammaTrack()->GetTrack()
                                        : theTLData->GetPrimaryElectronTrack()->GetTrack();
  if (isGamma && onBoundary) {
    thePrimaryTrack->SetGStepLength(track.GetStepLength());
    G4HepEmGammaManager::UpdateNumIALeft(thePrimaryTrack);
    return fParticleChange;
  }
  // NOTE: this primary track is the same as in the last call in the HowFar()
  //       But transportation might changed its direction, geomertical step length,
  //       or status ( on boundary or not).
  const G4ThreeVector& primDir = track.GetDynamicParticle()->GetMomentumDirection();
  thePrimaryTrack->SetDirection(primDir[0], primDir[1], primDir[2]);
  thePrimaryTrack->SetGStepLength(track.GetStepLength());
  thePrimaryTrack->SetOnBoundary(onBoundary);
  // invoke the physics interactions (all i.e. all along- and post-step as well as possible at rest)
  double pStepLength = track.GetStepLength();
  if (isGamma) {
    G4HepEmGammaManager::Perform(fTheG4HepEmRunManager->GetHepEmData(), fTheG4HepEmRunManager->GetHepEmParameters(), theTLData);
    //
    // set dummy G4VProcess pointers to provide (name) information regarding processes limited the step
    switch (thePrimaryTrack->GetWinnerProcessIndex()) {
      case 0: theG4PostStepPoint->SetProcessDefinedStep(fGammaNoProcessVector[0]);
              break;
      case 1: theG4PostStepPoint->SetProcessDefinedStep(fGammaNoProcessVector[1]);
              break;
      case 2: theG4PostStepPoint->SetProcessDefinedStep(fGammaNoProcessVector[2]);
              break;
    }
  } else {
    G4HepEmElectronManager::Perform(fTheG4HepEmRunManager->GetHepEmData(), fTheG4HepEmRunManager->GetHepEmParameters(), theTLData);
    // account possible change in the physics/true step length due to MSC
    pStepLength = theTLData->GetPrimaryElectronTrack()->GetPStepLength();
    //
    // set dummy G4VProcess pointers to provide (name) information regarding processes limited the step
    if (!onBoundary) {
      switch (thePrimaryTrack->GetWinnerProcessIndex()) {
        case  0: theG4PostStepPoint->SetProcessDefinedStep(fElectronNoProcessVector[0]);
                 break;
        case  1: theG4PostStepPoint->SetProcessDefinedStep(fElectronNoProcessVector[1]);
                 break;
        case  2: theG4PostStepPoint->SetProcessDefinedStep(fElectronNoProcessVector[2]);
                 break;
        case -1: theG4PostStepPoint->SetProcessDefinedStep(fElectronNoProcessVector[0]);
                 break;
        case -2: theG4PostStepPoint->SetProcessDefinedStep(fElectronNoProcessVector[3]);
                 break;
      }
    }
  }
  fParticleChange->ProposeTrueStepLength(pStepLength);

  // energy, e-depo, momentum direction and status
  const double ekin = thePrimaryTrack->GetEKin();
  double edep = thePrimaryTrack->GetEnergyDeposit();
  fParticleChange->ProposeEnergy(ekin);
  if (ekin<=0.0) {
    fParticleChange->ProposeTrackStatus(fStopAndKill);
  }
  // apply MSC displacement if its length is lonegr than a minimum and we are not on boudnry
  bool isRelocate  = false;
  G4ThreeVector position = theG4PostStepPoint->GetPosition();
  if (!isGamma && !onBoundary) {
    const double* displacement = theTLData->GetPrimaryElectronTrack()->GetMSCTrackData()->GetDisplacement();
    const double dLength2 = displacement[0]*displacement[0] + displacement[1]*displacement[1] + displacement[2]*displacement[2];
    const double kGeomMinLength  = 5.0e-8;  // 0.05 [nm]
    const double kGeomMinLength2 = kGeomMinLength*kGeomMinLength; // (0.05 [nm])^2
//    G4ThreeVector position = step.GetPostStepPoint()->GetPosition();
    if (dLength2 > kGeomMinLength2) {
        // apply displacement
        bool isPositionChanged  = true;
        const double      dispR = std::sqrt(dLength2);
        const double postSafety = 0.99*fSafetyHelper->ComputeSafety(position, dispR);
        const G4ThreeVector theDisplacement(displacement[0], displacement[1], displacement[2]);
        // far away from geometry boundary
        if (postSafety > 0.0 && dispR <= postSafety) {
          position += theDisplacement;
          //near the boundary
        } else {
          // displaced point is definitely within the volume
          if (dispR < postSafety) {
            position += theDisplacement;
            // reduced displacement
          } else if(postSafety > kGeomMinLength) {
            position += theDisplacement*(postSafety/dispR);
            // very small postSafety
          } else {
            isPositionChanged = false;
          }
        }
        isRelocate = isPositionChanged;
        if (isPositionChanged) {
          fSafetyHelper->ReLocateWithinVolume(position);
          fParticleChange->ProposePosition(position);
        }
    }
  }
  //
  const double* pdir = thePrimaryTrack->GetDirection();
  fParticleChange->ProposeMomentumDirection(G4ThreeVector(pdir[0], pdir[1], pdir[2]));

  const int g4IMC = step.GetPreStepPoint()->GetMaterialCutsCouple()->GetIndex();
  // secondary: only possible is e-/e+ or gamma at the moemnt
  const int numSecElectron = theTLData->GetNumSecondaryElectronTrack();
  const int numSecGamma    = theTLData->GetNumSecondaryGammaTrack();
  const int numSecondaries = numSecElectron+numSecGamma;
  if (numSecondaries>0) {
    fParticleChange->SetNumberOfSecondaries(numSecondaries);
    const G4ThreeVector& theG4PostStepPointPosition = isRelocate ? position : theG4PostStepPoint->GetPosition();
    const G4double          theG4PostStepGlobalTime = theG4PostStepPoint->GetGlobalTime();
    const G4TouchableHandle&   theG4TouchableHandle = track.GetTouchableHandle();
    for (int is=0; is<numSecElectron; ++is) {
      G4HepEmTrack* secTrack = theTLData->GetSecondaryElectronTrack(is)->GetTrack();
      const double secEKin   = secTrack->GetEKin();
      const bool isElectron  = secTrack->GetCharge() < 0.0;
      if (applyCuts) {
        if (isElectron && secEKin < (*theCutsElectron)[g4IMC]) {
          edep += secEKin;
          continue;
        } else if (!isElectron && CLHEP::electron_mass_c2 < (*theCutsGamma)[g4IMC] &&
                   secEKin < (*theCutsPositron)[g4IMC]) {
          edep += secEKin + 2 * CLHEP::electron_mass_c2;
          continue;
        }
      }

      const double*      dir = secTrack->GetDirection();
      const G4ParticleDefinition* partDef = G4Electron::Definition();
      if (!isElectron) {
        partDef = G4Positron::Definition();
      }
      G4DynamicParticle*  dp = new G4DynamicParticle( partDef, G4ThreeVector( dir[0], dir[1], dir[2] ), secEKin );
      G4Track*     aG4Track  = new G4Track( dp, theG4PostStepGlobalTime, theG4PostStepPointPosition );
      aG4Track->SetTouchableHandle( theG4TouchableHandle );
      fParticleChange->AddSecondary( aG4Track );
    }
    theTLData->ResetNumSecondaryElectronTrack();

    for (int is=0; is<numSecGamma; ++is) {
      G4HepEmTrack* secTrack = theTLData->GetSecondaryGammaTrack(is)->GetTrack();
      const double secEKin   = secTrack->GetEKin();
      if (applyCuts && secEKin < (*theCutsGamma)[g4IMC]) {
        edep += secEKin;
        continue;
      }

      const double*      dir = secTrack->GetDirection();
      G4DynamicParticle*  dp = new G4DynamicParticle( G4Gamma::Definition(), G4ThreeVector( dir[0], dir[1], dir[2] ), secEKin );
      G4Track*     aG4Track  = new G4Track(  dp, theG4PostStepGlobalTime, theG4PostStepPointPosition );
      aG4Track->SetTouchableHandle( theG4TouchableHandle );
      fParticleChange->AddSecondary( aG4Track );
    }
    theTLData->ResetNumSecondaryGammaTrack();
  }

  fParticleChange->ProposeLocalEnergyDeposit(edep);

  return fParticleChange;
}


G4VProcess* G4HepEmProcess::GetProcess(const G4String& procname) {
  for (std::size_t ip=0; ip<fElectronNoProcessVector.size(); ++ip) {
    if (fElectronNoProcessVector[ip] && fElectronNoProcessVector[ip]->GetProcessName() == procname) {
      return fElectronNoProcessVector[ip];
    }
  }
  for (std::size_t ip=0; ip<fGammaNoProcessVector.size(); ++ip) {
    if (fGammaNoProcessVector[ip] && fGammaNoProcessVector[ip]->GetProcessName() == procname) {
      return fGammaNoProcessVector[ip];
    }
  }
  return nullptr;
}


void G4HepEmProcess::StreamInfo(std::ostream& out, const G4ParticleDefinition& part) const  {
  out << std::setprecision(6);
  out << G4endl << GetProcessName()  << ": for " << part.GetParticleName();
  out << "  More later! " << G4endl;
}
