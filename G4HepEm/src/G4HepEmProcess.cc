
#include "G4HepEmProcess.hh"


#include "G4HepEmRunManager.hh"
#include "G4HepEmData.hh"
#include "G4HepEmMatCutData.hh"
#include "G4HepEmTLData.hh"
#include "G4HepEmRunManager.hh"

#include "G4HepEmElectronTrack.hh"
#include "G4HepEmGammaTrack.hh"
#include "G4HepEmElectronManager.hh"
#include "G4HepEmGammaManager.hh"

#include "G4Threading.hh"
#include "G4Track.hh"
#include "G4Step.hh"
#include "G4StepStatus.hh"
#include "G4MaterialCutsCouple.hh"
#include "G4ParticleChangeForLoss.hh"

#include "G4Electron.hh"
#include "G4Positron.hh"
#include "G4Gamma.hh"


G4HepEmProcess::G4HepEmProcess() 
: G4VProcess("hepEm", fElectromagnetic), 
  fTheG4HepEmRunManager(nullptr) {
  enableAtRestDoIt    = false;
  enablePostStepDoIt  = false;
  
  fTheG4HepEmRunManager  = new G4HepEmRunManager(G4Threading::IsMasterThread());  
  fParticleChangeForLoss = new G4ParticleChangeForLoss();
}

G4HepEmProcess::~G4HepEmProcess() {
  delete fTheG4HepEmRunManager;
}


void G4HepEmProcess::BuildPhysicsTable(const G4ParticleDefinition& partDef) {
  G4cout << " G4HepEmProcess::BuildPhysicsTable for Particle = " << partDef.GetParticleName() << G4endl;    
  StreamInfo(G4cout, partDef);

  // Extracts EM and other configuration parameters, builds all the elememnt, 
  // matrial and material-production-cuts related data structures shared by all 
  // workers and all processes (read-only) at run time.
  // Action is done only for the master runmanager and only if it was not done 
  // yet: in case of re-init, first the G4HepEmRunManager::Clear() method needs 
  // to be invoked.
//  fTheG4HepEmRunManager->InitializeGlobal();  
  
  // The ptr-s to global data structures created and filled in InitializeGlobal() 
  // will be copied to the workers and the TL-data structure will be created.
  //fTheG4HepEmRunManager->Initialize(G4Random::getTheEngine());
  
  if (partDef.GetPDGEncoding()==11) {          // e- 
    fTheG4HepEmRunManager->Initialize(G4Random::getTheEngine(), 0);
  } else if (partDef.GetPDGEncoding()==-11) {  // e+ 
    fTheG4HepEmRunManager->Initialize(G4Random::getTheEngine(), 1);
  } else if (partDef.GetPDGEncoding()==22) {   // gamma  
    fTheG4HepEmRunManager->Initialize(G4Random::getTheEngine(), 2);
  } else {
    std::cerr << " **** ERROR in G4HepEmProcess::BuildPhysicsTable: unknown particle " << std::endl;
    exit(-1); 
  }       
}


void     G4HepEmProcess::StartTracking(G4Track* track) {
    // reset number of interaction length left to -1
  const G4ParticleDefinition* partDef = track->GetParticleDefinition();  
  if (std::abs(partDef->GetPDGEncoding())==11) {          // e- and e+ 
    double* numInterALeft = fTheG4HepEmRunManager->GetTheTLData()->GetPrimaryElectronTrack()->GetTrack()->GetNumIALeft();
    numInterALeft[0] = -1.0;
    numInterALeft[1] = -1.0;
    numInterALeft[2] = -1.0;
  } else if (partDef->GetPDGEncoding()==22) {   // gamma    
    double* numInterALeft = fTheG4HepEmRunManager->GetTheTLData()->GetPrimaryGammaTrack()->GetTrack()->GetNumIALeft();
    numInterALeft[0] = -1.0;
    numInterALeft[1] = -1.0;
    numInterALeft[2] = -1.0;
//    numInterALeft[3] = -1.0;
  }  
}

G4double G4HepEmProcess::AlongStepGetPhysicalInteractionLength ( 
                                               const G4Track& track,
                                               G4double  previousStepSize,
                                               G4double  currentMinimumStep,
                                               G4double& proposedSafety,
                                               G4GPILSelection* selection) {
  *selection = CandidateForSelection;   
  G4HepEmTLData*            theTLData = fTheG4HepEmRunManager->GetTheTLData();  
  const G4ParticleDefinition* partDef = track.GetParticleDefinition();    
  const bool                 isGamma  = (partDef->GetPDGEncoding()==22);
  G4HepEmTrack*       thePrimaryTrack = isGamma
                                        ? theTLData->GetPrimaryGammaTrack()->GetTrack()
                                        : theTLData->GetPrimaryElectronTrack()->GetTrack();
  thePrimaryTrack->SetCharge(partDef->GetPDGCharge());
  thePrimaryTrack->SetEKin(track.GetDynamicParticle()->GetKineticEnergy());
  thePrimaryTrack->SetLEKin(track.GetDynamicParticle()->GetLogKineticEnergy());
  const int    g4IMC = track.GetMaterialCutsCouple()->GetIndex();
  const int hepEmIMC = fTheG4HepEmRunManager->GetHepEmData()->fTheMatCutData->fG4MCIndexToHepEmMCIndex[g4IMC];
  thePrimaryTrack->SetMCIndex(hepEmIMC);
  thePrimaryTrack->SetOnBoundary(track.GetStep()->GetPreStepPoint()->GetStepStatus()==G4StepStatus::fGeomBoundary);

  if (isGamma) {
    fTheG4HepEmRunManager->GetTheGammaManager()->HowFar(fTheG4HepEmRunManager->GetHepEmData(), fTheG4HepEmRunManager->GetHepEmParameters(), theTLData);
  } else {
    fTheG4HepEmRunManager->GetTheElectronManager()->HowFar(fTheG4HepEmRunManager->GetHepEmData(), fTheG4HepEmRunManager->GetHepEmParameters(), theTLData);
  }
  // returns with the geometrcal step length: straight line distance to make along the org direction
  return thePrimaryTrack->GetGStepLength();
}




G4VParticleChange* G4HepEmProcess::AlongStepDoIt( const G4Track& track, const G4Step& step) {
  // init particle change: it might be more special we need to see later
  fParticleChangeForLoss->InitializeForPostStep(track);
  
  G4HepEmTLData*            theTLData = fTheG4HepEmRunManager->GetTheTLData();
  const G4ParticleDefinition* partDef = track.GetParticleDefinition();    
  const bool                  isGamma = (partDef->GetPDGEncoding()==22);
  G4HepEmTrack*       thePrimaryTrack = isGamma
                                        ? theTLData->GetPrimaryGammaTrack()->GetTrack()
                                        : theTLData->GetPrimaryElectronTrack()->GetTrack();
  // NOTE: this primary track is the same as in the last call in the HowFar()
  //       But transportation might changed its direction, geomertical step length,
  //       or status ( on boundary or not).
  const G4ThreeVector& primDir = track.GetDynamicParticle()->GetMomentumDirection();
  thePrimaryTrack->SetDirection(primDir[0], primDir[1], primDir[2]);
  thePrimaryTrack->SetGStepLength(track.GetStepLength());
  thePrimaryTrack->SetOnBoundary(track.GetStep()->GetPostStepPoint()->GetStepStatus()==G4StepStatus::fGeomBoundary);
  if (isGamma) {
    fTheG4HepEmRunManager->GetTheGammaManager()->Perform(fTheG4HepEmRunManager->GetHepEmData(), fTheG4HepEmRunManager->GetHepEmParameters(), theTLData);
  } else {
    fTheG4HepEmRunManager->GetTheElectronManager()->Perform(fTheG4HepEmRunManager->GetHepEmData(), fTheG4HepEmRunManager->GetHepEmParameters(), theTLData);
  }
  // energy, e-depo, momentum direction and status
  const double ekin = thePrimaryTrack->GetEKin();
  const double edep = thePrimaryTrack->GetEnergyDeposit();
  fParticleChangeForLoss->SetProposedKineticEnergy(ekin);
  if (ekin<=0.0) {
    fParticleChangeForLoss->ProposeTrackStatus(fStopAndKill);  
  }
  fParticleChangeForLoss->ProposeLocalEnergyDeposit(edep);
  const double* pdir = thePrimaryTrack->GetDirection();
  fParticleChangeForLoss->ProposeMomentumDirection(G4ThreeVector(pdir[0], pdir[1], pdir[2]));
  
  // secondary: only possible is e- or gamma at the moemnt
  const int numSecElectron = theTLData->GetNumSecondaryElectronTrack();
  const int numSecGamma    = theTLData->GetNumSecondaryGammaTrack();
  const int numSecondaries = numSecElectron+numSecGamma;
  if (numSecondaries>0) {
    fParticleChangeForLoss->SetNumberOfSecondaries(numSecondaries);    
    double time = track.GetGlobalTime();
    for (int is=0; is<numSecElectron; ++is) {
      G4HepEmTrack* secTrack = theTLData->GetSecondaryElectronTrack(is)->GetTrack();
      const double* dir = secTrack->GetDirection();
      // MUST BE CHANGED WHEN e+ is added 
      G4DynamicParticle* dp = secTrack->GetCharge() < 0.0 
                              ? new G4DynamicParticle(G4Electron::Definition(), G4ThreeVector(dir[0], dir[1], dir[2]), secTrack->GetEKin())
                              : new G4DynamicParticle(G4Positron::Definition(), G4ThreeVector(dir[0], dir[1], dir[2]), secTrack->GetEKin());
      G4Track* t = new G4Track(dp, time, track.GetPosition());
      t->SetTouchableHandle(track.GetTouchableHandle());
      fParticleChangeForLoss->AddSecondary(t);
    }
    theTLData->ResetNumSecondaryElectronTrack();

    for (int is=0; is<numSecGamma; ++is) {
      G4HepEmTrack* secTrack = theTLData->GetSecondaryGammaTrack(is)->GetTrack();
      const double* dir = secTrack->GetDirection();
      G4DynamicParticle* dp = new G4DynamicParticle(G4Gamma::Definition(), G4ThreeVector(dir[0], dir[1], dir[2]), secTrack->GetEKin());
      G4Track* t = new G4Track(dp, time, track.GetPosition());
      t->SetTouchableHandle(track.GetTouchableHandle());
      fParticleChangeForLoss->AddSecondary(t);
    }

    theTLData->ResetNumSecondaryGammaTrack();
 } 
  
  
    
  return fParticleChangeForLoss;
} 







void G4HepEmProcess::StreamInfo(std::ostream& out, const G4ParticleDefinition& part) const  {
  out << std::setprecision(6);
  out << G4endl << GetProcessName()  << ": for " << part.GetParticleName();
  out << "  More later! " << G4endl;
}



