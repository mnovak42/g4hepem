
#include "Declaration.hh"

#include "TestUtils/Hist.hh"

// G4 includes
#include "G4SystemOfUnits.hh"
#include "G4ParticleDefinition.hh"
#include "G4ParticleTable.hh"
#include "G4Electron.hh"
#include "G4Positron.hh"
#include "G4Gamma.hh"
#include "G4ProductionCutsTable.hh"

#include "G4StateManager.hh"
#include "G4TouchableHandle.hh"
#include "G4TouchableHistory.hh"
#include "G4SeltzerBergerModel.hh"
#include "G4eBremsstrahlungRelModel.hh"
#include "G4ParticleChangeForLoss.hh"

#include "G4Timer.hh"


// G4HepEm includes
#include "G4HepEmData.hh"
#include "G4HepEmTLData.hh"
#include "G4HepEmMatCutData.hh"

#include "G4HepEmRunManager.hh"
#include "G4HepEmElectronInteractionBrem.hh"

#include "G4HepEmTrack.hh"

#include <vector>
#include <cmath>

void G4SBTest(const G4MaterialCutsCouple* g4MatCut, G4double ekin, G4double numSamples, G4int numHistBins, G4bool isSBmodel, G4bool iselectron) {
  //
  // --- Get primary particle: e- or e+
  G4Electron::Electron();
  G4Positron::Positron();
  G4ParticleDefinition *part = G4Positron::Positron();
  if (iselectron) {
    part = G4Electron::Electron();
  }
  //
  // --- Get the `cuts` data vector in energy
  const G4DataVector* cuts = static_cast<const G4DataVector*>(G4ProductionCutsTable::GetProductionCutsTable()->GetEnergyCutsVector(0));
  G4double gammaCutEnergy = (*cuts)[g4MatCut->GetIndex()]; // should be [0]
  //
  // --- Start run processing
  G4StateManager *g4State = G4StateManager::GetStateManager();
  if (!g4State->SetNewState(G4State_Init)) {
    G4cout << "error changing G4state" << G4endl;
  }
  //
  // --- Initilize the model
  //  G4EmParameters::Instance()->Dump();
  //
  G4VEmModel*                       theModel = nullptr;
  G4ParticleChangeForLoss* theParticleChange = nullptr;
  if (isSBmodel) {
    // create the G4 SB-brem model and set up
    theModel = new G4SeltzerBergerModel;
    // bremSB.SetBicubicInterpolationFlag(true);
  } else {
    theModel = new G4eBremsstrahlungRelModel;
  }
  theParticleChange = new G4ParticleChangeForLoss();
  theModel->SetParticleChange(theParticleChange, 0);
  theModel->Initialise(part, *cuts);


  //
  // --- Create a dynamic particle and track for the pirmay
  G4ThreeVector aPosition  = G4ThreeVector(0.0, 0.0, 0.0);
  G4ThreeVector aDirection = G4ThreeVector(0.0, 0.0, 1.0);
  G4DynamicParticle dParticle(part, aDirection, ekin);
  G4Track *gTrack = new G4Track(&dParticle, 0.0, aPosition);
  G4TouchableHandle fpTouchable(new G4TouchableHistory());
  gTrack->SetTouchableHandle(fpTouchable);
  const G4Track *track = const_cast<G4Track *>(gTrack);
  //
  // --- Step
  if (!G4StateManager::GetStateManager()->SetNewState(G4State_Idle)) {
    G4cout << "G4StateManager PROBLEM! " << G4endl;
  }
  // --- Event loop
  std::vector<G4DynamicParticle *> vdp;
  vdp.reserve(1);
//  dParticle.SetKineticEnergy(ekin);

  // print outs
  G4cout << g4MatCut->GetMaterial() << G4endl;
  G4ProductionCutsTable::GetProductionCutsTable()->DumpCouples();
  G4cout << "   -------------------------------------------------------------------------------- " << G4endl;
  G4cout << "   Particle       =  " << part->GetParticleName() << G4endl;
  G4cout << "   -------------------------------------------------------------------------------- " << G4endl;
  G4cout << "   Kinetic energy =  " << ekin / MeV << "  [MeV] " << G4endl;
  G4cout << "   -------------------------------------------------------------------------------- " << G4endl;
  G4cout << "   Model name     =  " << theModel->GetName() << G4endl;

  // --- Prepare histograms:
  //
  // energy distribution(k) is sampled in varibale log10(k/primaryEnergy)
  // angular distribution(theta) is sampled in variable log10(1-cos(theta)*0.5)
  //
  // set histo name prefix
  G4String hname = "brem_SB_G4_";
  //
  // set up a histogram for the secondary gamma energy(k) : log10(k/primaryEnergy)
  G4int nbins       = numHistBins;
  G4double xmin     = std::log10(gammaCutEnergy / ekin);
  G4double xmax     = 0.1;
  G4String h1Name   = hname + "gamma_energy.dat";
  Hist* h1 = new Hist(xmin, xmax, nbins);
  //
  // set up histogram for the secondary gamma direction(theta) : log10(1-cos(theta)*0.5)
  xmin     = -12.;
  xmax     = 0.5;
  G4String h2Name  = hname + "gamma_angular.dat";
  Hist* h2 = new Hist(xmin, xmax, nbins);
  //
  // set up a histogram for the post interaction primary e-/e+ energy(E1) : log10(E1/primaryEnergy)
  xmin     = -12.;
  xmax     = 0.1;
  G4String h3Name   = hname + "energy.dat";
  Hist* h3 = new Hist(xmin, xmax, nbins);
  //
  // set up a histogram for the post interaction primary e-/e+ direction(theta) : log10(1-cos(theta)*0.5)
  xmin     = -16.;
  xmax     = 0.5;
  G4String h4Name   =  hname + "angular.dat";
  Hist* h4 = new Hist(xmin, xmax, nbins);

  // start sampling
  G4cout << "   -------------------------------------------------------------------------------- " << G4endl;
  G4cout << "   Sampling is running : .........................................................  " << G4endl;
  // Sampling

  // SB ---------------------------
  G4double timeInSec = 0.0;
  G4Timer *timer     = new G4Timer();
  timer->Start();
  for (long int iter = 0; iter < numSamples; ++iter) {
    theParticleChange->InitializeForPostStep(*track);
    theModel->SampleSecondaries(&vdp, g4MatCut, &dParticle, gammaCutEnergy, ekin);
    // if there is any secondary gamma then get it
    if (vdp.size() > 0) {
      // reduced gamma energy
      G4double eGamma = vdp[0]->GetKineticEnergy() / ekin; // k/E_prim
      if (eGamma > 0.0) {
        h1->Fill(std::log10(eGamma), 1.0);
      }
      G4double costGamma = vdp[0]->GetMomentumDirection().z();
      costGamma          = 0.5 * (1.0 - costGamma);
      if (costGamma > 0.0) {
        costGamma = std::log10(costGamma);
        if (costGamma > -12.) {
          h2->Fill(costGamma, 1.0);
        }
      }
      // go for the post interaction primary
      G4double ePrim = theParticleChange->GetProposedKineticEnergy() / ekin;
      if (ePrim > 0.0) {
        ePrim = std::log10(ePrim);
        if (ePrim > -12.0) {
          h3->Fill(ePrim, 1.0);
        }
      }
      G4double costPrim = theParticleChange->GetProposedMomentumDirection().z();
      costPrim          = 0.5 * (1.0 - costPrim);
      if (costPrim > 0.0) {
        costPrim = std::log10(costPrim);
        if (costPrim > -16.) {
          h4->Fill(costPrim, 1.0);
        }
      }
      delete vdp[0];
      vdp.clear();
    }
  }
  timer->Stop();
  timeInSec = timer->GetRealElapsed();
  delete timer;

  //
  G4cout << "   -------------------------------------------------------------------------------- " << G4endl;
  G4cout << "   Time of sampling =  " << timeInSec << " [s]" << G4endl;
  G4cout << "   -------------------------------------------------------------------------------- " << G4endl;

  // --- Write histograms
  h1->Write(h1Name, 0.25 / numSamples);
  h2->Write(h2Name, 1.0  / numSamples);
  h3->Write(h3Name, 0.25 / numSamples);
  h4->Write(h4Name, 1.0  / numSamples);


  delete h1;
  delete h2;
  delete h3;
  delete h4;
}





void G4HepEmSBTest(const G4MaterialCutsCouple* g4MatCut, G4double ekin, G4double numSamples, G4int numHistBins, G4bool isSBmodel, G4bool iselectron) {
  //
  // Get the maser G4HepEmRunManager (already initialised in the main) then:
  //  - get the pointer to the global G4HepEmData structure (already initialised)
  //  - get the pointer to the corresponding G4HepEmTLData
  const G4HepEmRunManager* theRunMgr    = G4HepEmRunManager::GetMasterRunManager();
  G4HepEmData*             theHepEmData = theRunMgr->GetHepEmData();
  G4HepEmTLData*           theTLData    = theRunMgr->GetTheTLData();
  //
  // set primary particle related data in the G4HepEmTLData
  G4HepEmTrack*         thePrimaryTrack = theTLData->GetPrimaryElectronTrack()->GetTrack();
  double charge      = iselectron ? -1.0 : 1.0;
  thePrimaryTrack->SetCharge(charge);
  const double lekin = std::log(ekin);
  thePrimaryTrack->SetEKin(ekin, lekin);
  const int    g4IMC = g4MatCut->GetIndex();
  const int hepEmIMC = theRunMgr->GetHepEmData()->fTheMatCutData->fG4MCIndexToHepEmMCIndex[g4IMC];
  // these two will be updated to the post-interaction values so need to be re-set each time
  thePrimaryTrack->SetMCIndex(hepEmIMC);
  thePrimaryTrack->SetDirection(0.0, 0.0, 1.0);
  //
  double gammaCutEnergy = theRunMgr->GetHepEmData()->fTheMatCutData->fMatCutData[hepEmIMC].fSecGamProdCutE;
  //
  // Create Histograms
  // - set histo name prefix
  std::string hname = "brem_SB_G4HepEm_";
  // set up a histogram for the secondary gamma energy(k) : log10(k/primaryEnergy)
  int nbins       = numHistBins;
  double xmin     = std::log10(gammaCutEnergy / ekin);
  double xmax     = 0.1;
  std::string h1Name   = hname + "gamma_energy.dat";
  Hist* h1 = new Hist(xmin, xmax, nbins);
  //
  // set up histogram for the secondary gamma direction(theta) : log10(1-cos(theta)*0.5)
  xmin     = -12.;
  xmax     = 0.5;
  std::string h2Name  = hname + "gamma_angular.dat";
  Hist* h2 = new Hist(xmin, xmax, nbins);
  //
  // set up a histogram for the post interaction primary e-/e+ energy(E1) : log10(E1/primaryEnergy)
  xmin     = -12.;
  xmax     = 0.1;
  std::string h3Name   = hname + "energy.dat";
  Hist* h3 = new Hist(xmin, xmax, nbins);
  //
  // set up a histogram for the post interaction primary e-/e+ direction(theta) : log10(1-cos(theta)*0.5)
  xmin     = -16.;
  xmax     = 0.5;
  std::string h4Name   =  hname + "angular.dat";
  Hist* h4 = new Hist(xmin, xmax, nbins);
  // start sampling
  std::cout << "   -------------------------------------------------------------------------------- " << std::endl;
  std::cout << "   Sampling is running : .........................................................  " << std::endl;
  // Sampling
  // SB ---------------------------
  double timeInSec = 0.0;
  G4Timer *timer = new G4Timer();
  timer->Start();
  for (long int iter = 0; iter < numSamples; ++iter) {
    thePrimaryTrack->SetEKin(ekin, lekin);
    thePrimaryTrack->SetDirection(0.0, 0.0, 1.0);
    // invoke SB brem intercation from G4HepEmElectronInteractionBrem.hh
    PerformElectronBrem(theTLData, theHepEmData, iselectron, isSBmodel);
    // get secondary related results (energy, direction) if any
    const int numSecGamma = theTLData->GetNumSecondaryGammaTrack();
    if (numSecGamma > 0) {
      G4HepEmTrack* secTrack = theTLData->GetSecondaryGammaTrack(0)->GetTrack();
      // reduced gamma energy
      double eGamma = secTrack->GetEKin() / ekin; // k/E_prim
      if (eGamma > 0.0) {
        h1->Fill(std::log10(eGamma), 1.0);
      }
      double costGamma = secTrack->GetDirection()[2];
      costGamma        = 0.5 * (1.0 - costGamma);
      if (costGamma > 0.0) {
        costGamma = std::log10(costGamma);
        if (costGamma > -12.) {
          h2->Fill(costGamma, 1.0);
        }
      }
      // release used secondary track buffer
      theTLData->ResetNumSecondaryGammaTrack();
    }
    // get post-interaction primary related results (energy, direction)
    double ePrim = thePrimaryTrack->GetEKin() / ekin;
    if (ePrim > 0.0) {
      ePrim = std::log10(ePrim);
      if (ePrim > -12.0) {
        h3->Fill(ePrim, 1.0);
      }
    }
    double costPrim = thePrimaryTrack->GetDirection()[2];
    costPrim        = 0.5 * (1.0 - costPrim);
    if (costPrim > 0.0) {
      costPrim = std::log10(costPrim);
      if (costPrim > -16.) {
        h4->Fill(costPrim, 1.0);
      }
    }
  }
  timer->Stop();
  timeInSec = timer->GetRealElapsed();
  delete timer;
  //
  std::cout << "   -------------------------------------------------------------------------------- " << std::endl;
  std::cout << "   Time of sampling =  " << timeInSec << " [s]" << std::endl;
  std::cout << "   -------------------------------------------------------------------------------- " << std::endl;

  // --- Write histograms
  h1->Write(h1Name, 0.25 / numSamples);
  h2->Write(h2Name, 1.0  / numSamples);
  h3->Write(h3Name, 0.25 / numSamples);
  h4->Write(h4Name, 1.0  / numSamples);


  delete h1;
  delete h2;
  delete h3;
  delete h4;
}
