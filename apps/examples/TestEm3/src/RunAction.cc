//
// ********************************************************************
// * License and Disclaimer                                           *
// *                                                                  *
// * The  Geant4 software  is  copyright of the Copyright Holders  of *
// * the Geant4 Collaboration.  It is provided  under  the terms  and *
// * conditions of the Geant4 Software License,  included in the file *
// * LICENSE and available at  http://cern.ch/geant4/license .  These *
// * include a list of copyright holders.                             *
// *                                                                  *
// * Neither the authors of this software system, nor their employing *
// * institutes,nor the agencies providing financial support for this *
// * work  make  any representation or  warranty, express or implied, *
// * regarding  this  software system or assume any liability for its *
// * use.  Please see the license in the file  LICENSE  and URL above *
// * for the full disclaimer and the limitation of liability.         *
// *                                                                  *
// * This  code  implementation is the result of  the  scientific and *
// * technical work of the GEANT4 collaboration.                      *
// * By using,  copying,  modifying or  distributing the software (or *
// * any work based  on the software)  you  agree  to acknowledge its *
// * use  in  resulting  scientific  publications,  and indicate your *
// * acceptance of all terms of the Geant4 Software license.          *
// ********************************************************************
//
/// \file electromagnetic/TestEm3/src/RunAction.cc
/// \brief Implementation of the RunAction class
//
// $Id: RunAction.cc 90969 2015-06-12 08:12:57Z gcosmo $
//
//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......
//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

#include "RunAction.hh"

#include "DetectorConstruction.hh"
#include "PrimaryGeneratorAction.hh"
#include "RunActionMessenger.hh"
#include "HistoManager.hh"
#include "Run.hh"
#include "G4Timer.hh"
#include "G4RunManager.hh"
#include "Randomize.hh"

#include "G4ProductionCutsTable.hh"
//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

RunAction::RunAction(DetectorConstruction* det, PrimaryGeneratorAction* prim)
:G4UserRunAction(), fIsPerformance(false), fDetector(det), fPrimary(prim), fRun(0), fRunMessenger(0),
 fHistoManager(0), fTimer(0)
{
  fRunMessenger = new RunActionMessenger(this);
  fHistoManager = new HistoManager();
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

RunAction::~RunAction()
{
  delete fRunMessenger;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

G4Run* RunAction::GenerateRun()
{
  if (!fIsPerformance) {
    fRun = new Run(fDetector);
    return fRun;
  }
  return nullptr;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void RunAction::BeginOfRunAction(const G4Run*)
{
  if (isMaster) {
    //    G4Random::showEngineStatus();
    G4ProductionCutsTable::GetProductionCutsTable()->DumpCouples();

    fTimer = new G4Timer();
    fTimer->Start();
  }

  if (fIsPerformance) {
    return;
  }

  // keep run condition
  if (fPrimary) {
    G4ParticleDefinition* particle
      = fPrimary->GetParticleGun()->GetParticleDefinition();
    G4double energy = fPrimary->GetParticleGun()->GetParticleEnergy();
    fRun->SetPrimary(particle, energy);
  }

  //histograms
  //
  G4AnalysisManager* analysis = G4AnalysisManager::Instance();
  if (analysis->IsActive()) analysis->OpenFile();

  // save Rndm status and open the timer

}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void RunAction::EndOfRunAction(const G4Run*)
{
  // compute and print statistic
  if (isMaster) {
    fTimer->Stop();
    G4cout << "  ======================================================" << G4endl;
    G4cout << "   Time:  "  << *fTimer << G4endl;
    G4cout << "  ======================================================" << G4endl;
    delete fTimer;
    if (!fIsPerformance) {
      fRun->EndOfRun();
    } else {
      return;
    }
  }
  //save histograms
  G4AnalysisManager* analysis = G4AnalysisManager::Instance();
  if (analysis->IsActive()) {
    analysis->Write();
    analysis->CloseFile();
  }

  // show Rndm status
  //  if (isMaster)  G4Random::showEngineStatus();
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void RunAction::SetEdepAndRMS(G4int i,G4double edep,G4double rms,G4double lim)
{
  if (fRun) fRun->SetEdepAndRMS(i, edep, rms, lim);
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void RunAction::SetApplyLimit(G4bool val)
{
  if (fRun) fRun->SetApplyLimit(val);
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......
