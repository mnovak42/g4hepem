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
/// \file electromagnetic/TestEm3/src/PhysicsList.cc
/// \brief Implementation of the PhysicsList class
//
// $Id: PhysicsList.cc 95227 2016-02-01 09:19:15Z gcosmo $
//
//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......
//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

#include "PhysicsList.hh"
#include "PhysicsListMessenger.hh"

#include "PhysListEmStandard.hh"
#include "PhysListHepEm.hh"
#include "PhysListG4Em.hh"

#include "G4Version.hh"
#if G4VERSION_NUMBER >= 1100
#include "PhysListHepEmTracking.hh"
#include "PhysListG4EmTracking.hh"
#endif

#include "G4EmStandardPhysics.hh"
#include "G4EmStandardPhysics_option1.hh"
#include "G4EmStandardPhysics_option2.hh"

#include "G4EmExtraPhysics.hh"
#include "G4HadronicProcessStore.hh"
#if G4VERSION_NUMBER >= 1100
#include "G4HadronicParameters.hh"
#endif

#include "G4HadronicParameters.hh"

#include "G4LossTableManager.hh"

#include "G4UnitsTable.hh"
#include "G4SystemOfUnits.hh"

// particles

#include "G4BosonConstructor.hh"
#include "G4LeptonConstructor.hh"
#include "G4MesonConstructor.hh"
#include "G4BosonConstructor.hh"
#include "G4BaryonConstructor.hh"
#include "G4IonConstructor.hh"
#include "G4ShortLivedConstructor.hh"


// HepEm: the G4HepEm EM phyics constructor is (see PhysListHepEm)
// G4Em : the G4 EM phyics constructor, that is equivalent to the G4HepEm, is (see PhysListG4Em)

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

PhysicsList::PhysicsList() : G4VModularPhysicsList(),
 fEmPhysicsList(0), fMessenger(0)
{
  G4LossTableManager::Instance();
  SetDefaultCutValue(1*mm);

  fMessenger = new PhysicsListMessenger(this);
  verboseLevel = 0;

#if G4HepEm_HAS_G4VTRACKINGMANAGER
  // make the `G4HepEmTrackingManager` the default whenever it's available
  fEmName = G4String("HepEmTracking");
  fEmPhysicsList = new PhysListHepEmTracking(fEmName);
#else
  // use the process interface but only as a backup solution as not efficient (g4<11.0)
  fEmName        = G4String("HepEm");
  fEmPhysicsList = new PhysListHepEm(fEmName);
#endif

  fEmPhysicsList->SetVerboseLevel(verboseLevel);
  G4EmParameters::Instance()->SetVerbose(verboseLevel);

  // Hardonic verbose needs to be set before construction
#if G4VERSION_NUMBER >= 1100
  G4HadronicParameters::Instance()->SetVerboseLevel(verboseLevel);
#endif
  G4HadronicProcessStore::Instance()->SetVerbose(verboseLevel);

  // Create the G4EmExtraPhysics to add gamma and lepton nuclear interactions
  G4EmExtraPhysics* emExtra = new G4EmExtraPhysics(verboseLevel);
  // During the development: deactiavte electron nuclear till we don't have in HepEm
  // emExtra->ElectroNuclear(false);
  // Turn off muon nuclear as well (not improtant as no muon production but
  // remove it as we don't have in HepEm)
  emExtra->MuonNuclear(false);

  fEmExtraPhysics = emExtra;

}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

PhysicsList::~PhysicsList()
{
  delete fEmPhysicsList;
  delete fEmExtraPhysics;
  delete fMessenger;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void PhysicsList::ConstructParticle()
{
    G4BosonConstructor  pBosonConstructor;
    pBosonConstructor.ConstructParticle();

    G4LeptonConstructor pLeptonConstructor;
    pLeptonConstructor.ConstructParticle();

    G4MesonConstructor pMesonConstructor;
    pMesonConstructor.ConstructParticle();

    G4BaryonConstructor pBaryonConstructor;
    pBaryonConstructor.ConstructParticle();

    G4IonConstructor pIonConstructor;
    pIonConstructor.ConstructParticle();

    G4ShortLivedConstructor pShortLivedConstructor;
    pShortLivedConstructor.ConstructParticle();

    fEmExtraPhysics->ConstructParticle();
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void PhysicsList::ConstructProcess()
{
  // Transportation
  AddTransportation();

  // Electromagnetic Physics List
  fEmPhysicsList->SetVerboseLevel(verboseLevel);

  fEmPhysicsList->ConstructProcess();
  // EM extra physics, i.e. gamma end lepton nuclear
  fEmExtraPhysics->SetVerboseLevel(verboseLevel);
  fEmExtraPhysics->ConstructProcess();
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void PhysicsList::AddPhysicsList(const G4String& name)
{
  if (verboseLevel>1) {
    G4cout << "PhysicsList::AddPhysicsList: <" << name << ">" << G4endl;
  }

  if (name == fEmName) return;

  if (name == "local") {

    fEmName = name;
    delete fEmPhysicsList;
    fEmPhysicsList = new PhysListEmStandard(name);

  } else if (name == "HepEm") {

    fEmName = name;
    delete fEmPhysicsList;
    fEmPhysicsList = new PhysListHepEm();

#if G4VERSION_NUMBER >= 1100
  } else if (name == "HepEmTracking") {

    fEmName = name;
    delete fEmPhysicsList;
    fEmPhysicsList = new PhysListHepEmTracking();
#endif

  } else if (name == "G4Em") {

    fEmName = name;
    delete fEmPhysicsList;
    fEmPhysicsList = new PhysListG4Em();

#if G4VERSION_NUMBER >= 1100
  } else if (name == "G4EmTracking") {

    fEmName = name;
    delete fEmPhysicsList;
    fEmPhysicsList = new PhysListG4EmTracking();
#endif

  } else if (name == "emstandard_opt0") {

    fEmName = name;
    delete fEmPhysicsList;
    fEmPhysicsList = new G4EmStandardPhysics();

  } else if (name == "emstandard_opt1") {

    fEmName = name;
    delete fEmPhysicsList;
    fEmPhysicsList = new G4EmStandardPhysics_option1();

  } else {

    G4cout << "PhysicsList::AddPhysicsList: <" << name << ">"
           << " is not defined"
           << G4endl;
  }
  fEmPhysicsList->SetVerboseLevel(verboseLevel);
  G4EmParameters::Instance()->SetVerbose(verboseLevel);
}
