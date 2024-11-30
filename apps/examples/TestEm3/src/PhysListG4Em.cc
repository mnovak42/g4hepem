
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
//
// M.Novak: G4HepEm like EM constructor

#include "PhysListG4Em.hh"
#include "G4ParticleDefinition.hh"
#include "G4EmParameters.hh"

#include "G4ComptonScattering.hh"
#include "G4GammaConversion.hh"
#include "G4PhotoElectricEffect.hh"
#include "G4GammaGeneralProcess.hh"

#include "G4eMultipleScattering.hh"
#include "G4eIonisation.hh"
#include "G4eBremsstrahlung.hh"
#include "G4eplusAnnihilation.hh"


#include "G4Gamma.hh"
#include "G4Electron.hh"
#include "G4Positron.hh"

#include "G4PhysicsListHelper.hh"
#include "G4BuilderType.hh"
#include "G4LossTableManager.hh"


PhysListG4Em::PhysListG4Em(const G4String& name)
  : G4VPhysicsConstructor(name)
{
  G4EmParameters* param = G4EmParameters::Instance();
  param->SetDefaults();

  param->SetMscRangeFactor(0.04);

  SetPhysicsType(bElectromagnetic);
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

PhysListG4Em::~PhysListG4Em()
{}


//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void PhysListG4Em::ConstructProcess()
{
  if(verboseLevel > 1) {
    G4cout << "### " << GetPhysicsName() << " Construct Processes " << G4endl;
  }
//  G4EmBuilder::PrepareEMPhysics();
  G4PhysicsListHelper* ph = G4PhysicsListHelper::GetPhysicsListHelper();

  // Add standard EM Processes
  //
  G4ParticleDefinition* particle = nullptr;

  // Add gamma EM processes
  particle = G4Gamma::Gamma();
  if (G4EmParameters::Instance()->GeneralProcessActive()) {
    // Gamma general or Woodcock process depending is the latter was set
    G4GammaGeneralProcess* sp = new G4GammaGeneralProcess;
    sp->AddEmProcess(new G4PhotoElectricEffect);
    sp->AddEmProcess(new G4ComptonScattering);
    sp->AddEmProcess(new G4GammaConversion);
    G4LossTableManager::Instance()->SetGammaGeneralProcess(sp);
    ph->RegisterProcess(sp, particle);
  } else {
    ph->RegisterProcess(new G4PhotoElectricEffect, particle);
    ph->RegisterProcess(new G4ComptonScattering, particle);
    ph->RegisterProcess(new G4GammaConversion, particle);
  }

  //
  // Add e- EM processes
  particle = G4Electron::Electron();

  ph->RegisterProcess(new G4eMultipleScattering, particle);
  ph->RegisterProcess(new G4eIonisation, particle);
  ph->RegisterProcess(new G4eBremsstrahlung, particle);

  //
  // Add e+ EM processes

  particle = G4Positron::Positron();

  ph->RegisterProcess(new G4eMultipleScattering, particle);
  ph->RegisterProcess(new G4eIonisation, particle);
  ph->RegisterProcess(new G4eBremsstrahlung, particle);
  ph->RegisterProcess(new G4eplusAnnihilation, particle);

}
