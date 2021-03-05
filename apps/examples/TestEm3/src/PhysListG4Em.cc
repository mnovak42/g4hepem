
#include "PhysListG4Em.hh"

#include "G4ParticleDefinition.hh"
#include "G4ProcessManager.hh"
#include "G4PhysicsListHelper.hh"

#include "G4ComptonScattering.hh"
//#include "G4KleinNishinaModel.hh"  // by defult in G4ComptonScattering

#include "G4GammaConversion.hh"

#include "GammaAbsorption.hh"
#include "G4PhotoElectricEffect.hh"
#include "G4LivermorePhotoElectricModel.hh"

//#include "G4RayleighScattering.hh"

#include "G4eMultipleScattering.hh"
#include "G4GoudsmitSaundersonMscModel.hh"
#include "G4eIonisation.hh"
#include "G4eBremsstrahlung.hh"
#include "G4eplusAnnihilation.hh"


#include "G4EmParameters.hh"
#include "G4MscStepLimitType.hh"

#include "G4BuilderType.hh"
#include "G4LossTableManager.hh"
//#include "G4UAtomicDeexcitation.hh"

#include "G4SystemOfUnits.hh"


PhysListG4Em::PhysListG4Em(const G4String& name) : G4VPhysicsConstructor(name) {
  G4EmParameters* param = G4EmParameters::Instance();
  param->SetDefaults();
  param->SetVerbose(1);
  // inactivate energy loss fluctuations
  param->SetLossFluctuations(false);
//  param->SetEnableSamplingTable(true);
  // set min/max energy for tables: 100 eV - 100 TeV by default
  //param->SetMinEnergy(100*eV);
  //param->SetMaxEnergy(100*TeV);
  // set lowest kinetic energy i.e. tracking cut for charged particles having energy loss process: 1 keV by default
  // param->SetLowestElectronEnergy(1.*keV);
  // activate/inactivate integral approach: true by default
  // param->SetIntegral(true);
  // inactivate to use cuts as final range
  param->SetUseCutAsFinalRange(false);

  //
  // MSC options and parameters: 3 different stepping algorithms (can be set from the G4 macro)
  // 1. fUseSafety: opt0 step limit [corresponds to G4-Urban fUseSafety]
  param->SetMscStepLimitType(fUseSafety);
  // 2. fUseDistanceToBoundary: opt3 step limit [corresponds to G4-Urban fUseDistanceToBoundary]
  // param->SetMscStepLimitType(fUseDistanceToBoundary);
  // 3. fUseSafetyPlus: error free G4-GS stepping [there is no corresponding G4-Urban]
  // param->SetMscStepLimitType(fUseSafetyPlus);
  // Skin depth: times elastic mean free path skin near boundaries (can be set from the G4 macro)
  // - used by the G4-GS model when fUseDistanceToBoundary and fUseSafety stepping is set)
  param->SetMscSkin(3);
  // Range factor: (can be set from the G4 macro)
  param->SetMscRangeFactor(0.06);
  //
  SetPhysicsType(bElectromagnetic);
}


PhysListG4Em::~PhysListG4Em() {}


void PhysListG4Em::ConstructProcess() {
  G4PhysicsListHelper* ph = G4PhysicsListHelper::GetPhysicsListHelper();

  // Add standard EM Processes
  //
  auto aParticleIterator = GetParticleIterator();
  aParticleIterator->reset();
  while( (*aParticleIterator)() ){
    G4ParticleDefinition* particle = aParticleIterator->value();
    G4String particleName = particle->GetParticleName();

    if (particleName == "gamma") {
      ph->RegisterProcess(new G4ComptonScattering(), particle);

      ph->RegisterProcess(new G4GammaConversion, particle);

      ph->RegisterProcess(new GammaAbsorption, particle);

/*
      G4double LivermoreLowEnergyLimit = 1*eV;
      G4double LivermoreHighEnergyLimit = 1*TeV;
      G4PhotoElectricEffect* thePhotoElectricEffect = new G4PhotoElectricEffect();
      G4LivermorePhotoElectricModel* theLivermorePhotoElectricModel = new G4LivermorePhotoElectricModel();
      theLivermorePhotoElectricModel->SetLowEnergyLimit(LivermoreLowEnergyLimit);
      theLivermorePhotoElectricModel->SetHighEnergyLimit(LivermoreHighEnergyLimit);
      thePhotoElectricEffect->AddEmModel(0, theLivermorePhotoElectricModel);
      ph->RegisterProcess(thePhotoElectricEffect, particle);
*/
    } else if (particleName == "e-") {

// //     ph->RegisterProcess(new G4eMultipleScattering(), particle);

/*
      G4eMultipleScattering* msc = new G4eMultipleScattering;
      G4GoudsmitSaundersonMscModel* msc1 = new G4GoudsmitSaundersonMscModel();
      msc->AddEmModel(0, msc1);
      ph->RegisterProcess(msc,particle);

*/



      //
      G4eIonisation* eIoni = new G4eIonisation();
      ph->RegisterProcess(eIoni, particle);

      ph->RegisterProcess(new G4eBremsstrahlung(), particle);

    } else if (particleName == "e+") {
//      ph->RegisterProcess(new G4eMultipleScattering(), particle);

/*
      G4eMultipleScattering* msc = new G4eMultipleScattering;
      G4GoudsmitSaundersonMscModel* msc1 = new G4GoudsmitSaundersonMscModel();
      msc->AddEmModel(0, msc1);
      ph->RegisterProcess(msc,particle);
*/


      G4eIonisation* eIoni = new G4eIonisation();
      ph->RegisterProcess(eIoni, particle);
      ph->RegisterProcess(new G4eBremsstrahlung(), particle);
      ph->RegisterProcess(new G4eplusAnnihilation(), particle);

    }
  }


  // Deexcitation
  //
//  G4VAtomDeexcitation* de = new G4UAtomicDeexcitation();
//  G4LossTableManager::Instance()->SetAtomDeexcitation(de);
}
