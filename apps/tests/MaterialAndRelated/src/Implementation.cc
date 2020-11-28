
#include "Declaration.hh"

#include "G4SystemOfUnits.hh"

#include "G4NistManager.hh"
#include "G4Material.hh"
#include "G4String.hh"

#include "G4Box.hh"
#include "G4LogicalVolume.hh"
#include "G4PVPlacement.hh"
#include "G4Region.hh"

#include "G4ParticleTable.hh"
#include "G4Electron.hh"
#include "G4Positron.hh"
#include "G4Gamma.hh"
#include "G4Proton.hh"

#include "G4DataVector.hh"
#include "G4ProductionCuts.hh"
#include "G4MaterialCutsCouple.hh"
#include "G4ProductionCutsTable.hh"
#include "G4EmParameters.hh"


#include "G4HepEmData.hh"
#include "G4HepEmElementData.hh"
#include "G4HepEmMaterialData.hh"
#include "G4HepEmMatCutData.hh"


#include <vector>
#include <cmath>

// builds a fake Geant4 geometry just to be able to produce material-cuts couple
void FakeG4Setup ( G4double prodCutInLength, G4int verbose) {
  //
  // --- Geometry definition: create the word
  G4double wDimX      = 0.6*mm;
  G4double wDimY      = 0.6*mm;
  G4double wDimZ      = 0.6*mm;
  G4Material* wMat    = G4NistManager::Instance()->FindOrBuildMaterial("G4_Galactic");  
  G4Box*           sW = new G4Box ("Box",wDimX, wDimY, wDimZ);
  G4LogicalVolume* lW = new G4LogicalVolume(sW,wMat,"Box",0,0,0);
  G4PVPlacement*   pW = new G4PVPlacement(0,G4ThreeVector(),"Box",lW,0,false,0);
  //
  // --- Build all NIST materials and set a logical volume for each 
  const std::vector<G4String>& namesMat = G4NistManager::Instance()->GetNistMaterialNames();  
  const G4int     numMat = namesMat.size();
  const G4double  halfX  =  0.5/numMat;  // half width of one material-box
  const G4double     x0  = -0.5+halfX;   // start x-position of the first material-box
  for (int im=0; im<numMat; ++im) {
    G4Material*       mat = G4NistManager::Instance()->FindOrBuildMaterial(namesMat[im]);
    G4Box*             ss = new G4Box ("Box", halfX, 0.5, 0.5);
    G4LogicalVolume*   ll = new G4LogicalVolume(ss, mat, "Box", 0, 0, 0);
    new G4PVPlacement(0, G4ThreeVector(x0+im*halfX , 0, 0), "Box", ll, pW, false, 0);    
  }
  //
  // --- Create particles that has secondary production threshold 
  G4Gamma::Gamma();
  G4Electron::Electron();
  G4Positron::Positron();
  G4Proton::Proton();
  G4ParticleTable* partTable = G4ParticleTable::GetParticleTable();
  partTable->SetReadiness();
  //
  // --- Create production - cuts object and set the secondary production threshold
  G4double prodCutValue = prodCutInLength; 
  G4DataVector cuts;
  cuts.push_back(prodCutValue);
  G4ProductionCuts* pcut = new G4ProductionCuts();
  pcut->SetProductionCut(cuts[0], 0); // set cut for gamma
  pcut->SetProductionCut(cuts[0], 1); // set cut for e-
  pcut->SetProductionCut(cuts[0], 2); // set cut for e+
  pcut->SetProductionCut(cuts[0], 3); // set cut for p+
  // 
  // --- Create the material-cuts couple objects: first the for the word, then 
  //     create default region, add this word material-cuts couple  then all others.
  G4MaterialCutsCouple* couple0 = new G4MaterialCutsCouple(wMat, pcut);
  couple0->SetIndex(0);  
  //
  G4Region* reg = new G4Region("DefaultRegionForTheWorld");
  reg->AddRootLogicalVolume(lW);
  reg->UsedInMassGeometry(true);
  reg->SetProductionCuts(pcut);
  reg->RegisterMaterialCouplePair(wMat, couple0);
  for (G4int im=0; im<numMat; ++im) {
    G4Material*              mat = G4NistManager::Instance()->GetMaterial(im);
    G4MaterialCutsCouple* couple = new G4MaterialCutsCouple(mat, pcut);
    couple->SetIndex(im+1);
    reg->RegisterMaterialCouplePair(mat, couple);
  }  
  // --- Update the couple tables
  G4ProductionCutsTable* theCoupleTable = G4ProductionCutsTable::GetProductionCutsTable();
  theCoupleTable->UpdateCoupleTable(pW);  
  //
  if ( verbose>0 ) {
    G4cout << " === FakeG4Setup() completed: \n" 
           << "     - number of G4MaterialCutsCouple objects built = " << numMat          << "     \n" 
           << "     - with secondary production threshold          = " << prodCutInLength << " [mm]\n" 
           << G4endl;
  }
}



bool TestElementData ( const struct G4HepEmData* hepEmData ) {
  bool isPassed = true;
  // loop over all G4MaterialCutsCouple objects
  G4ProductionCutsTable* theCoupleTable = G4ProductionCutsTable::GetProductionCutsTable();
  const G4int numG4MatCuts = theCoupleTable->GetTableSize(); 
  for (G4int imc=0; imc<numG4MatCuts && isPassed; ++imc) {
    const G4MaterialCutsCouple* matCut = theCoupleTable->GetMaterialCutsCouple(imc);
    // care only material-cuts couples used in the current geometry
    if ( !matCut->IsUsed() ) { 
      continue;
    }  
    // obtain the element composition of the material of this couple
    const G4Material*          mat = matCut->GetMaterial();
    const G4ElementVector*  elmVec = mat->GetElementVector();
    size_t            numOfElement = mat->GetNumberOfElements();
    for (size_t ie=0; ie<numOfElement && isPassed; ++ie) {
      const G4Element* g4Element = ((*elmVec)[ie]);
      G4int izet = g4Element->GetZasInt();
      // get the corresonding G4HepEmElemData structure 
      izet       = std::min(izet, (G4int)hepEmData->fTheElementData->fMaxZet);
      const struct G4HepEmElemData& elData = hepEmData->fTheElementData->fElementData[izet];
      if (elData.fZet != (double)izet) {
        isPassed = false;
        std::cerr << "\n*** ERROR:\nElementData: G4HepEm-Geant4 mismatch: fZet = " << elData.fZet << " != " << (double)izet << std::endl; 
        break;             
      }
      double g4Z13 = std::pow((double)izet, 1.0/3.0);
      if (elData.fZet13 != g4Z13 ) {
        isPassed = false;
        std::cerr << "\n*** ERROR:\nElementData: G4HepEm-Geant4 mismatch: fZet13 != " << elData.fZet13 << " != "  << g4Z13 << std::endl; 
        break;             
      }
      double g4Z23 = std::pow((double)izet, 2.0/3.0);
      if (elData.fZet23 != g4Z23 ) {
        isPassed = false;
        std::cerr << "\n*** ERROR:\nElementData: G4HepEm-Geant4 mismatch: fZet23 != " << elData.fZet23 << " != "  << g4Z23 << std::endl; 
        break;             
      }
      double fc = g4Element->GetfCoulomb();
      if (elData.fCoulomb != fc ) {
        isPassed = false;
        std::cerr << "\n*** ERROR:\nElementData: G4HepEm-Geant4 mismatch: fc != " << elData.fCoulomb << " != "  << fc << std::endl; 
        break;             
      }
      double logZ = std::log(izet);
      if (elData.fLogZ != logZ ) {
        isPassed = false;
        std::cerr << "\n*** ERROR:\nElementData: G4HepEm-Geant4 mismatch: fLogZ != " << elData.fLogZ << " != "  << logZ << std::endl; 
        break;             
      }      
    } 
  }
  return isPassed;
}


bool TestMaterialData ( const struct G4HepEmData* hepEmData ) {
  bool isPassed = true;
  // loop over all G4MaterialCutsCouple objects
  G4ProductionCutsTable* theCoupleTable = G4ProductionCutsTable::GetProductionCutsTable();
  const G4int numG4MatCuts = theCoupleTable->GetTableSize(); 
  for (G4int imc=0; imc<numG4MatCuts && isPassed; ++imc) {
    const G4MaterialCutsCouple* matCut = theCoupleTable->GetMaterialCutsCouple(imc);
    // care only material-cuts couples used in the current geometry
    if ( !matCut->IsUsed() ) { 
      continue;
    }  
    // get the G4Material object of this couple 
    const G4Material*            g4Mat = matCut->GetMaterial();
    // the corresponding G4HepEmMatData structure
    const int            hepEmMatIndex = hepEmData->fTheMaterialData->fG4MatIndexToHepEmMatIndex[g4Mat->GetIndex()];
    const struct G4HepEmMatData& heMat = hepEmData->fTheMaterialData->fMaterialData[hepEmMatIndex];
    // compare the stored properties: mass density, eelectron density, #elements
    if ( heMat.fDensity != g4Mat->GetDensity() ) {
      isPassed = false;
      std::cerr << "\n*** ERROR:\nMaterialData: G4HepEm-Geant4 mismatch: fDensity != "         << heMat.fDensity         << " != "  << g4Mat->GetDensity()         << std::endl; 
      continue;
    }
    if ( heMat.fElectronDensity != g4Mat->GetElectronDensity() ) {
      isPassed = false;
      std::cerr << "\n*** ERROR:\nMaterialData: G4HepEm-Geant4 mismatch: fElectronDensity != " << heMat.fElectronDensity << " != "  << g4Mat->GetElectronDensity() << std::endl; 
      continue;
    }   
    if ( heMat.fNumOfElement != (int)g4Mat->GetNumberOfElements() ) {
      isPassed = false;
      std::cerr << "\n*** ERROR:\nMaterialData: G4HepEm-Geant4 mismatch: fNumOfElement != "    << heMat.fNumOfElement    << " != "  << g4Mat->GetNumberOfElements() << std::endl; 
      continue;
    }    
    if ( heMat.fRadiationLength != g4Mat->GetRadlen() ) {
      isPassed = false;
      std::cerr << "\n*** ERROR:\nMaterialData: G4HepEm-Geant4 mismatch: fRadiationLength != " << heMat.fRadiationLength << " != "  << g4Mat->GetRadlen() << std::endl; 
      continue;
    }   
    // obtain the element composition of the g4 material and compare to those stored in hepEm
    const G4ElementVector*  elmVec = g4Mat->GetElementVector();
    const G4double* nbOfAtomPerVol = g4Mat->GetVecNbOfAtomsPerVolume();
    size_t            numOfElement = g4Mat->GetNumberOfElements();
    for (size_t ie=0; ie<numOfElement && isPassed; ++ie) {
      int izet = ((*elmVec)[ie])->GetZasInt();
      if ( heMat.fElementVect[ie] != izet ) {
        isPassed = false;
        std::cerr << "\n*** ERROR:\nMaterialData: G4HepEm-Geant4 mismatch: heMat.fElementVect[ " << ie << "] = " << heMat.fElementVect[ie] << " != " << izet << std::endl; 
        break;             
      }
      if ( heMat.fNumOfAtomsPerVolumeVect[ie] != nbOfAtomPerVol[ie] ) {
        isPassed = false;
        std::cerr << "\n*** ERROR:\nMaterialData: G4HepEm-Geant4 mismatch: heMat.fNumOfAtomsPerVolumeVect[ " << ie << "] = " << heMat.fNumOfAtomsPerVolumeVect[ie] << " != " << nbOfAtomPerVol[ie] << std::endl; 
        break;             
      }
    } 
  }
  return isPassed;
}


bool TestMatCutData ( const struct G4HepEmData* hepEmData ) {
  bool isPassed = true;
  // obtain the global e- tracking cut: e- prod cut = max (tracking cut, rod cut)
  const G4double elTrackingCutE = G4EmParameters::Instance()->LowestElectronEnergy();
  // loop over all G4MaterialCutsCouple objects
  G4ProductionCutsTable* theCoupleTable = G4ProductionCutsTable::GetProductionCutsTable();
  const G4int numG4MatCuts = theCoupleTable->GetTableSize(); 
  for (G4int imc=0; imc<numG4MatCuts && isPassed; ++imc) {
    const G4MaterialCutsCouple* g4MatCut = theCoupleTable->GetMaterialCutsCouple(imc);
    // care only material-cuts couples used in the current geometry
    if ( !g4MatCut->IsUsed() ) { 
      continue;
    }  
    // the G4HepEmMCCData structure that corresponds to this G4MaterialCutsCouple object
    const int                hepEmMCIndex = hepEmData->fTheMatCutData->fG4MCIndexToHepEmMCIndex[g4MatCut->GetIndex()];
    const struct G4HepEmMCCData& heMatCut = hepEmData->fTheMatCutData->fMatCutData[hepEmMCIndex];
    // compare the stored properties: 
    double  g4ElCutE = (*(theCoupleTable->GetEnergyCutsVector(1)))[g4MatCut->GetIndex()];
    double g4GamCutE = (*(theCoupleTable->GetEnergyCutsVector(0)))[g4MatCut->GetIndex()];
    if ( heMatCut.fSecElProdCutE != std::max ( elTrackingCutE, g4ElCutE ) ) {
      isPassed = false;
      std::cerr << "\n*** ERROR:\nMatCutData: G4HepEm-Geant4 mismatch: fSecElProdCutE != "         << heMatCut.fSecElProdCutE        << " != "  << g4ElCutE        << std::endl; 
      continue;
    }
    if ( heMatCut.fSecGamProdCutE != g4GamCutE ) {
      isPassed = false;
      std::cerr << "\n*** ERROR:\nMatCutData: G4HepEm-Geant4 mismatch: fSecGamProdCutE != "        << heMatCut.fSecGamProdCutE       << " != "  << g4GamCutE       << std::endl; 
      continue;
    }
    int g4MatIndxFromHepEm = hepEmData->fTheMaterialData->fMaterialData[heMatCut.fHepEmMatIndex].fG4MatIndex;
    int g4MatIndx = g4MatCut->GetMaterial()->GetIndex();
    if ( g4MatIndxFromHepEm != g4MatIndx ) {
      isPassed = false;
      std::cerr << "\n*** ERROR:\nMatCutData: G4HepEm-Geant4 mismatch: [heMatCut.fHepEmMatIndex].fG4MatIndex != "    << g4MatIndxFromHepEm    << " != "  << g4MatIndx << std::endl; 
      continue;
    } 
  }   
  return isPassed;
}


