#include "G4HepEmGammaTableBuilder.hh"

#include "G4HepEmData.hh"
#include "G4HepEmMatCutData.hh"
#include "G4HepEmMaterialData.hh"
#include "G4HepEmElementData.hh"
#include "G4HepEmGammaData.hh"

#include "G4HepEmParameters.hh"

#include "G4HepEmInitUtils.hh"


// g4 includes
#include "G4PairProductionRelModel.hh"
#include "G4KleinNishinaCompton.hh"

#include "G4CrossSectionDataStore.hh"

#include "G4ParticleDefinition.hh"
#include "G4Gamma.hh"
#include "G4ProductionCutsTable.hh"
#include "G4Material.hh"
#include "G4Element.hh"
#include "G4NistManager.hh"
#include "G4EmParameters.hh"

#include <cmath>

void BuildLambdaTables(G4PairProductionRelModel* ppModel, G4KleinNishinaCompton* knModel,
                       G4CrossSectionDataStore* hadGNucXSDataStore, struct G4HepEmData* hepEmData) {
  // get the pointer to the already allocated G4HepEmGammaData from the HepEmData
  struct G4HepEmGammaData* gmData = hepEmData->fTheGammaData;
  //
  // == Generate the enegry grid for Conversion
  double emin = 2.0*CLHEP::electron_mass_c2;
  double emax = 100.0*CLHEP::TeV;
  int numConvEkin = gmData->fConvEnergyGridSize;
  delete [] gmData->fConvEnergyGrid;
  gmData->fConvEnergyGrid = new double[numConvEkin]{};
  G4HepEmInitUtils::FillLogarithmicGrid(emin, emax, numConvEkin, gmData->fConvLogMinEkin, gmData->fConvEILDelta, gmData->fConvEnergyGrid);

  // == Generate the enegry grid for Compton
  emin = 100.0* CLHEP::eV;
  emax = 100.0*CLHEP::TeV;
  int numCompEkin = gmData->fCompEnergyGridSize;
  delete [] gmData->fCompEnergyGrid;
  gmData->fCompEnergyGrid = new double[numCompEkin]{};
  G4HepEmInitUtils::FillLogarithmicGrid(emin, emax, numCompEkin, gmData->fCompLogMinEkin, gmData->fCompEILDelta, gmData->fCompEnergyGrid);

  // == Generate the enegry grid for Gamma-nuclear
  emin = 2.0*CLHEP::electron_mass_c2;
  emax = 100.0*CLHEP::TeV;
  int numGNucEkin = gmData->fGNucEnergyGridSize;
  delete [] gmData->fGNucEnergyGrid;
  gmData->fGNucEnergyGrid = new double[numGNucEkin]{};
  G4HepEmInitUtils::FillLogarithmicGrid(emin, emax, numGNucEkin, gmData->fGNucLogMinEkin, gmData->fGNucEILDelta, gmData->fGNucEnergyGrid);

  //
  // == Compute the macroscopic cross sections: for Conversion, Compton and
  //    gamma-nuclear interactions over all materials
  //
  // get the G4HepEm material-cuts and material data: allocate memory for the
  // max-xsec data
  const struct G4HepEmMatCutData*   hepEmMCData  = hepEmData->fTheMatCutData;
  const struct G4HepEmMaterialData* hepEmMatData = hepEmData->fTheMaterialData;
  int numHepEmMCCData   = hepEmMCData->fNumMatCutData;
  int numHepEmMatData   = hepEmMatData->fNumMaterialData;
  gmData->fNumMaterials = numHepEmMatData;
  gmData->fConvCompGNucMacXsecData = new double[numHepEmMatData*2*(numConvEkin + numCompEkin + numGNucEkin)]{};
  std::vector<bool> isThisMatDone  = std::vector<bool>(numHepEmMatData,false);
  //
  // copute the macroscopic cross sections
  // get the g4 particle-definition
  G4ParticleDefinition* g4PartDef = G4Gamma::Gamma();
  // we will need to obtain the correspondig G4MaterialCutsCouple object pointers
  G4ProductionCutsTable* theCoupleTable = G4ProductionCutsTable::GetProductionCutsTable();
  // a temporary container for the mxsec data and for their second deriv
  double* macXSec   = new double[std::max(std::max(numConvEkin,numCompEkin), numGNucEkin)]{};
  double* secDerivs = new double[std::max(std::max(numConvEkin,numCompEkin), numGNucEkin)]{};
  for (int imc=0; imc<numHepEmMCCData; ++imc) {
    const struct G4HepEmMCCData& mccData = hepEmMCData->fMatCutData[imc];
    int hepEmMatIndx = mccData.fHepEmMatIndex;
    // mac-xsecs has already been computed for this material
    if (isThisMatDone[hepEmMatIndx])
      continue;
    // mac-xsecs needs to be computed for this material
    const G4MaterialCutsCouple* g4MatCut = theCoupleTable->GetMaterialCutsCouple(mccData.fG4MatCutIndex);
    // == Conversion
    for (int ie=0; ie<numConvEkin; ++ie) {
      const double theEKin = gmData->fConvEnergyGrid[ie];
      macXSec[ie] = std::max(0.0, ppModel->CrossSection(g4MatCut, g4PartDef, theEKin));
    }
    // prepare for spline by computing the second derivatives
    G4HepEmInitUtils::PrepareSpline(numConvEkin, gmData->fConvEnergyGrid, macXSec, secDerivs);
    // fill in into the continuous array: index where data for this material starts from
    int mxStartIndx = hepEmMatIndx*2*(numConvEkin + numCompEkin + numGNucEkin);
    int indxCont    = mxStartIndx;
    for (int i=0; i<numConvEkin; ++i) {
      gmData->fConvCompGNucMacXsecData[indxCont++] = macXSec[i];
      gmData->fConvCompGNucMacXsecData[indxCont++] = secDerivs[i];
    }
    // == Compton
//    std::cout << " ===== Material = " << g4MatCut->GetMaterial()->GetName() << std::endl;
    for (int ie=0; ie<numCompEkin; ++ie) {
      const double theEKin = gmData->fCompEnergyGrid[ie];
      macXSec[ie] = std::max(0.0, knModel->CrossSection(g4MatCut, g4PartDef, theEKin));
//      std::cout << " E = " << theEKin << " [MeV] Sigam-Compton(E) = " << macXSec[ie] << std::endl;
    }
    // prepare for spline by computing the second derivatives
    G4HepEmInitUtils::PrepareSpline(numCompEkin, gmData->fCompEnergyGrid, macXSec, secDerivs);
    // fill in into the continuous array: the continuous index is used further here
    for (int i=0; i<numCompEkin; ++i) {
      gmData->fConvCompGNucMacXsecData[indxCont++] = macXSec[i];
      gmData->fConvCompGNucMacXsecData[indxCont++] = secDerivs[i];
    }
    // == Gamma-nuclear
    // std::cout << " ===== Material = " << g4MatCut->GetMaterial()->GetName() << std::endl;
    G4DynamicParticle* dyGamma = new G4DynamicParticle(g4PartDef, G4ThreeVector(0,0,1), 0);
    for (int ie=0; ie<numGNucEkin; ++ie) {
      const double theEKin = gmData->fGNucEnergyGrid[ie];
      dyGamma->SetKineticEnergy(theEKin);
      macXSec[ie] = std::max(0.0, hadGNucXSDataStore->ComputeCrossSection(dyGamma, g4MatCut->GetMaterial()));
      // std::cout << " E = " << theEKin << " [MeV] Sigam-GNuc(E) = " << macXSec[ie] << std::endl;
    }
    delete dyGamma;
    // prepare for spline by computing the second derivatives
    G4HepEmInitUtils::PrepareSpline(numGNucEkin, gmData->fGNucEnergyGrid, macXSec, secDerivs);
    // fill in into the continuous array: the continuous index is used further here
    for (int i=0; i<numGNucEkin; ++i) {
      gmData->fConvCompGNucMacXsecData[indxCont++] = macXSec[i];
      gmData->fConvCompGNucMacXsecData[indxCont++] = secDerivs[i];
    }

    //
    // set this material index to be done
    isThisMatDone[hepEmMatIndx] = true;
  }
  //
  // free all dynamically allocated auxilary memory
  delete[] macXSec;
  delete[] secDerivs;
}


// element selectro only for Conversion (compton model is too dummy to care)
void BuildElementSelectorTables(G4PairProductionRelModel* ppModel, struct G4HepEmData* hepEmData) {
  // get the pointer to the already allocated G4HepEmGammaData from the HepEmData
  struct G4HepEmGammaData* gmData = hepEmData->fTheGammaData;
  //
  // == Generate the enegry grid for Conversion-element selectors
  const double emin      = gmData->fConvEnergyGrid[0];
  const double emax      = gmData->fConvEnergyGrid[gmData->fConvEnergyGridSize-1];
  const double invlog106 = 1.0/(6.0*std::log(10.0));
  int numConvEkin = (int)(G4EmParameters::Instance()->NumberOfBinsPerDecade()*std::log(emax/emin)*invlog106);
  gmData->fElemSelectorConvEgridSize = numConvEkin;
  gmData->fElemSelectorConvEgrid = new double[numConvEkin]{};
  G4HepEmInitUtils::FillLogarithmicGrid(emin, emax, numConvEkin,
                                        gmData->fElemSelectorConvLogMinEkin, gmData->fElemSelectorConvEILDelta, gmData->fElemSelectorConvEgrid);

  //
  // fill in with the element selectors (only for materials with #elemnt > 1)
  // get the G4HepEm material-cuts and material data: allocate memory for the
  // max-xsec data
//  const struct G4HepEmMatCutData*   hepEmMCData  = hepEmData->fTheMatCutData;
  const struct G4HepEmMaterialData* hepEmMatData = hepEmData->fTheMaterialData;
//  int numHepEmMCCData   = hepEmMCData->fNumMatCutData;
  int numHepEmMatData   = hepEmMatData->fNumMaterialData;
  gmData->fElemSelectorConvStartIndexPerMat = new int[numHepEmMatData]{};
  // count size of containers: #elements(in mat. with #eleme>1) * (#elem-1)*numConvEkin
  int num = 0;
  for (int im=0; im<numHepEmMatData; ++im) {
    const struct G4HepEmMatData& matData = hepEmMatData->fMaterialData[im];
    int numElem = matData.fNumOfElement;
    if (numElem>1) {
      // should be numElem-1 but for each material 1 extra is #elements that is the first elem
      num += numElem;
    } else {
      gmData->fElemSelectorConvStartIndexPerMat[im] = -1;
    }
  }
  // allocate memory:
  int size = num*numConvEkin;
  gmData->fElemSelectorConvNumData = size;
  if (size == 0) {
    return;
  }
  gmData->fElemSelectorConvData = new double[size]{};
  G4VEmModel* emModel = ppModel;
  int indxCont = 0;
  for (int im=0; im<numHepEmMatData; ++im) {
    const struct G4HepEmMatData& matData = hepEmMatData->fMaterialData[im];
    int numElem = matData.fNumOfElement;
    if (numElem < 2) {
//      gmData->fElemSelectorConvStartIndexPerMat[im] = -1;
      continue;
    }
    gmData->fElemSelectorConvStartIndexPerMat[im] = indxCont;
    gmData->fElemSelectorConvData[indxCont++]     = numElem;
    // build element selector for this material starting the data from indxCont:
    // loop over the kinetic energy grid
    for (int ie=0; ie<numConvEkin; ++ie) {
      double      ekin = gmData->fElemSelectorConvEgrid[ie];
      double       sum = 0.0;
      int          ist = indxCont;
      for (int iz=0; iz<numElem; ++iz) {
        // compute atomic cross section x number of atoms per volume
        int      izet = matData.fElementVect[iz];
        double natoms = matData.fNumOfAtomsPerVolumeVect[iz];
        const G4Element* g4Elem  = G4NistManager::Instance()->FindOrBuildElement(izet);
        double   xsec = std::max(0.0, emModel->ComputeCrossSectionPerAtom(G4Gamma::Gamma(), g4Elem, ekin));
        sum += natoms*xsec;
        if (iz<numElem-1) {
          gmData->fElemSelectorConvData[indxCont++] = sum;
        }
      }
      // normalise
      if (sum>0.0) {
        sum = 1.0/sum;
        for (int i=0; i<numElem-1; ++i) {
            gmData->fElemSelectorConvData[ist+i] *= sum;
        }
      }
    }
  }
}
