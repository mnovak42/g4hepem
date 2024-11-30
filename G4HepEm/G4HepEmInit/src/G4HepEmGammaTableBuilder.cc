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
  // == Generate the energy grids for the 3 kinetic energy window of the total macroscopic cross sections
  // window: 1
  double emin = 100.0*CLHEP::eV;
  double emax = 150.0*CLHEP::keV;
  gmData->fEMin0 = emin;
  gmData->fEMax0 = emax;
  int numEkin0 = gmData->fEGridSize0;
  double* mxsecEGrid0 = new double[numEkin0];
  G4HepEmInitUtils::FillLogarithmicGrid(emin, emax, numEkin0, gmData->fLogEMin0, gmData->fEILDelta0, mxsecEGrid0);
  double* mxComp_w0   = new double[numEkin0]; // mxsec for compton (as PE cannot be interpolated here)
  //
  // window: 2
  emin = 150.0*CLHEP::keV;
  emax =   2.0*CLHEP::electron_mass_c2;
  gmData->fEMax1 = emax;
  int numEkin1 = gmData->fEGridSize1;
  double* mxsecEGrid1 = new double[numEkin1];
  G4HepEmInitUtils::FillLogarithmicGrid(emin, emax, numEkin1, gmData->fLogEMin1, gmData->fEILDelta1, mxsecEGrid1);
  // allocate some auxiliary arrays to prepare data
  double* mxTot_w1 = new double[numEkin1]; // sum of Compton and PE mxsec
  double* mxPE_w1  = new double[numEkin1]; // mxsec PE
  //
  // window: 3
  emin =   2.0*CLHEP::electron_mass_c2;
  emax = 100.0*CLHEP::TeV;
  gmData->fEMax2 = emax;
  int numEkin2 = gmData->fEGridSize2;
  double* mxsecEGrid2 = new double[numEkin2];
  G4HepEmInitUtils::FillLogarithmicGrid(emin, emax, numEkin2, gmData->fLogEMin2, gmData->fEILDelta2, mxsecEGrid2);
  // allocate some auxiliary arrays to prepare all data needed
  double* mxTot_w2  = new double[numEkin2]; // Conversion + compton + PE + Gamma-Nuclear mxsec
  double* sdTot_w2  = new double[numEkin2]; // the second derivative of that
  double* mxConv_w2 = new double[numEkin2]; // Conversion mxsec
  double* sdConv_w2 = new double[numEkin2]; // the second derivative of that
  double* mxComp_w2 = new double[numEkin2]; // Compton  mxsec
  double* sdComp_w2 = new double[numEkin2]; // the second derivative of that
  double* mxPE_w2   = new double[numEkin2]; // PE  mxsec
  double* sdPE_w2   = new double[numEkin2]; // the second derivative of that

  // get the G4HepEm material-cuts and material data: allocate memory for the
  // max-xsec data
  const struct G4HepEmMatCutData*   hepEmMCData  = hepEmData->fTheMatCutData;
  const struct G4HepEmMaterialData* hepEmMatData = hepEmData->fTheMaterialData;
  int numHepEmMCCData   = hepEmMCData->fNumMatCutData;
  int numHepEmMatData   = hepEmMatData->fNumMaterialData;
  gmData->fNumMaterials = numHepEmMatData;
  gmData->fNumData0     = 2*numEkin0;
  gmData->fNumData1     = 3*numEkin1;
  gmData->fDataPerMat   = 2*numEkin0 + 3*numEkin1 + 9*numEkin2;
  gmData->fMacXsecData = new double[numHepEmMatData*gmData->fDataPerMat]{};
  std::vector<bool> isThisMatDone  = std::vector<bool>(numHepEmMatData,false);
  //
  // copute the macroscopic cross sections
  // get the g4 particle-definition
  G4ParticleDefinition* g4PartDef = G4Gamma::Gamma();
  // we will need to obtain the correspondig G4MaterialCutsCouple object pointers
  G4ProductionCutsTable* theCoupleTable = G4ProductionCutsTable::GetProductionCutsTable();
  G4DynamicParticle* dyGamma = new G4DynamicParticle(g4PartDef, G4ThreeVector(0,0,1), 0);
  for (int imc=0; imc<numHepEmMCCData; ++imc) {
    const struct G4HepEmMCCData& mccData = hepEmMCData->fMatCutData[imc];
    int hepEmMatIndx = mccData.fHepEmMatIndex;
    // mac-xsecs has already been computed for this material
    if (isThisMatDone[hepEmMatIndx])
      continue;
    // mac-xsecs needs to be computed for this material
    const G4MaterialCutsCouple* g4MatCut = theCoupleTable->GetMaterialCutsCouple(mccData.fG4MatCutIndex);
    //
    // window: 1 calculate the Compton scattering macroscopic ross section
    for (int ie=0; ie<numEkin0; ++ie) {
      const double theEKin = mxsecEGrid0[ie];
      mxComp_w0[ie] = std::max(0.0, knModel->CrossSection(g4MatCut, g4PartDef, theEKin));
    }
    //
    // window: 2 calculate the Compton and PE macroscopic ross sections
    for (int ie=0; ie<numEkin1; ++ie) {
      const double theEKin = mxsecEGrid1[ie];
      const double comp = std::max(0.0, knModel->CrossSection(g4MatCut, g4PartDef, theEKin));
      const double pe   = std::max(0.0, GetMacXSecPE(hepEmData, hepEmMatIndx, theEKin));
      mxTot_w1[ie] = comp+pe;
      mxPE_w1[ie]  = pe;
    }
    //
    // window: 3 calculate the Conversion, Compton, PE and Gamma-Nuclear macroscopic cross sections
    for (int ie=0; ie<numEkin2; ++ie) {
      const double theEKin = mxsecEGrid2[ie];
      const double conv = std::max(0.0, ppModel->CrossSection(g4MatCut, g4PartDef, theEKin));
      const double comp = std::max(0.0, knModel->CrossSection(g4MatCut, g4PartDef, theEKin));
      const double pe   = std::max(0.0, GetMacXSecPE(hepEmData, hepEmMatIndx, theEKin));
      dyGamma->SetKineticEnergy(theEKin);
      const double gnuc = std::max(0.0, hadGNucXSDataStore->ComputeCrossSection(dyGamma, g4MatCut->GetMaterial()));
      mxTot_w2[ie]  = conv+comp+pe+gnuc;
      mxConv_w2[ie] = conv;
      mxComp_w2[ie] = comp;
      mxPE_w2[ie]   = pe;
      sdTot_w2[ie]  = 0.0;
      sdConv_w2[ie] = 0.0;
      sdComp_w2[ie] = 0.0;
      sdPE_w2[ie]   = 0.0;
    }
    // prepare for spline by computing the second derivatives
    G4HepEmInitUtils::PrepareSpline(numEkin2, mxsecEGrid2,  mxTot_w2, sdTot_w2);
    G4HepEmInitUtils::PrepareSpline(numEkin2, mxsecEGrid2, mxConv_w2, sdConv_w2);
    G4HepEmInitUtils::PrepareSpline(numEkin2, mxsecEGrid2, mxComp_w2, sdComp_w2);
    G4HepEmInitUtils::PrepareSpline(numEkin2, mxsecEGrid2,   mxPE_w2, sdPE_w2);
    // fill in the data for this material into the final location
    int indxCont = hepEmMatIndx*gmData->fDataPerMat; // data for this material starts here
    for (int ie=0; ie<numEkin0; ++ie) {
      const double theEKin = mxsecEGrid0[ie];
      gmData->fMacXsecData[indxCont++] = theEKin;
      gmData->fMacXsecData[indxCont++] = mxComp_w0[ie];  // mac. x-sec Compton
    }
    for (int ie=0; ie<numEkin1; ++ie) {
      const double theEKin = mxsecEGrid1[ie];
      gmData->fMacXsecData[indxCont++] = theEKin;
      gmData->fMacXsecData[indxCont++] = mxTot_w1[ie];  // mac. x-sec total: Compton + PE
      gmData->fMacXsecData[indxCont++] = mxPE_w1[ie];   // mac. x-sec PE
    }
    for (int ie=0; ie<numEkin2; ++ie) {
      const double theEKin = mxsecEGrid2[ie];
      gmData->fMacXsecData[indxCont++] = theEKin;
      gmData->fMacXsecData[indxCont++] = mxTot_w2[ie]; // mac. x-sec total: Conv. + Compt. + PE + GN
      gmData->fMacXsecData[indxCont++] = sdTot_w2[ie]; // second derivative of that
      gmData->fMacXsecData[indxCont++] = mxConv_w2[ie]; // mac. x-sec Conversion
      gmData->fMacXsecData[indxCont++] = sdConv_w2[ie]; // second derivative of that
      gmData->fMacXsecData[indxCont++] = mxComp_w2[ie]; // mac. x-sec Compton
      gmData->fMacXsecData[indxCont++] = sdComp_w2[ie]; // second derivative of that
      gmData->fMacXsecData[indxCont++] = mxPE_w2[ie];   // mac. x-sec PE
      gmData->fMacXsecData[indxCont++] = sdPE_w2[ie];   // second derivative of that
    }
    //
    // set this material index to be done
    isThisMatDone[hepEmMatIndx] = true;
  }
  delete dyGamma;

  // free all dynamically allocated auxiliary memory
  delete[] mxsecEGrid0;
  delete[] mxsecEGrid1;
  delete[] mxsecEGrid2;
  delete[] mxComp_w0;
  delete[] mxTot_w1;
  delete[] mxPE_w1;
  delete[] mxTot_w2;
  delete[] sdTot_w2;
  delete[] mxConv_w2;
  delete[] sdConv_w2;
  delete[] mxComp_w2;
  delete[] sdComp_w2;
  delete[] mxPE_w2;
  delete[] sdPE_w2;
}


// element selectro only for Conversion (compton model is too dummy to care)
void BuildElementSelectorTables(G4PairProductionRelModel* ppModel, struct G4HepEmData* hepEmData) {
  // get the pointer to the already allocated G4HepEmGammaData from the HepEmData
  struct G4HepEmGammaData* gmData = hepEmData->fTheGammaData;
  //
  // == Generate the enegry grid for Conversion-element selectors
  const double emin      = gmData->fEMax1;
  const double emax      = gmData->fEMax2;
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



double GetMacXSecPE(const struct G4HepEmData* hepEmData, const int imat, const double ekin) {
  const G4HepEmMatData* matData = &hepEmData->fTheMaterialData->fMaterialData[imat];
  int interval = 0;
  if (ekin >= matData->fSandiaEnergies[0]) {
    // Optimization: linear search starting with intervals for higher energies.
    for (int i = matData->fNumOfSandiaIntervals - 1; i >= 0; i--) {
      if (ekin >= matData->fSandiaEnergies[i]) {
        interval = i;
        break;
      }
    }
  }
  const double* sandiaCof = &matData->fSandiaCoefficients[4 * interval];
  const double inv = 1 / ekin;
  return inv * (sandiaCof[0] + inv * (sandiaCof[1] + inv * (sandiaCof[2] + inv * sandiaCof[3])));
}
