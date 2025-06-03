
#include "G4HepEmElectronTableBuilder.hh"

#include "G4HepEmData.hh"
#include "G4HepEmMatCutData.hh"
#include "G4HepEmMaterialData.hh"
#include "G4HepEmElementData.hh"
#include "G4HepEmElectronData.hh"

#include "G4HepEmParameters.hh"

#include "G4HepEmInitUtils.hh"

#include "G4HepEmSBBremTableBuilder.hh"
#include "G4HepEmSBTableData.hh"


// g4 includes
#include "G4Version.hh"
#include "G4MollerBhabhaModel.hh"
#include "G4SeltzerBergerModel.hh"
#include "G4eBremsstrahlungRelModel.hh"

#include "G4CrossSectionDataStore.hh"

#include "G4ParticleDefinition.hh"
#include "G4Electron.hh"
#include "G4Positron.hh"
#include "G4ProductionCutsTable.hh"
#include "G4Material.hh"
#include "G4Element.hh"
#include "G4NistManager.hh"
#include "G4EmParameters.hh"



#include <cmath>


void BuildELossTables(G4MollerBhabhaModel* mbModel, G4SeltzerBergerModel* sbModel,
                      G4eBremsstrahlungRelModel* rbModel, struct G4HepEmData* hepEmData,
                      struct G4HepEmParameters* hepEmParams, bool iselectron) {
  // get the pointer to the already allocated G4HepEmElectronData from the HepEmData
  struct G4HepEmElectronData* elData = iselectron
                                       ? hepEmData->fTheElectronData
                                       : hepEmData->fThePositronData;
  //
  // generate the enegry grid (common for all mat-cuts)
  const int numELoss = hepEmParams->fNumLossTableBins+1;
  elData->fELossEnergyGridSize = numELoss;
  delete [] elData->fELossEnergyGrid;
  elData->fELossEnergyGrid = new double[numELoss]{};
  G4HepEmInitUtils::FillLogarithmicGrid(hepEmParams->fMinLossTableEnergy, hepEmParams->fMaxLossTableEnergy, numELoss,
                                        elData->fELossLogMinEkin, elData->fELossEILDelta, elData->fELossEnergyGrid);

  // get the g4 particle-definition
  G4ParticleDefinition* g4PartDef = G4Positron::Positron();
  if  (iselectron) {
    g4PartDef = G4Electron::Electron();
  }
  // get the HepEm Material-cut couple data
  const struct G4HepEmMatCutData*    hepEmMCData = hepEmData->fTheMatCutData;
  // we will need to obtain the correspondig G4MaterialCutsCouple object pointers
  G4ProductionCutsTable* theCoupleTable = G4ProductionCutsTable::GetProductionCutsTable();
  // loop over the HepEm material-cuts couples:
  //  - get the corresponding G4Material pointer
  //  - loop over the eloss-energy grid and compute the sum of the electronic and
  //    radiative (restricted) stopping powers for each material-cuts couple
  //  - apply smoothing in case of the radiative (i.e. brem.) part between the
  //    two models connected at bermModelLimitEnergy = 1 GeV = 1000 MeV
  //  - store the dE/dx in an intermediate array, compute the range array and
  //    their second derivative for a spline interpolation
  //  - fill these 4 later data into the elData structure for this mat-cut
  //
  double* theDEDXArray       = new double[numELoss]{};
  double* theRangeArray      = new double[numELoss]{};
  double* theDEDXSDArray     = new double[numELoss]{};  // second derivatives for dedx
  double* theRangeSDArray    = new double[numELoss]{};  // second derivatives for range
  double* theInvRangeSDArray = new double[numELoss]{};  // second derivatives for inverse range
  //
  int numHepEmMCCData   = hepEmMCData->fNumMatCutData;
  elData->fNumMatCuts   = numHepEmMCCData;
  elData->fNumMaterials = hepEmData->fTheMaterialData->fNumMaterialData;
  //
  // allocate array to store the [range, sec-deriv, dedx, sec-deriv, in-range-sec-deriv]
  // (numELoss values each) for all matrial-cuts couple
//
//  std::out << " ==== Allocating " << 5.0*numELoss*numHepEmMCCData*sizeof(double)/1024/1024
//            << " [MB] memory in G4HepEmELossTableBuilder::BuildELossTables \n"
//            << " for Range, dE/dx and inv-Range for the " << numHepEmMCCData
//            << "\n material-cuts couples used in the geometry. ( 5x " << numELoss
//            << " `double` value for each)." << std::endl;
  elData->fELossData = new double[5*numELoss*numHepEmMCCData]{};
  //
  // starts the computations for all mat-cut couples
  for (int imc=0; imc<numHepEmMCCData; ++imc) {
    const struct G4HepEmMCCData& mccData = hepEmMCData->fMatCutData[imc];
    const G4MaterialCutsCouple* g4MatCut = theCoupleTable->GetMaterialCutsCouple(mccData.fG4MatCutIndex);
    const double     elCutE = mccData.fSecElProdCutE;  // already includes e- tracking cut
    const double    gamCutE = mccData.fSecGamProdCutE;
    // loop over the eloss-energy grid
    for (int ie=0; ie<numELoss; ++ie) {
      double ekin = elData->fELossEnergyGrid[ie];
      // compute the electronic dE/dx
      double dedxIoni = std::max(0.0, mbModel->ComputeDEDX(g4MatCut, g4PartDef, ekin, elCutE));
      // compute the radiative dE/dx: smoothing of the high energy model value
      double dlta = 0.0;
      if ( ekin > hepEmParams->fElectronBremModelLim ) {
        double dedx1  = sbModel->ComputeDEDX(g4MatCut, g4PartDef, hepEmParams->fElectronBremModelLim, gamCutE);
        double dedx2  = rbModel->ComputeDEDX(g4MatCut, g4PartDef, hepEmParams->fElectronBremModelLim, gamCutE);
        dlta = dedx2 > 0.0 ? (dedx1/dedx2-1.0)*hepEmParams->fElectronBremModelLim : 0.0;
      }
      //sbModel->SetupForMaterial(g4PartDef, g4MatCut->GetMaterial(), ekin);
      double dedxBrem = ekin > hepEmParams->fElectronBremModelLim
                        ? rbModel->ComputeDEDX(g4MatCut, g4PartDef, ekin, gamCutE)
                        : sbModel->ComputeDEDX(g4MatCut, g4PartDef, ekin, gamCutE);
      dedxBrem *= (1.0+dlta/ekin);
      // sum up the electronic and radiative parts
      theDEDXArray[ie] = dedxIoni + std::max(0.0, dedxBrem);
    }
    // set up a spline on the DEDX array for interpolation
    G4HepEmInitUtils::PrepareSpline(numELoss, elData->fELossEnergyGrid, theDEDXArray, theDEDXSDArray);
    // integrate the restricted dedx to get the corresponding restricted range:
    // - first set the very first range value i.e. approximate the integral of
    //   the dE/dx on [0, E_0] by assuming that the dE/dx is proportional to $\beta$
    theRangeArray[0] = 2.0*elData->fELossEnergyGrid[0]/theDEDXArray[0];
    // - integrate the dE/dx by using a 16 point GL integral on [0,1] at each bin
    int ngl     = 16;
    double* glX = new double[ngl]{};
    double* glW = new double[ngl]{};
    G4HepEmInitUtils::GLIntegral(ngl, glX, glW);
    for (int i=0; i<numELoss-1; ++i) {
      // for each E_i, E_i+1 interval apply the GL by substitution
      const double emin  = elData->fELossEnergyGrid[i];
      const double emax  = elData->fELossEnergyGrid[i+1];
      const double del   = (emax-emin);
      double res   = 0.0;
      for (int j=0; j<ngl; ++j) {
        const double xi = del*glX[j]+emin;
        double dedx = G4HepEmInitUtils::GetSpline(elData->fELossEnergyGrid, theDEDXArray, theDEDXSDArray, xi, i); // i is the low Energy bin index
        if (dedx>0.0) {
          res += glW[j]/dedx;
        }
      }
      res *= del;
      theRangeArray[i+1] = res+theRangeArray[i];
    }
    // clean auxilary arrays
    delete[] glX;
    delete[] glW;


    //
    // prepare final form of the Range, dE/dx, inverse range and their second
    // derivatives for this macc and fill in to the G4HepEmElemData elData struct
    // - set up a spline on the range array for spline interpolation
    G4HepEmInitUtils::PrepareSpline(numELoss, elData->fELossEnergyGrid, theRangeArray, theRangeSDArray);
    G4HepEmInitUtils::PrepareSpline(numELoss, theRangeArray, elData->fELossEnergyGrid, theInvRangeSDArray);
    // start index of the [range,sd, dedx, sd, inv-range sd] values for this
    // material-cuts couple in the elData->fELossData array
    int indxStart = 5*numELoss*imc;
    for (int i=0; i<numELoss; ++i) {
      // first the range then the dedx and finally the inverse range
      elData->fELossData[indxStart+2*i]              = theRangeArray[i];
      elData->fELossData[indxStart+2*i+1]            = theRangeSDArray[i];
      elData->fELossData[indxStart+2*(numELoss+i)]   = theDEDXArray[i];
      elData->fELossData[indxStart+2*(numELoss+i)+1] = theDEDXSDArray[i];
      elData->fELossData[indxStart+4*(numELoss)+i]   = theInvRangeSDArray[i];
    }
  }
  // free auxilary arrays
  delete[] theDEDXArray;
  delete[] theRangeArray;
  delete[] theDEDXSDArray;
  delete[] theRangeSDArray;
  delete[] theInvRangeSDArray;
}


// G4ElectroNuclearCrossSection::GetElementCrossSection
// G4PhotoNuclearCrossSection::GetElementCrossSection


void BuildLambdaTables(G4MollerBhabhaModel* mbModel, G4SeltzerBergerModel* sbModel,
                       G4eBremsstrahlungRelModel* rbModel, struct G4HepEmData* hepEmData,
                       struct G4HepEmParameters* hepEmParams, bool iselectron) {
  // get the pointer to the already allocated G4HepEmElectronData from the HepEmData
  struct G4HepEmElectronData* elData = iselectron
                                       ? hepEmData->fTheElectronData
                                       : hepEmData->fThePositronData;
  //
  // get the g4 particle-definition
  G4ParticleDefinition* g4PartDef = G4Positron::Positron();
  if  (iselectron) {
    g4PartDef = G4Electron::Electron();
  }
  // we will need to obtain the correspondig G4MaterialCutsCouple object pointers
  G4ProductionCutsTable* theCoupleTable = G4ProductionCutsTable::GetProductionCutsTable();
  //
  // Loop over the HepEm material-cuts couple:s
  // - get the secondary e- and gamma production thresholds
  // - determine the minimum primary energy at which the ioni/brem can happen
  // - generate the energy grid (for ioni/brem) and compute the resctricted
  //   macroscopic cross sections
  // - keep track of the maximum value and its energy position
  // - store the data in th efollowing format for a given material-cuts couple
  //   === Ioni then Brem:
  //   === for each first 4 values: energyOfMaxValue, maxValue, logE_0, invLodEDelta
  //   === then the values: energy, value, second-derivative order i.e.
  //       [...,E_i,Sigma_i,SD_i, E_{i+1}, Sigma_{i}, SD_{i},...]
  //   === then the same for Brem, etc.
  // - record the start index of the data for each material-cuts couple in a separate array
  // - record the start index of the brem data relative to the ioni (i.e. number of ioni entires)
  //   in a separate array
  //
  // With these, we can find easily where the ioni and brem data starts in the flatten array
  //
  // ON GPU, first the ioni energy grid, ioni sigmas, their sec-derive, then for brem

  // prepare some space (for sure enough) to store an energy grid and mac-xsec
  double*  energyGrid = new double[hepEmParams->fNumLossTableBins+2]{};
  double*  macXSec    = new double[hepEmParams->fNumLossTableBins+2]{};
  double*  secDerivs  = new double[hepEmParams->fNumLossTableBins+2]{};
  // also prepare a maximal size array: 2 x 3 x (N+2) for each mat-cuts where
  // the 2 is for ioni + brem, the 3 is for E,Sig,SD and N+2 is the max number
  // of possible such entires and the + 5 is the #data, max value and energy grid related
  // 4 first entires.
  //
  // get the HepEm Material-cut couple data
  const struct G4HepEmMatCutData*  hepEmMCData = hepEmData->fTheMatCutData;
  int numHepEmMCCData = hepEmMCData->fNumMatCutData;
  double*    xsecData = new double[2*3*(hepEmParams->fNumLossTableBins+2+5)*numHepEmMCCData]{};
  //
  // allocate the arrays to store start indices per matrial-cuts couples
  elData->fResMacXSecStartIndexPerMatCut = new int[numHepEmMCCData]{};
  // a continuous index
  int indxCont = 0;
  for (int imc=0; imc<numHepEmMCCData; ++imc) {
    // ====== Common data
    const struct G4HepEmMCCData& mccData = hepEmMCData->fMatCutData[imc];
    const G4MaterialCutsCouple* g4MatCut = theCoupleTable->GetMaterialCutsCouple(mccData.fG4MatCutIndex);
    const double     elCutE = mccData.fSecElProdCutE;  // already includes e- tracking cut
    const double    gamCutE = mccData.fSecGamProdCutE;

    // Energy grid in/out parameters
    const double       emax = hepEmParams->fMaxLossTableEnergy;
    const int    numDefEkin = hepEmParams->fNumLossTableBins+1;
    const double      scale = std::log(hepEmParams->fMaxLossTableEnergy/hepEmParams->fMinLossTableEnergy);

    double logEmin  = -1.0;
    double invLEDel = -1.0;

    //
    // ===== Ionisation
    //
    // Fill the energy grid for Ioni
    // find out the lowest energy of the ioni Ekin grid and the number of entries
    const double   eminIoni = iselectron ? 2*elCutE : elCutE;
    const double  scaleIoni = std::log(emax/eminIoni);
    const int      numEIoni = std::max(4, (int)std::lrint(numDefEkin*scaleIoni/scale)+1);
    G4HepEmInitUtils::FillLogarithmicGrid(eminIoni, emax, numEIoni, logEmin, invLEDel, energyGrid);

    // compute macroscopic cross section for Ioni.
    // track macroscopic cross section max and its energy
    double       macXSecMax = -1.0;
    double   macXSecMaxEner = -1.0;

    for (int ie=0; ie<numEIoni; ++ie) {
      const double theEKin  = energyGrid[ie];
      const double theXSec  = std::max(0.0, mbModel->CrossSection(g4MatCut, g4PartDef, theEKin, elCutE, theEKin));
      if (theXSec>macXSecMax) {
        macXSecMax     = theXSec;
        macXSecMaxEner = theEKin;
      }
      macXSec[ie] = theXSec;
    }
    // prepare for sline by computing the second derivatives
    G4HepEmInitUtils::PrepareSpline(numEIoni, energyGrid, macXSec, secDerivs);
    // fill in into the continuous array:
    // - set the current index as starting point for this material-cust couple
    elData->fResMacXSecStartIndexPerMatCut[imc] = indxCont;
    // - fill in the number of ioni data, energyOfMaxVal, maxVal, logEmin and 1/log-delta values first
    xsecData[indxCont++] = numEIoni;
    xsecData[indxCont++] = macXSecMaxEner;
    xsecData[indxCont++] = macXSecMax;
    xsecData[indxCont++] = logEmin;
    xsecData[indxCont++] = invLEDel;
    for (int ie=0; ie<numEIoni; ++ie) {
      xsecData[indxCont++] = energyGrid[ie];
      xsecData[indxCont++] = macXSec[ie];
      xsecData[indxCont++] = secDerivs[ie];
    }

    //
    // ===== Bremsstrahlung
    //
    // Fill the energy grid for Brem:
    const double   eminBrem = gamCutE;
    const double  scaleBrem = std::log(emax/eminBrem);
    const int      numEBrem = std::max(4, (int)std::lrint(numDefEkin*scaleBrem/scale)+1);
    G4HepEmInitUtils::FillLogarithmicGrid(eminBrem, emax, numEBrem, logEmin, invLEDel, energyGrid);

    // compute macroscopic cross section for Brem: smooth the xsection values between the 2 models
    // keep track of macroscopic cross section max and its energy
    macXSecMax              = -1.0;
    macXSecMaxEner          = -1.0;

    for (int ie=0; ie<numEBrem; ++ie) {
      const double theEKin  = energyGrid[ie];
      double dlta = 0.0;
      if ( theEKin > hepEmParams->fElectronBremModelLim ) {
        double xsec1  = std::max(0.0, sbModel->CrossSection(g4MatCut, g4PartDef, hepEmParams->fElectronBremModelLim, gamCutE, hepEmParams->fElectronBremModelLim));
        double xsec2  = std::max(0.0, rbModel->CrossSection(g4MatCut, g4PartDef, hepEmParams->fElectronBremModelLim, gamCutE, hepEmParams->fElectronBremModelLim));
        dlta = xsec2 > 0.0 ? (xsec1/xsec2-1.0)*hepEmParams->fElectronBremModelLim : 0.0;
      }
      double theXSec = theEKin > hepEmParams->fElectronBremModelLim
                        ? std::max(0.0, rbModel->CrossSection(g4MatCut, g4PartDef, theEKin, gamCutE, theEKin))
                        : std::max(0.0, sbModel->CrossSection(g4MatCut, g4PartDef, theEKin, gamCutE, theEKin));
      theXSec *= (1.0+dlta/theEKin);

      if (theXSec>macXSecMax) {
        macXSecMax     = theXSec;
        macXSecMaxEner = theEKin;
      }
      macXSec[ie] = theXSec;
    }
    // prepare for sline by computing the second derivatives
    G4HepEmInitUtils::PrepareSpline(numEBrem, energyGrid, macXSec, secDerivs);
    // - fill in the number of Brem data, energyOfMaxVal, maxVal, logEmin and 1/log-delta values first
    xsecData[indxCont++] = numEBrem;
    xsecData[indxCont++] = macXSecMaxEner;
    xsecData[indxCont++] = macXSecMax;
    xsecData[indxCont++] = logEmin;
    xsecData[indxCont++] = invLEDel;
    for (int ie=0; ie<numEBrem; ++ie) {
      xsecData[indxCont++] = energyGrid[ie];
      xsecData[indxCont++] = macXSec[ie];
      xsecData[indxCont++] = secDerivs[ie];
    }
  }
  // allocate data, in the fTheElectronData member of the top level data structure,
  // for all the macroscopic-scross section data for all mat-cuts and store them
  if (elData->fResMacXSecData) {
    delete[] elData->fResMacXSecData;
  }
//  std::cerr << " ==== Allocating " << indxCont*sizeof(double)/1024./1024.
//            << " [MB] memory in G4HepEmELossTableBuilder::BuildLambdaTables \n"
//            << " for Ioni and Brem macroscopic scross secion for the " << numHepEmMCCData
//            << "\n material-cuts couples used in the geometry. "
//            << std::endl;
  elData->fResMacXSecNumData = indxCont;
  elData->fResMacXSecData = new double[indxCont]{};
  for (int i=0; i<indxCont; ++i) {
    elData->fResMacXSecData[i] = xsecData[i];
  }
  //
  // free all dynamically allocated auxilary memory
  delete[] energyGrid;
  delete[] macXSec;
  delete[] secDerivs;
  delete[] xsecData;
}


void BuildNuclearLambdaTables(G4CrossSectionDataStore* hadENucXSDataStore, struct G4HepEmData* hepEmData,
                              struct G4HepEmParameters* /*hepEmParams*/, bool iselectron) {
  // get the pointer to the already allocated G4HepEmElectronData from the HepEmData
  struct G4HepEmElectronData* elData = iselectron
                                       ? hepEmData->fTheElectronData
                                       : hepEmData->fThePositronData;
  //
  // get the g4 particle-definition
  G4ParticleDefinition* g4PartDef = G4Positron::Positron();
  if  (iselectron) {
    g4PartDef = G4Electron::Electron();
  }
  //
  // generate the energy grid (common for all materials)
  const double emin = 100.0*CLHEP::MeV;
  const double emax = 100.0*CLHEP::TeV;
  int numENucEkin = elData->fENucEnergyGridSize;
  delete [] elData->fENucEnergyGrid;
  elData->fENucEnergyGrid = new double[numENucEkin]{};
  G4HepEmInitUtils::FillLogarithmicGrid(emin, emax, numENucEkin, elData->fENucLogMinEkin, elData->fENucEILDelta, elData->fENucEnergyGrid);
  //
  // allocate some array for intermediate storage of the nuclear MXsec and its second
  // derivatives for a given material
  double* theENucMXsec     = new double[numENucEkin]{};
  double* theENucMXsecSD   = new double[numENucEkin]{};
  // allocate the array to store (continuously) all macroscopic xsec
  const int numMaterials   = elData->fNumMaterials;
  elData->fENucMacXsecData = new double[2*numENucEkin*numMaterials]{};
  //
  // loop over the HepEm materials and for each:
  // - get the corresponding G4Material
  // - compute the macroscopic electron-nuclear cross section at each of the
  //   discrte kinetic energies
  //
  // get the HepEm material data
  const struct G4HepEmMaterialData*  hepEmMatData = hepEmData->fTheMaterialData;
  // get the correspondibg G4Material table (i.e. global vector of G4Material*)
  const G4MaterialTable* theG4MaterialTable = G4Material::GetMaterialTable();
  // a g4 dynamic particle will be needed
  G4DynamicParticle* dyPart = new G4DynamicParticle(g4PartDef, G4ThreeVector(0,0,1), 0);
  for (int im=0; im<numMaterials; ++im) {
    const struct G4HepEmMatData& matData = hepEmMatData->fMaterialData[im];
    const G4Material* g4Mat = (*theG4MaterialTable)[matData.fG4MatIndex];
    // std::cout << " ===== Material = " << g4Mat->GetName() << std::endl;
    // loop over the kinetic energies and comput the electron-nuclear mxsec
    for (int ie=0; ie<numENucEkin; ++ie) {
      const double ekin = ie==0 ? elData->fENucEnergyGrid[ie]+0.000001 : elData->fENucEnergyGrid[ie];
      dyPart->SetKineticEnergy(ekin);
      double mxsec = std::max(0.0, hadENucXSDataStore->ComputeCrossSection(dyPart, g4Mat));
      theENucMXsec[ie]   = mxsec;
      theENucMXsecSD[ie] = 0.0;
      // std::cout << " E = " << ekin << " [MeV] Sigam-ENuc(E) = " << mxsec << std::endl;
    }
    // set up a spline on the electron-nuclear MXsec array for interpolation
    G4HepEmInitUtils::PrepareSpline(numENucEkin, elData->fENucEnergyGrid, theENucMXsec, theENucMXsecSD);
    // write the data into its final location
    int iStart = 2*numENucEkin*im;
    for (int ie=0; ie<numENucEkin; ++ie) {
      elData->fENucMacXsecData[iStart++] = theENucMXsec[ie];
      elData->fENucMacXsecData[iStart++] = theENucMXsecSD[ie];
    }
  }
  // free auxilary arrays
  delete[] theENucMXsec;
  delete[] theENucMXsecSD;
  delete dyPart;
}


void BuildTransportXSectionTables(G4VEmModel* mscModel, struct G4HepEmData* hepEmData,
                                  struct G4HepEmParameters* /*hepEmParams*/, bool iselectron) {
  // get the pointer to the already allocated G4HepEmElectronData from the HepEmData
  struct G4HepEmElectronData* elData = iselectron
                                       ? hepEmData->fTheElectronData
                                       : hepEmData->fThePositronData;
  //
  // get the g4 particle-definition
  G4ParticleDefinition* g4PartDef = G4Positron::Positron();
  if  (iselectron) {
    g4PartDef = G4Electron::Electron();
  }
  //
  // NOTE: the energy grid is the same taht is used for the e-loss data and it
  //       has already been generated at this point (BuildELossTables should have
  //       been called before)
  const int numEner       = elData->fELossEnergyGridSize;
  const int numMaterials  = elData->fNumMaterials;
  //
  // allocate some array for intermediate storage of the TR1 MXsec and its second
  // derivatives for a given material
  double* theTr1MXsec     = new double[numEner]{};
  double* theTr1MXsecSD   = new double[numEner]{};
  // allocate the array to store (continuously) all macroscopic first tr. xsec
  elData->fTr1MacXSecData = new double[2*numEner*numMaterials]{};
  //
  // loop over the HepEm materials and for each:
  // - get the corresponding G4Material
  // - compute the first transprt cross section per volume (i.e. first transport
  //   macroscopic cross section, since the G4VMscModel-s implements transport
  //   cross section per atom in the ComputeCrossSectionPerAtom interface) at
  //   each of the discrte kinetic energies
  //
  // get the HepEm material data
  const struct G4HepEmMaterialData*  hepEmMatData = hepEmData->fTheMaterialData;
  // get the correspondibg G4Material table (i.e. global vector of G4Material*)
  const G4MaterialTable* theG4MaterialTable = G4Material::GetMaterialTable();
  for (int im=0; im<numMaterials; ++im) {
    const struct G4HepEmMatData& matData = hepEmMatData->fMaterialData[im];
    const G4Material* g4Mat = (*theG4MaterialTable)[matData.fG4MatIndex];
    // loop over the kinetic energies and comput the tr1 mxsec
    for (int ie=0; ie<numEner; ++ie) {
      double ekin = elData->fELossEnergyGrid[ie];
#if G4VERSION_NUMBER >= 1100
      if (std::abs(ekin - 10.0) < 1.0E-6) {
        ekin = 10.0;
      }
#endif
      double tr1mxsec   = std::max(0.0, mscModel->CrossSectionPerVolume(g4Mat, g4PartDef, ekin));
      theTr1MXsec[ie]   = tr1mxsec;
      theTr1MXsecSD[ie] = 0.0;
    }
    // set up a spline on the TR1 MXsec array for interpolation
    G4HepEmInitUtils::PrepareSpline(numEner, elData->fELossEnergyGrid, theTr1MXsec, theTr1MXsecSD);
    // write the data into its final location
    int iStart = 2*numEner*im;
    for (int ie=0; ie<numEner; ++ie) {
      elData->fTr1MacXSecData[iStart++] = theTr1MXsec[ie];
      elData->fTr1MacXSecData[iStart++] = theTr1MXsecSD[ie];
    }
  }
  // free auxilary arrays
  delete[] theTr1MXsec;
  delete[] theTr1MXsecSD;
}


void BuildElementSelectorTables(G4MollerBhabhaModel* mbModel, G4SeltzerBergerModel* sbModel,
                       G4eBremsstrahlungRelModel* rbModel, struct G4HepEmData* hepEmData,
                       struct G4HepEmParameters* hepEmParams, bool iselectron) {
  // get the pointer to the already allocated G4HepEmElectronData from the HepEmData
  struct G4HepEmElectronData* elData = iselectron
                                       ? hepEmData->fTheElectronData
                                       : hepEmData->fThePositronData;
  //
  // get the g4 particle-definition
  G4ParticleDefinition* g4PartDef = G4Positron::Positron();
  if  (iselectron) {
    g4PartDef = G4Electron::Electron();
  }
  // get the HepEm Material-cut couple data
  const struct G4HepEmMatCutData*    hepEmMCData = hepEmData->fTheMatCutData;
  const struct G4HepEmMaterialData* hepEmMatData = hepEmData->fTheMaterialData;
  // number of HepEm material-cuts couples
  int numHepEmMCCData = hepEmMCData->fNumMatCutData;
  // estimate buffer size by counting #element and energy grids
  int num = 0;
  for (int imc=0; imc<numHepEmMCCData; ++imc) {
    const struct G4HepEmMCCData& mccData = hepEmMCData->fMatCutData[imc];
    const struct G4HepEmMatData& matData = hepEmMatData->fMaterialData[mccData.fHepEmMatIndex];
    int numElem = matData.fNumOfElement;
    if (numElem>1) {
      num += numElem+1; // +1 for the enrgy grid
    }
  }
  // allocate buffer
  double* ioniData   = new double[(hepEmParams->fNumLossTableBins+4)*num]{};
  double* bremSBData = new double[(hepEmParams->fNumLossTableBins+4)*num]{};
  double* bremRBData = new double[(hepEmParams->fNumLossTableBins+4)*num]{};
  //
  // allocate the arrays to store start indices per matrial-cuts couples
  elData->fElemSelectorIoniStartIndexPerMatCut   = new int[numHepEmMCCData]{};
  elData->fElemSelectorBremSBStartIndexPerMatCut = new int[numHepEmMCCData]{};
  elData->fElemSelectorBremRBStartIndexPerMatCut = new int[numHepEmMCCData]{};
  //
  int numBinsPerDecade = G4EmParameters::Instance()->NumberOfBinsPerDecade();
  // a continuous index
  int indxContIoni   = 0;
  int indxContBremSB = 0;
  int indxContBremRB = 0;
  for (int imc=0; imc<numHepEmMCCData; ++imc) {
    // get the hepEm mat-cut and material structures
    const struct G4HepEmMCCData& mccData  = hepEmMCData->fMatCutData[imc];
    const struct G4HepEmMatData& matData  = hepEmMatData->fMaterialData[mccData.fHepEmMatIndex];
    int numElem = matData.fNumOfElement;
    // no element selectors for single elemnt materials
    if (numElem<2) {
      elData->fElemSelectorIoniStartIndexPerMatCut[imc]   = -1;
      elData->fElemSelectorBremSBStartIndexPerMatCut[imc] = -1;
      elData->fElemSelectorBremRBStartIndexPerMatCut[imc] = -1;
      continue;
    }

    // get the the secondary e- and gamma production energy thresholds
    const double     elCutE = mccData.fSecElProdCutE;  // already includes e- tracking cut
    const double    gamCutE = mccData.fSecGamProdCutE;
    //
    // ===== Ionisation
    //
    // generate the kinetic energy grid for this material-cut for ioni
    double    minEKin = iselectron ? 2*elCutE : elCutE;
    double    maxEKin = hepEmParams->fMaxLossTableEnergy;
    if (minEKin>=maxEKin) {
      elData->fElemSelectorIoniStartIndexPerMatCut[imc] = -1;
    } else {
      // fill in the first values as #data
      elData->fElemSelectorIoniStartIndexPerMatCut[imc] = indxContIoni;
      BuildElementSelector(minEKin, maxEKin, numBinsPerDecade, ioniData, indxContIoni, matData, mbModel, elCutE, g4PartDef);
    }
    //
    // ===== Brem: Seltzer-Berger
    //
    // generate the kinetic energy grid for this material-cut for sb-brem
    minEKin = gamCutE;
    maxEKin = hepEmParams->fElectronBremModelLim;
    if (minEKin >= maxEKin) {
      // no element selector for this mat-cut in case of SB-brem since the interaction cannot happen
      elData->fElemSelectorBremSBStartIndexPerMatCut[imc] = -1;
    } else  {
      elData->fElemSelectorBremSBStartIndexPerMatCut[imc] = indxContBremSB;
      BuildElementSelector(minEKin, maxEKin, numBinsPerDecade, bremSBData, indxContBremSB, matData, sbModel, gamCutE, g4PartDef);
    }
    //
    // ===== Brem: Relativistic
    //
    // generate the kinetic energy grid for this material-cut for rel-brem
    minEKin = std::max(gamCutE, hepEmParams->fElectronBremModelLim);
    maxEKin = hepEmParams->fMaxLossTableEnergy;
    if (minEKin >= maxEKin) {
      // no element selector for this mat-cut in case of SB-brem since the interaction cannot happen
      elData->fElemSelectorBremRBStartIndexPerMatCut[imc] = -1;
    } else  {
      elData->fElemSelectorBremRBStartIndexPerMatCut[imc] = indxContBremRB;
      BuildElementSelector(minEKin, maxEKin, numBinsPerDecade, bremRBData, indxContBremRB, matData, rbModel, gamCutE, g4PartDef);

    }
  }

  // write data to the final destination and clean all dynamically allocated auxilary memory
  elData->fElemSelectorIoniNumData = indxContIoni;
  if (indxContIoni > 0) {
    elData->fElemSelectorIoniData    = new double[indxContIoni]{};
    for (int i=0; i<indxContIoni; ++i) {
      elData->fElemSelectorIoniData[i] = ioniData[i];
    }
  }

  elData->fElemSelectorBremSBNumData = indxContBremSB;
  if (indxContBremSB > 0) {
    elData->fElemSelectorBremSBData    = new double[indxContBremSB]{};
    for (int i=0; i<indxContBremSB; ++i) {
      elData->fElemSelectorBremSBData[i] = bremSBData[i];
    }
  }

  elData->fElemSelectorBremRBNumData = indxContBremRB;
  if (indxContBremRB > 0) {
    elData->fElemSelectorBremRBData    = new double[indxContBremRB]{};
    for (int i=0; i<indxContBremRB; ++i) {
      elData->fElemSelectorBremRBData[i] = bremRBData[i];
    }
  }

  delete[] ioniData;
  delete[] bremSBData;
  delete[] bremRBData;

}


void BuildElementSelector(double minEKin, double maxEKin, int numBinsPerDecade, double *data, int& indxCont, const struct G4HepEmMatData& matData, G4VEmModel* emModel, double cut, const G4ParticleDefinition* g4PartDef) {
  int     numElem    = matData.fNumOfElement;
  double  logMinEKin = 0.0;
  double  invLEDelta = 0.0;
  double egridData[500];
  int       numEKins = InitElementSelectorEnergyGrid(numBinsPerDecade, egridData, minEKin, maxEKin, logMinEKin, invLEDelta);
  // fill in the first 3 values as #data, logMinEKin, and invLodEDelta
  data[indxCont++]   = numEKins;
  data[indxCont++]   = numElem;
  data[indxCont++]   = logMinEKin;
  data[indxCont++]   = invLEDelta;
  // loop over the kinetic energy grid
  for (int ie=0; ie<numEKins; ++ie) {
    double      ekin = egridData[ie];
    data[indxCont++] = ekin;
    int          ist = indxCont;
    double       sum = 0.0;
    for (int iz=0; iz<numElem; ++iz) {
      // compute atomic cross section x number of atoms per volume
      int      izet = matData.fElementVect[iz];
      double natoms = matData.fNumOfAtomsPerVolumeVect[iz];
      const G4Element* g4Elem  = G4NistManager::Instance()->FindOrBuildElement(izet);
      double   xsec = std::max(0.0, emModel->ComputeCrossSectionPerAtom(g4PartDef, g4Elem, ekin, cut, ekin));
      sum += natoms*xsec;
      if (iz<numElem-1) {
        data[indxCont++] = sum;
      }
    }
    // normalise
    for (int i=0; i<numElem-1; ++i) {
      if (sum>0.0) {
        data[ist+i] /= sum;
      }
    }
  }
}

int InitElementSelectorEnergyGrid(int binsperdecade, double* egrid, double mine, double maxe, double& logMinEnergy, double& invLEDelta) {
  const double invlog106 = 1.0/(6.0*std::log(10.0));
  int numEnergyBins = (int)(binsperdecade*std::log(maxe/mine)*invlog106);
  if (numEnergyBins<3) {
    numEnergyBins = 3;
  }
  ++numEnergyBins;

  G4HepEmInitUtils::FillLogarithmicGrid(mine, maxe, numEnergyBins, logMinEnergy, invLEDelta, egrid);

  return numEnergyBins;
}


void BuildSBBremSTables(struct G4HepEmData* hepEmData, struct G4HepEmParameters* hepEmPars, G4SeltzerBergerModel* sbModel) {
  G4HepEmSBBremTableBuilder* sbTables = new G4HepEmSBBremTableBuilder();
  sbTables->Initialize( std::max(hepEmPars->fElectronTrackingCut, sbModel->LowEnergyLimit()), sbModel->HighEnergyLimit());

  // we need the HepEm-MatCut data to convert G4-mc indices to hepEm-mc indices
  // we need the HepEm-element data to loop over the required Z values
  const G4HepEmMatCutData*   theMCData   = hepEmData->fTheMatCutData;
  const G4HepEmMaterialData* theMatData  = hepEmData->fTheMaterialData;
  const G4HepEmElementData*  theElemData = hepEmData->fTheElementData;

  // count: #hepEM-mc, sum-#elements-in-mcs, #unique-elements
  int numHepEmMatCuts = theMCData->fNumMatCutData;
  int numElemsInMC    = 0;
  for (int imc=0; imc<numHepEmMatCuts; ++imc) {
    // #elments of the material in this material cut couple
    numElemsInMC += theMatData->fMaterialData[theMCData->fMatCutData[imc].fHepEmMatIndex].fNumOfElement;
  }
//  int numElemsUnique  = 0;
//  for (int iz=1; iz<theElemData->fMaxZet; ++iz) {
//    if (theElemData->fElementData[iz].fZet > -1.0) {
//      ++numElemsUnique;
//    }
//  }

  // loop over the elements used in the geometry
  int numSBData = 0;
  for (int iz=1; iz<theElemData->fMaxZet; ++iz) {
    const int izet = (int)(theElemData->fElementData[iz].fZet);
    if (izet<0) {
      continue;
    }
    int izST = std::min(iz, sbTables->fMaxZet);
    // and construct the HepEm-SB-sampling tables for each
    const G4HepEmSBBremTableBuilder::SamplingTablePerZ* stPerZ = sbTables->GetSamplingTablesForZ(izST);
    if (stPerZ == nullptr) {
      std::cout << " *** No SB-STable for iz = " << iz << ": gcut probably above max-model-energy of 1 GeV " << std::endl;
      continue;
    }
    // #sampling tables i.e. energy grid = stPerZ->fMaxElEnergyIndx - stPerZ->fMinElEnergyIndx + 1
    // 54 + #gamma-cuts for this Z elememnts at each energy grid
    // + 4 values: [0] #data; [1] min-; [2] max-energy grid index; [3] #mat-cuts this Z appears (with g-cut below 1 geV)
    int num = (stPerZ->fMaxElEnergyIndx - stPerZ->fMinElEnergyIndx + 1)*(stPerZ->fNumGammaCuts + 3*sbTables->fNumKappa) + 4;
    numSBData += num;
/*
    std::cout << " ======= SB Table for Z = " << iz << std::endl;
    std::cout << "  - # gamm-cuts      = " << stPerZ->fNumGammaCuts    << std::endl;
    std::cout << "  - fMinElEnergyIndx = " << stPerZ->fMinElEnergyIndx << "  E[x] = " << sbTables->fElEnergyVect[stPerZ->fMinElEnergyIndx]<< std::endl;
    std::cout << "  - fMaxElEnergyIndx = " << stPerZ->fMaxElEnergyIndx << "  E[x] = " << sbTables->fElEnergyVect[stPerZ->fMaxElEnergyIndx]<< std::endl;
    std::cout << "  size of 'fTablesPerEnergy' = " << stPerZ->fTablesPerEnergy.size() << std::endl;
    std::cout << "     null     ? " << stPerZ->fTablesPerEnergy[std::max(0,stPerZ->fMinElEnergyIndx-1)] << std::endl;
    std::cout << "     not null ? " << stPerZ->fTablesPerEnergy[stPerZ->fMinElEnergyIndx] << std::endl;
    std::cout << "     not null ? " << stPerZ->fTablesPerEnergy[stPerZ->fMaxElEnergyIndx] << std::endl;
    std::cout << "     null     ? " << stPerZ->fTablesPerEnergy[stPerZ->fMaxElEnergyIndx+1] << std::endl;
*/
  }


  // allocate G4HepEmSBTables data structure
  AllocateSBTableData(&(hepEmData->fTheSBTableData), numHepEmMatCuts, numElemsInMC, numSBData);
  G4HepEmSBTableData *sbData = hepEmData->fTheSBTableData;


  // check and allert !
//  std::cout << "  == max-Z  = " << sbData->fMaxZet      << " v.s. " << sbTables->fMaxZet << std::endl;
//  std::cout << "  == #E     = " << sbData->fNumElEnergy << " v.s. " << sbTables->fNumElEnergy << std::endl;
//  std::cout << "  == #Kappa = " << sbData->fNumKappa    << " v.s. " << sbTables->fNumKappa << std::endl;

  sbData->fLogMinElEnergy  = sbTables->fLogMinElEnergy;
  sbData->fILDeltaElEnergy = sbTables->fILDeltaElEnergy;
  // copy electron energy and kappa value grids
  for (int ie=0; ie<sbTables->fNumElEnergy; ++ie) {
    sbData->fElEnergyVect[ie]  = sbTables->fElEnergyVect[ie];
    sbData->fLElEnergyVect[ie] = sbTables->fLElEnergyVect[ie];
  }
  for (int ik=0; ik<sbTables->fNumKappa; ++ik) {
    sbData->fKappaVect[ik]  = sbTables->fKappaVect[ik];
    sbData->fLKappaVect[ik] = sbTables->fLKappaVect[ik];
  }
  //

  //int indxCumKappaVals    = 0;
  int indxCumSBTableData  = 0;
  // loop over the elements used in the geometry and construct HepEM SB-tables
  for (int iz=1; iz<theElemData->fMaxZet; ++iz) {
    const int izet = (int)(theElemData->fElementData[iz].fZet);
    if (izet<0) {
      continue;
    }
    int izST = std::min(iz,sbTables->fMaxZet);
    // and construct the HepEm-SB-sampling tables for each
    const G4HepEmSBBremTableBuilder::SamplingTablePerZ* stPerZ = sbTables->GetSamplingTablesForZ(izST);
    if (stPerZ == nullptr) {
      continue;
    }
    // Construct the HepEm-Samplng-tables for this Z:
    // 1. record where the S-tables start in fSBTableData for this Z (iz)
    sbData->fSBTablesStartPerZ[iz] = indxCumSBTableData;
    // 2. fill in the first 4 values:
    int minEindex      = stPerZ->fMinElEnergyIndx;
    int maxEindex      = stPerZ->fMaxElEnergyIndx;
    int numGammaCuts   = stPerZ->fNumGammaCuts;
    int numData        = (maxEindex - minEindex + 1)*(numGammaCuts + 3*sbTables->fNumKappa)+4;
    sbData->fSBTableData[indxCumSBTableData++] = numData;
    sbData->fSBTableData[indxCumSBTableData++] = minEindex;
    sbData->fSBTableData[indxCumSBTableData++] = maxEindex;
    sbData->fSBTableData[indxCumSBTableData++] = numGammaCuts;
    for (int ist=minEindex; ist<=maxEindex; ++ist) {
      const G4HepEmSBBremTableBuilder::STable* stPerE = stPerZ->fTablesPerEnergy[ist];
      for (int igc=0; igc<numGammaCuts; ++igc) {
        sbData->fSBTableData[indxCumSBTableData++] = stPerE->fCumCutValues[igc];
      }
      for (int ik=0; ik<sbTables->fNumKappa; ++ik) {
        sbData->fSBTableData[indxCumSBTableData++] = stPerE->fSTable[ik].fCum;
        sbData->fSBTableData[indxCumSBTableData++] = stPerE->fSTable[ik].fParA;
        sbData->fSBTableData[indxCumSBTableData++] = stPerE->fSTable[ik].fParB;
      }
    }
  }
  sbData->fNumSBTableData = indxCumSBTableData;
  // 3. fill the HepEm-mc index to gamma-cut index (in a given Zet data) translation
  // loop over the HepEm mat-cuts, get the corresponding g4 mat-cut index and
  // loop over the elements of the HepEm mat-cut material and get the gamma-cut index
  // for each element
  int indxCumKappaCut = 0;
  for (int imc=0; imc<numHepEmMatCuts; ++imc) {
    const G4HepEmMCCData& hepEmMCData = theMCData->fMatCutData[imc];
    int  iMCG4     = hepEmMCData.fG4MatCutIndex;
    int  iMatHepEM = hepEmMCData.fHepEmMatIndex;
    int  numElems  = theMatData->fMaterialData[iMatHepEM].fNumOfElement;
    int* elemVect  = theMatData->fMaterialData[iMatHepEM].fElementVect;
    bool isfirst = true;
    for (int ie=0; ie<numElems; ++ie) {
      int izST = std::min(elemVect[ie], sbTables->fMaxZet);
      const G4HepEmSBBremTableBuilder::SamplingTablePerZ* stPerZ = sbTables->GetSamplingTablesForZ(izST);
      if (stPerZ == nullptr) {
        continue;
      }
      // get the index of the gamma-cut in this element that corresponds to this mat-cut
      int indx = stPerZ->fMatCutIndxToGamCutIndx[iMCG4];
      if (indx>-1) {
        if (isfirst) {
          sbData->fGammaCutIndxStartIndexPerMC[imc] = indxCumKappaCut;
          isfirst = false;
        }
        sbData->fGammaCutIndices[indxCumKappaCut++] = indx;
//        std::cout << " ==> imc = "<< imc<< " g-cut index for Z = " << izST << " ==>" << sbData->fGammaCutIndices[indxCumKappaCut-1] << std::endl;
      }
    }
  }
}
