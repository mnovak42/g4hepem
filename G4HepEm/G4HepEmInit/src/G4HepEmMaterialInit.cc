
#include "G4HepEmMaterialInit.hh"

#include "G4HepEmData.hh"
#include "G4HepEmParameters.hh"

#include "G4HepEmMatCutData.hh"
#include "G4HepEmMaterialData.hh"
#include "G4HepEmElementData.hh"

// g4 includes
#include "G4Version.hh"
#include "G4ProductionCutsTable.hh"
#include "G4SandiaTable.hh"
#include "G4MaterialCutsCouple.hh"
#include "G4Material.hh"
#include "G4Element.hh"
#include "G4ElementVector.hh"
#include "G4RegionStore.hh"

#include <vector>
#include <iostream>


// - translates all G4MaterialCutsCouple, used in the current geometry, to a
//   G4HepEmMatCutData structure element used by G4HepEm
// - generates a G4HepEmMaterialData structure that stores material information
//   for all unique materials, used in the current geometry
// - builds the G4HepEmElementData structure
void InitMaterialAndCoupleData(struct G4HepEmData* hepEmData, struct G4HepEmParameters* hepEmPars) {
  // get material-cuts couple and material tables
  G4ProductionCutsTable* theCoupleTable = G4ProductionCutsTable::GetProductionCutsTable();
  G4MaterialTable*     theMaterialTable = G4Material::GetMaterialTable();
  size_t numG4MatCuts = theCoupleTable->GetTableSize();
  size_t numG4Mat     = theMaterialTable->size();
  //  theCoupleTable->DumpCouples();
  //  G4cout << *theMaterialTable;
  //
  // Allocate temporary vectors to query the Sandia coefficients per atom.
  std::vector<double> sandiaCofsPerAtom(4, 0.0);
  std::vector<double> lastSandiaCofsPerAtom(4, 0.0);
  // Allocate temporary vectors to store the energy intervals and coefficients
  // for the current element.
  std::vector<double> sandiaEnergies;
  std::vector<double> sandiaCoefficients;
  //
  // 0. count G4MaterialCutsCouple and unique G4Material objects that are used in
  //    the courrent geometry. Record the indices of the used, unique materials.
  G4int  numUsedG4MatCuts = 0;
  G4int  numUsedG4Mat     = 0;
  std::vector<G4int> theUsedG4MatIndices(numG4Mat, -2);
  for (size_t imc=0; imc<numG4MatCuts; ++imc) {
    const G4MaterialCutsCouple* matCut = theCoupleTable->GetMaterialCutsCouple(imc);
    if (!matCut->IsUsed()) {
      continue;
    }
    ++numUsedG4MatCuts;
    size_t matIndx = matCut->GetMaterial()->GetIndex();
    if (theUsedG4MatIndices[matIndx]>-2) {
      continue;
    }
    // mark to be used in the geometry (but only once: numUsedG4Mat = unique used mats)
    theUsedG4MatIndices[matIndx] = -1;
    ++numUsedG4Mat;
  }

  // 1. Allocate the MatCutData and MaterialData sub-structure of the global G4HepEmData
  AllocateMatCutData   (&(hepEmData->fTheMatCutData), numG4MatCuts, numUsedG4MatCuts);
  AllocateMaterialData (&(hepEmData->fTheMaterialData), numG4Mat, numUsedG4Mat);
  AllocateElementData  (&(hepEmData->fTheElementData));
  //
  // auxiliary arrays for screening: complete coherent and incoherent screening constant computed by using the DF model of the atom.
  const double kFelLowZet  [] = { 0.0, 5.3104, 4.7935, 4.7402, 4.7112, 4.6694, 4.6134, 4.5520 };
  const double kFinelLowZet[] = { 0.0, 5.9173, 5.6125, 5.5377, 5.4728, 5.4174, 5.3688, 5.3236 };
  // 2. Fill them in
  numUsedG4MatCuts = 0;
  numUsedG4Mat     = 0;
  for (size_t imc=0; imc<numG4MatCuts; ++imc) {
    const G4MaterialCutsCouple* matCut = theCoupleTable->GetMaterialCutsCouple(imc);
    if (!matCut->IsUsed()) {
      continue;
    }
    G4int mcIndx          = matCut->GetIndex();
    G4double gammaCutE    = (*(theCoupleTable->GetEnergyCutsVector(0)))[mcIndx];
    G4double electronCutE = (*(theCoupleTable->GetEnergyCutsVector(1)))[mcIndx];
    G4double positronCutE = (*(theCoupleTable->GetEnergyCutsVector(2)))[mcIndx];
    const G4Material* mat = matCut->GetMaterial();
    G4int matIndx         = mat->GetIndex();

    hepEmData->fTheMatCutData->fG4MCIndexToHepEmMCIndex[mcIndx] = numUsedG4MatCuts;
    struct G4HepEmMCCData& mccData = hepEmData->fTheMatCutData->fMatCutData[numUsedG4MatCuts];

    // NOTE: mccData.fHepEmMatIndex will be set below when it becomes known
    mccData.fG4MatCutIndex  = mcIndx;
    mccData.fSecElProdCutE  = std::max(electronCutE, hepEmPars->fElectronTrackingCut);
    mccData.fSecPosProdCutE = positronCutE;
    mccData.fSecGamProdCutE = gammaCutE;
    mccData.fLogSecGamCutE  = std::log(gammaCutE);
    mccData.fG4RegionIndex  = -1; // will be set at the very end
    ++numUsedG4MatCuts;

    // check if the corresponding G4HepEm material struct has already been created
    if (theUsedG4MatIndices[matIndx]>-1) {
      // already created:
      mccData.fHepEmMatIndex = theUsedG4MatIndices[matIndx];
    } else {
      // material structure has not been created so do it
      // take care of the corresponding element dat as well
      hepEmData->fTheMaterialData->fG4MatIndexToHepEmMatIndex[matIndx] = numUsedG4Mat;
      struct G4HepEmMatData& matData = hepEmData->fTheMaterialData->fMaterialData[numUsedG4Mat];
      //
      const G4ElementVector*  elmVec = mat->GetElementVector();
      const G4double* nAtomPerVolVec = mat->GetVecNbOfAtomsPerVolume();
      size_t            numOfElement = mat->GetNumberOfElements();
      //
      matData.fG4MatIndex              = matIndx;
      matData.fNumOfElement            = numOfElement;
      matData.fElementVect             = new int[numOfElement]{};
      matData.fNumOfAtomsPerVolumeVect = new double[numOfElement]{};
      matData.fDensity                 = mat->GetDensity();
      matData.fDensityCorFactor        = 4.0*CLHEP::pi*CLHEP::classic_electr_radius*CLHEP::electron_Compton_length*CLHEP::electron_Compton_length*mat->GetElectronDensity();
      matData.fElectronDensity         = mat->GetElectronDensity();
      matData.fRadiationLength         = mat->GetRadlen();
      matData.fMeanExEnergy            = mat->GetIonisation()->GetMeanExcitationEnergy();

      // go for some U-msc related data per materials
      const double zeff                = mat->GetIonisation()->GetZeffective();
      const double zeff16              = std::pow(zeff, 1.0/6.0);
      const double zeff13              = zeff16*zeff16;
      matData.fZeff                    = zeff;
      matData.fZeff23                  = zeff13*zeff13;
      matData.fZeffSqrt                = std::sqrt(zeff);
      //
// these parameters are taken from Geant4-11.0 (below are the values used before)
// See G4HepEmElectronInteractionUMSC::StepLimit for further details on this.
#if G4VERSION_NUMBER >= 1070
      matData.fUMSCPar                 = 9.62800E-1 - 8.4848E-2*matData.fZeffSqrt + 4.3769E-3*zeff;
      matData.fUMSCStepMinPars[0]      = 2.7725E+1/(1.0 + 2.03E-1*zeff);
      matData.fUMSCStepMinPars[1]      = 6.152    /(1.0 + 1.11E-1*zeff);
#else // G4 version before 10.7
      matData.fUMSCPar                 = 1.2 - zeff*(1.62e-2 - 9.22e-5*zeff);
      matData.fUMSCStepMinPars[0]      = 15.99/(1. + 0.119*zeff);
      matData.fUMSCStepMinPars[1]      = 4.390/(1. + 0.079*zeff);
#endif
      const double dum0                = 9.90395E-1 + zeff16*(-1.68386E-1 + zeff16*9.3286E-2);
      matData.fUMSCThetaCoeff[0]       = dum0*(1.0 - 8.7780E-2/zeff);
      matData.fUMSCThetaCoeff[1]       = dum0*(4.0780E-2 + 1.7315E-4*zeff);
      matData.fUMSCTailCoeff[0]        = 2.3785    - zeff13*(4.1981E-1 - zeff13*6.3100E-2);
      matData.fUMSCTailCoeff[1]        = 4.7526E-1 + zeff13*(1.7694    - zeff13*3.3885E-1);
      matData.fUMSCTailCoeff[2]        = 2.3683E-1 - zeff13*(1.8111    - zeff13*3.2774E-1);
      matData.fUMSCTailCoeff[3]        = 1.7888E-2 + zeff13*(1.9659E-2 - zeff13*2.6664E-3);

      // Copy the intervals from the table.
      G4SandiaTable* sandia         = mat->GetSandiaTable();
      int matNbOfSandiaIntervals    = sandia->GetMatNbOfIntervals();
      matData.fNumOfSandiaIntervals = matNbOfSandiaIntervals;
      matData.fSandiaEnergies       = new double[matNbOfSandiaIntervals]{};
      matData.fSandiaCoefficients   = new double[4 * matNbOfSandiaIntervals]{};
      for (int i = 0; i < matNbOfSandiaIntervals; i++) {
        matData.fSandiaEnergies[i] = sandia->GetSandiaCofForMaterial(i, 0);
        for (int j = 0; j < 4; j++) {
          // j + 1 because the first entry is the energy (see above)
          matData.fSandiaCoefficients[i * 4 + j] = sandia->GetSandiaCofForMaterial(i, j + 1);
        }
      }
      //
      for (size_t ie=0; ie<numOfElement; ++ie) {
        G4int izet = ((*elmVec)[ie])->GetZasInt();
        matData.fElementVect[ie] = izet;
        matData.fNumOfAtomsPerVolumeVect[ie] = nAtomPerVolVec[ie];
        // fill element data as well if haven't done yet
        izet = std::min ( izet, (G4int)hepEmData->fTheElementData->fMaxZet );
        struct G4HepEmElemData& elData = hepEmData->fTheElementData->fElementData[izet];
        if (elData.fZet<0) {
          double dZet          = (double)izet;
          elData.fZet          = dZet;
          elData.fZet13        = std::pow(dZet, 1.0/3.0);
          elData.fZet23        = std::pow(dZet, 2.0/3.0);
          elData.fCoulomb      = ((*elmVec)[ie])->GetfCoulomb();
          elData.fLogZ         = std::log(dZet);
          double Fel           = (izet<5) ? kFelLowZet[izet]   : std::log(184.15) -     elData.fLogZ/3.0;
          double Finel         = (izet<5) ? kFinelLowZet[izet] : std::log(1194.0) - 2.0*elData.fLogZ/3.0;
          elData.fZFactor1     = (Fel-elData.fCoulomb) + Finel/dZet;
          const double FZLow   = 8.0*elData.fLogZ/3.0;
          const double FZHigh  = 8.0*(elData.fLogZ/3.0 + elData.fCoulomb);
          elData.fDeltaMaxLow  = std::exp((42.038 - FZLow)/8.29) - 0.958;
          elData.fDeltaMaxHigh = std::exp((42.038 - FZHigh)/8.29) - 0.958;
          double varS1         = elData.fZet23/(184.15*184.15);
          elData.fILVarS1Cond  = 1./(std::log(std::sqrt(2.0)*varS1));
          elData.fILVarS1      = 1./std::log(varS1);

          // At the time of writing, G4SandiaTable::GetSandiaPerAtom is private and
          // GetSandiaCofPerAtom only takes an energy. Try the energy intervals from
          // the material, which we know must be a superset of those for the current
          // element, and discard / skip over duplicate intervals. A complication is
          // that GetSandiaCofPerAtom returns zeros if the energy is exactly Emin,
          // so add a small epsilon.
          constexpr double kEpsilon = 1e-9;
          // Now clear the last coefficients to make sure the first interval with
          // non-zero coefficients is taken.
          for (int i = 0; i < 4; i++) {
            lastSandiaCofsPerAtom[i] = 0.0;
          }

          int numberOfIntervals = 0;
          sandiaEnergies.clear();
          sandiaCoefficients.clear();
          for (int i = 0; i < matNbOfSandiaIntervals; i++) {
            const double energy = matData.fSandiaEnergies[i];
            sandia->GetSandiaCofPerAtom(izet, energy + kEpsilon, sandiaCofsPerAtom);
            if (sandiaCofsPerAtom == lastSandiaCofsPerAtom) {
              continue;
            }
            lastSandiaCofsPerAtom = sandiaCofsPerAtom;

            numberOfIntervals++;
            sandiaEnergies.push_back(energy);
            for (int j = 0; j < 4; j++) {
              sandiaCoefficients.push_back(sandiaCofsPerAtom[j]);
            }
          }

          elData.fNumOfSandiaIntervals = numberOfIntervals;
          elData.fSandiaEnergies       = new double[numberOfIntervals]{};
          elData.fSandiaCoefficients   = new double[4 * numberOfIntervals]{};
          for (int i = 0; i < numberOfIntervals; i++) {
            elData.fSandiaEnergies[i] = sandiaEnergies[i];
            for (int j = 0; j < 4; j++) {
              int idx = i * 4 + j;
              elData.fSandiaCoefficients[idx] = sandiaCoefficients[idx];
            }
          }
          elData.fKShellBindingEnergy = ((*elmVec)[ie])->GetAtomicShell(0);
        }
      }
      //
      theUsedG4MatIndices[matIndx] = numUsedG4Mat;
      mccData.fHepEmMatIndex = theUsedG4MatIndices[matIndx];
      ++numUsedG4Mat;
    }
  }
  // set the Geant4 detector region index for each material-cuts couple data
  for (std::size_t i=0; i<G4RegionStore::GetInstance()->size(); ++i) {
    G4Region* region = (*G4RegionStore::GetInstance())[i];
    const int indxRegion = region->GetInstanceID();
    std::vector<G4Material*>::const_iterator itrMat = region->GetMaterialIterator();
    for (std::size_t im=0; im<region->GetNumberOfMaterials(); ++im) {
      G4MaterialCutsCouple* couple = region->FindCouple(*itrMat);
      int indxMCC = hepEmData->fTheMatCutData->fG4MCIndexToHepEmMCIndex[couple->GetIndex()];
      hepEmData->fTheMatCutData->fMatCutData[indxMCC].fG4RegionIndex = indxRegion;
      // std::cout << (*itrMat)->GetName() << " " << indxRegion << std::endl;
      ++itrMat;
    }
  }
}
