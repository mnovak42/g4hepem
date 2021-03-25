#ifndef G4HepEmComparison_H
#define G4HepEmComparison_H

#include <tuple>

#include "G4HepEmData.hh"
#include "G4HepEmMatCutData.hh"
#include "G4HepEmElementData.hh"
#include "G4HepEmMaterialData.hh"
#include "G4HepEmElectronData.hh"
#include "G4HepEmSBTableData.hh"
#include "G4HepEmGammaData.hh"

// --- comparing two arrays...
template <typename T>
bool compare_arrays(int lhsSize, const T* lhsData, int rhsSize,
                    const T* rhsData)
{
  if(lhsSize != rhsSize)
  {
    return false;
  }

  // Same pointers (null or otherwise) are equal
  if(lhsData != rhsData)
  {
    if((lhsData == nullptr) || (rhsData == nullptr))
    {
      return false;
    }

    for(int i = 0; i < lhsSize; ++i)
    {
      if(lhsData[i] != rhsData[i])
      {
        return false;
      }
    }
  }

  return true;
}

// --- G4HepEmElemData
bool operator==(const G4HepEmElemData& lhs, const G4HepEmElemData& rhs)
{
  return std::tie(lhs.fZet, lhs.fZet13, lhs.fZet23, lhs.fCoulomb, lhs.fLogZ,
                  lhs.fZFactor1, lhs.fDeltaMaxLow, lhs.fDeltaMaxHigh,
                  lhs.fILVarS1, lhs.fILVarS1Cond) ==
         std::tie(rhs.fZet, rhs.fZet13, rhs.fZet23, rhs.fCoulomb, rhs.fLogZ,
                  rhs.fZFactor1, rhs.fDeltaMaxLow, rhs.fDeltaMaxHigh,
                  rhs.fILVarS1, rhs.fILVarS1Cond);
}

bool operator!=(const G4HepEmElemData& lhs, const G4HepEmElemData& rhs)
{
  return !(lhs == rhs);
}

// --- G4HepEmElementData
bool operator==(const G4HepEmElementData& lhs, const G4HepEmElementData& rhs)
{
  if(lhs.fMaxZet != rhs.fMaxZet)
  {
    return false;
  }

  return compare_arrays(lhs.fMaxZet + 1, lhs.fElementData, rhs.fMaxZet + 1,
                        rhs.fElementData);
}

bool operator!=(const G4HepEmElementData& lhs, const G4HepEmElementData& rhs)
{
  return !(lhs == rhs);
}

// --- G4HepEmMatCutData
bool operator==(const G4HepEmMCCData& lhs, const G4HepEmMCCData& rhs)
{
  return std::tie(lhs.fSecElProdCutE, lhs.fSecGamProdCutE, lhs.fLogSecGamCutE,
                  lhs.fHepEmMatIndex, lhs.fG4MatCutIndex) ==
         std::tie(rhs.fSecElProdCutE, rhs.fSecGamProdCutE, rhs.fLogSecGamCutE,
                  rhs.fHepEmMatIndex, rhs.fG4MatCutIndex);
}

bool operator!=(const G4HepEmMCCData& lhs, const G4HepEmMCCData& rhs)
{
  return !(lhs == rhs);
}

bool operator==(const G4HepEmMatCutData& lhs, const G4HepEmMatCutData& rhs)
{
  if(!compare_arrays(lhs.fNumG4MatCuts, lhs.fG4MCIndexToHepEmMCIndex,
                     rhs.fNumG4MatCuts, rhs.fG4MCIndexToHepEmMCIndex))
  {
    return false;
  }
  if(!compare_arrays(lhs.fNumMatCutData, lhs.fMatCutData, rhs.fNumMatCutData,
                     rhs.fMatCutData))
  {
    return false;
  }

  return true;
}

bool operator!=(const G4HepEmMatCutData& lhs, const G4HepEmMatCutData& rhs)
{
  return !(lhs == rhs);
}

// --- G4HepEmMaterialData
bool operator==(const G4HepEmMatData& lhs, const G4HepEmMatData& rhs)
{
  if(std::tie(lhs.fG4MatIndex, lhs.fNumOfElement, lhs.fDensity,
              lhs.fDensityCorFactor, lhs.fElectronDensity,
              lhs.fRadiationLength) !=
     std::tie(rhs.fG4MatIndex, rhs.fNumOfElement, rhs.fDensity,
              rhs.fDensityCorFactor, rhs.fElectronDensity,
              rhs.fRadiationLength))
  {
    return false;
  }

  if(!compare_arrays(lhs.fNumOfElement, lhs.fElementVect, rhs.fNumOfElement,
                     rhs.fElementVect))
  {
    return false;
  }

  if(!compare_arrays(lhs.fNumOfElement, lhs.fNumOfAtomsPerVolumeVect,
                     rhs.fNumOfElement, rhs.fNumOfAtomsPerVolumeVect))
  {
    return false;
  }

  return true;
}

bool operator!=(const G4HepEmMatData& lhs, const G4HepEmMatData& rhs)
{
  return !(lhs == rhs);
}

bool operator==(const G4HepEmMaterialData& lhs, const G4HepEmMaterialData& rhs)
{
  if(!compare_arrays(lhs.fNumG4Material, lhs.fG4MatIndexToHepEmMatIndex,
                     rhs.fNumG4Material, rhs.fG4MatIndexToHepEmMatIndex))
  {
    return false;
  }

  if(!compare_arrays(lhs.fNumMaterialData, lhs.fMaterialData,
                     lhs.fNumMaterialData, lhs.fMaterialData))
  {
    return false;
  }

  return true;
}

bool operator!=(const G4HepEmMaterialData& lhs, const G4HepEmMaterialData& rhs)
{
  return !(lhs == rhs);
}

// --- G4HepEmSBTableData
bool operator==(const G4HepEmSBTableData& lhs, const G4HepEmSBTableData& rhs)
{
  if(std::tie(lhs.fMaxZet, lhs.fNumElEnergy, lhs.fNumKappa, lhs.fLogMinElEnergy,
              lhs.fILDeltaElEnergy) !=
     std::tie(rhs.fMaxZet, lhs.fNumElEnergy, lhs.fNumKappa, lhs.fLogMinElEnergy,
              lhs.fILDeltaElEnergy))
  {
    return false;
  }

  if(!compare_arrays(lhs.fNumElEnergy, lhs.fElEnergyVect, rhs.fNumElEnergy,
                     rhs.fElEnergyVect))
  {
    return false;
  }
  if(!compare_arrays(lhs.fNumElEnergy, lhs.fLElEnergyVect, rhs.fNumElEnergy,
                     rhs.fLElEnergyVect))
  {
    return false;
  }
  if(!compare_arrays(lhs.fNumKappa, lhs.fKappaVect, rhs.fNumKappa,
                     rhs.fKappaVect))
  {
    return false;
  }
  if(!compare_arrays(lhs.fNumKappa, lhs.fLKappaVect, rhs.fNumKappa,
                     rhs.fLKappaVect))
  {
    return false;
  }

  if(!compare_arrays(lhs.fNumHepEmMatCuts, lhs.fGammaCutIndxStartIndexPerMC,
                     rhs.fNumHepEmMatCuts, lhs.fGammaCutIndxStartIndexPerMC))
  {
    return false;
  }
  if(!compare_arrays(lhs.fNumElemsInMatCuts, lhs.fGammaCutIndices,
                     rhs.fNumElemsInMatCuts, lhs.fGammaCutIndices))
  {
    return false;
  }

  // fixed val
  if(!compare_arrays(121, lhs.fSBTablesStartPerZ, 121, lhs.fSBTablesStartPerZ))
  {
    return false;
  }

  if(!compare_arrays(lhs.fNumSBTableData, lhs.fSBTableData, rhs.fNumSBTableData,
                     rhs.fSBTableData))
  {
    return false;
  }

  return true;
}

bool operator!=(const G4HepEmSBTableData& lhs, const G4HepEmSBTableData& rhs)
{
  return !(lhs == rhs);
}

// --- G4HepEmGammaData
bool operator==(const G4HepEmGammaData& lhs, const G4HepEmGammaData& rhs)
{
  // Data are a bit spread out, so go step by step rather than a full std::tie
  if(lhs.fNumMaterials != rhs.fNumMaterials)
  {
    return false;
  }

  // conversion data grid
  if(std::tie(lhs.fConvLogMinEkin, lhs.fConvEILDelta) !=
     std::tie(rhs.fConvLogMinEkin, rhs.fConvEILDelta))
  {
    return false;
  }

  if(!compare_arrays(lhs.fConvEnergyGridSize, lhs.fConvEnergyGrid,
                     rhs.fConvEnergyGridSize, rhs.fConvEnergyGrid))
  {
    return false;
  }

  // compton data grid
  if(std::tie(lhs.fCompLogMinEkin, lhs.fCompEILDelta) !=
     std::tie(rhs.fCompLogMinEkin, rhs.fCompEILDelta))
  {
    return false;
  }

  if(!compare_arrays(lhs.fCompEnergyGridSize, lhs.fCompEnergyGrid,
                     rhs.fCompEnergyGridSize, rhs.fCompEnergyGrid))
  {
    return false;
  }

  // the macroscopic cross sections for all materials and for
  // [conversion,compton] at each material
  const int lhsXsecSize =
    lhs.fNumMaterials * 2 * (lhs.fConvEnergyGridSize + lhs.fCompEnergyGridSize);
  const int rhsXsecSize =
    rhs.fNumMaterials * 2 * (rhs.fConvEnergyGridSize + rhs.fCompEnergyGridSize);
  if(!compare_arrays(lhsXsecSize, lhs.fConvCompMacXsecData, rhsXsecSize,
                     rhs.fConvCompMacXsecData))
  {
    return false;
  }

  if(std::tie(lhs.fElemSelectorConvLogMinEkin, lhs.fElemSelectorConvEILDelta) !=
     std::tie(rhs.fElemSelectorConvLogMinEkin, rhs.fElemSelectorConvEILDelta))
  {
    return false;
  }

  if(!compare_arrays(lhs.fNumMaterials, lhs.fElemSelectorConvStartIndexPerMat,
                     rhs.fNumMaterials, rhs.fElemSelectorConvStartIndexPerMat))
  {
    return false;
  }

  if(!compare_arrays(lhs.fElemSelectorConvEgridSize, lhs.fElemSelectorConvEgrid,
                     rhs.fElemSelectorConvEgridSize,
                     rhs.fElemSelectorConvEgrid))
  {
    return false;
  }

  if(!compare_arrays(lhs.fElemSelectorConvNumData, lhs.fElemSelectorConvData,
                     rhs.fElemSelectorConvNumData, rhs.fElemSelectorConvData))
  {
    return false;
  }

  return true;
}

bool operator!=(const G4HepEmGammaData& lhs, const G4HepEmGammaData& rhs)
{
  return !(lhs == rhs);
}

// --- G4HepEmElectronData
bool operator==(const G4HepEmElectronData& lhs, const G4HepEmElectronData& rhs)
{
  if(lhs.fNumMatCuts != rhs.fNumMatCuts)
  {
    return false;
  }

  if(std::tie(lhs.fELossLogMinEkin, lhs.fELossEILDelta) !=
     std::tie(lhs.fELossLogMinEkin, lhs.fELossEILDelta))
  {
    return false;
  }

  if(!compare_arrays(lhs.fELossEnergyGridSize, lhs.fELossEnergyGrid,
                     rhs.fELossEnergyGridSize, rhs.fELossEnergyGrid))
  {
    return false;
  }

  const int lhsELossDataSize = 5 * lhs.fELossEnergyGridSize * lhs.fNumMatCuts;
  const int rhsELossDataSize = 5 * rhs.fELossEnergyGridSize * rhs.fNumMatCuts;

  if(!compare_arrays(lhsELossDataSize, lhs.fELossData, rhsELossDataSize,
                     rhs.fELossData))
  {
    return false;
  }

  if(!compare_arrays(lhs.fNumMatCuts, lhs.fResMacXSecStartIndexPerMatCut,
                     rhs.fNumMatCuts, rhs.fResMacXSecStartIndexPerMatCut))
  {
    return false;
  }
  if(!compare_arrays(lhs.fResMacXSecNumData, lhs.fResMacXSecData,
                     rhs.fResMacXSecNumData, rhs.fResMacXSecData))
  {
    return false;
  }

  if(!compare_arrays(lhs.fNumMatCuts, lhs.fElemSelectorIoniStartIndexPerMatCut,
                     rhs.fNumMatCuts, rhs.fElemSelectorIoniStartIndexPerMatCut))
  {
    return false;
  }
  if(!compare_arrays(lhs.fElemSelectorIoniNumData, lhs.fElemSelectorIoniData,
                     rhs.fElemSelectorIoniNumData, rhs.fElemSelectorIoniData))
  {
    return false;
  }

  if(!compare_arrays(
       lhs.fNumMatCuts, lhs.fElemSelectorBremSBStartIndexPerMatCut,
       rhs.fNumMatCuts, rhs.fElemSelectorBremSBStartIndexPerMatCut))
  {
    return false;
  }
  if(!compare_arrays(
       lhs.fElemSelectorBremSBNumData, lhs.fElemSelectorBremSBData,
       rhs.fElemSelectorBremSBNumData, rhs.fElemSelectorBremSBData))
  {
    return false;
  }

  if(!compare_arrays(
       lhs.fNumMatCuts, lhs.fElemSelectorBremRBStartIndexPerMatCut,
       rhs.fNumMatCuts, rhs.fElemSelectorBremRBStartIndexPerMatCut))
  {
    return false;
  }
  if(!compare_arrays(
       lhs.fElemSelectorBremRBNumData, lhs.fElemSelectorBremRBData,
       rhs.fElemSelectorBremRBNumData, rhs.fElemSelectorBremRBData))
  {
    return false;
  }

  return true;
}

bool operator!=(const G4HepEmElectronData& lhs, const G4HepEmElectronData& rhs)
{
  return !(lhs == rhs);
}

// --- G4HepEmData
bool operator==(const G4HepEmData& lhs, const G4HepEmData& rhs)
{
  // Elements are pointers, but want to check underlying values
  // Can only do this for valid pointers, but nullptr == nullptr is fine
  if(!(lhs.fTheMatCutData == nullptr && rhs.fTheMatCutData == nullptr))
  {
    if(*(lhs.fTheMatCutData) != *(rhs.fTheMatCutData))
    {
      return false;
    }
  }

  if(!(lhs.fTheMaterialData == nullptr && rhs.fTheMaterialData == nullptr))
  {
    if(*(lhs.fTheMaterialData) != *(rhs.fTheMaterialData))
    {
      return false;
    }
  }

  if((lhs.fTheElementData != nullptr) && (rhs.fTheElementData != nullptr))
  {
    if(*(lhs.fTheElementData) != *(rhs.fTheElementData))
    {
      return false;
    }
  }

  if(!(lhs.fTheElectronData == nullptr && rhs.fTheElectronData == nullptr))
  {
    if(*(lhs.fTheElectronData) != *(rhs.fTheElectronData))
    {
      return false;
    }
  }

  if(!(lhs.fThePositronData == nullptr && rhs.fThePositronData == nullptr))
  {
    if(*(lhs.fThePositronData) != *(rhs.fThePositronData))
    {
      return false;
    }
  }

  if(!(lhs.fTheSBTableData == nullptr && rhs.fTheSBTableData == nullptr))
  {
    if(*(lhs.fTheSBTableData) != *(rhs.fTheSBTableData))
    {
      return false;
    }
  }

  if(!(lhs.fTheGammaData == nullptr && rhs.fTheGammaData == nullptr))
  {
    if(*(lhs.fTheGammaData) != *(rhs.fTheGammaData))
    {
      return false;
    }
  }

  return true;
}

bool operator!=(const G4HepEmData& lhs, const G4HepEmData& rhs)
{
  return !(lhs == rhs);
}

#endif  // G4HepEmComparison_H