#ifndef G4HepEmDataJsonIOImpl_H
#define G4HepEmDataJsonIOImpl_H

#include <exception>

#include "G4HepEmParameters.hh"
#include "G4HepEmData.hh"
#include "G4HepEmMatCutData.hh"
#include "G4HepEmElementData.hh"
#include "G4HepEmMaterialData.hh"
#include "G4HepEmElectronData.hh"
#include "G4HepEmSBTableData.hh"
#include "G4HepEmGammaData.hh"
#include "G4HepEmState.hh"

#include "nlohmann/json.hpp"

// for convenience
using json = nlohmann::json;

// As G4HepEm has a lot of dynamic C arrays, a minimal non-owning dynamic arrary
// type helps with serialization....
template <typename T>
struct dynamic_array
{
  int N   = 0;
  T* data = nullptr;

  dynamic_array() = default;

  dynamic_array(int n, T* d)
    : N(n)
    , data(d)
  {}

  T* begin() const { return data; }

  T* end() const { return data + N; }

  size_t size() const { return static_cast<size_t>(N); }
};

// This makes dynamic_array dual purpose as not just a view but a
// non-owning holder, so not ideal, but is internal so we can be careful
template <typename T>
dynamic_array<T> make_array(int n)
{
  if(n == 0)
  {
    return { 0, nullptr };
  }
  return { n, new T[n] };
}

template <typename T>
dynamic_array<T> make_span(int n, T* d)
{
  return { n, d };
}

// Free functions for static C arrays, from
// https://stackoverflow.com/questions/60328339/json-to-an-array-of-structs-in-nlohmann-json-lib
template <typename T, size_t N>
void to_json(json& j, T (&t)[N])
{
  for(size_t i = 0; i < N; ++i)
  {
    j.push_back(t[i]);
  }
}

template <typename T, size_t N>
void from_json(const json& j, T (&t)[N])
{
  if(j.size() != N)
  {
    throw std::runtime_error("JSON array size is different than expected");
  }
  size_t index = 0;
  for(auto& item : j)
  {
    from_json(item, t[index++]);
  }
}

namespace nlohmann
{
  template <typename T>
  struct adl_serializer<dynamic_array<T>>
  {
    static void to_json(json& j, const dynamic_array<T>& d)
    {
      if(d.N == 0 || d.data == nullptr)
      {
        j = nullptr;
      }

      // Assumes a to_json(j, T)
      for(auto& elem : d)
      {
        j.push_back(elem);
      }
    }

    static dynamic_array<T> from_json(const json& j)
    {
      if(j.is_null() || j.size() == 0)
      {
        return {};
      }

      auto d = make_array<T>(j.size());
      std::copy(j.begin(), j.end(), d.begin());
      return d;
    }
  };
}  // namespace nlohmann

// ===========================================================================
// --- G4HepEmParameters
namespace nlohmann
{

  template <>
  struct adl_serializer<G4HepEmRegionParmeters>
  {
    static void to_json(json& j, const G4HepEmRegionParmeters& d)
    {
      j["fFinalRange"]    = d.fFinalRange;
      j["fDRoverRange"]   = d.fDRoverRange;
      j["fLinELossLimit"] = d.fLinELossLimit;

      j["fMSCRangeFactor"]  = d.fMSCRangeFactor;
      j["fMSCSafetyFactor"] = d.fMSCSafetyFactor;

      j["fIsMSCMinimalStepLimit"]     = d.fIsMSCMinimalStepLimit;
      j["fIsELossFluctuation"]        = d.fIsELossFluctuation;
      j["fIsMultipleStepsInMSCTrans"] = d.fIsMultipleStepsInMSCTrans;
    }

    static G4HepEmRegionParmeters from_json(const json& j)
    {
      G4HepEmRegionParmeters d;

      j.at("fFinalRange").get_to(d.fFinalRange);
      j.at("fDRoverRange").get_to(d.fDRoverRange);
      j.at("fLinELossLimit").get_to(d.fLinELossLimit);

      j.at("fMSCRangeFactor").get_to(d.fMSCRangeFactor);
      j.at("fMSCSafetyFactor").get_to(d.fMSCSafetyFactor);

      j.at("fIsMSCMinimalStepLimit").get_to(d.fIsMSCMinimalStepLimit);
      j.at("fIsELossFluctuation").get_to(d.fIsELossFluctuation);
      j.at("fIsMultipleStepsInMSCTrans").get_to(d.fIsMultipleStepsInMSCTrans);

      return d;
    }
  };

  // We *can* have direct to/from_json functions for G4HepEmParameters
  // as it is simple. Use of adl_serializer is *purely* for consistency
  // with other structures!
  // We only support pointers as that's the form G4HepEmData expects
  template <>
  struct adl_serializer<G4HepEmParameters*>
  {
    static void to_json(json& j, const G4HepEmParameters* d)
    {
      if(d == nullptr)
      {
        j = nullptr;
      }
      else
      {
        j["fElectronTrackingCut"]  = d->fElectronTrackingCut;
        j["fMinLossTableEnergy"]   = d->fMinLossTableEnergy;
        j["fMaxLossTableEnergy"]   = d->fMaxLossTableEnergy;
        j["fNumLossTableBins"]     = d->fNumLossTableBins;
        j["fElectronBremModelLim"] = d->fElectronBremModelLim;
        j["fNumRegions"]           = d->fNumRegions;
        j["fParametersPerRegion"]  =
          make_span(d->fNumRegions, d->fParametersPerRegion);
      }
    }

    static G4HepEmParameters* from_json(const json& j)
    {
      if(j.is_null())
      {
        return nullptr;
      }
      else
      {
        auto* d = new G4HepEmParameters;

        d->fElectronTrackingCut  = j.at("fElectronTrackingCut").get<double>();
        d->fMinLossTableEnergy   = j.at("fMinLossTableEnergy").get<double>();
        d->fMaxLossTableEnergy   = j.at("fMaxLossTableEnergy").get<double>();
        d->fNumLossTableBins     = j.at("fNumLossTableBins").get<int>();
        d->fElectronBremModelLim = j.at("fElectronBremModelLim").get<double>();
        d->fNumRegions           = j.at("fNumRegions").get<int>();

        d->fParametersPerRegion  = new G4HepEmRegionParmeters[d->fNumRegions];
        auto tmpParPerRegion = j.at("fParametersPerRegion");
        std::copy(tmpParPerRegion.begin(), tmpParPerRegion.end(), d->fParametersPerRegion);

        return d;
      }
    }
  };
}  // namespace nlohmann

// ===========================================================================
// --- G4HepEmElementData
// Helpers
G4HepEmElemData* begin(const G4HepEmElementData& d) { return d.fElementData; }

G4HepEmElemData* end(const G4HepEmElementData& d)
{
  // Element array is indexed by Z, so has one extra element (Z=0)
  return d.fElementData + d.fMaxZet + 1;
}

namespace nlohmann
{
  // We *can* have direct to/from_json functions for G4HepEmElemData
  // as it is simple. Use of adl_serializer is *purely* for consistency
  // with other structures!
  template <>
  struct adl_serializer<G4HepEmElemData>
  {
    // JSON
    static void to_json(json& j, const G4HepEmElemData& d)
    {
      j["fZet"]          = d.fZet;
      j["fZet13"]        = d.fZet13;
      j["fZet23"]        = d.fZet23;
      j["fCoulomb"]      = d.fCoulomb;
      j["fLogZ"]         = d.fLogZ;
      j["fZFactor1"]     = d.fZFactor1;
      j["fDeltaMaxLow"]  = d.fDeltaMaxLow;
      j["fDeltaMaxHigh"] = d.fDeltaMaxHigh;
      j["fILVarS1"]      = d.fILVarS1;
      j["fILVarS1Cond"]  = d.fILVarS1Cond;
      j["fSandiaEnergies"] =
        make_span(d.fNumOfSandiaIntervals, d.fSandiaEnergies);
      j["fSandiaCoefficients"] =
        make_span(4 * d.fNumOfSandiaIntervals, d.fSandiaCoefficients);
      j["fKShellBindingEnergy"] = d.fKShellBindingEnergy;
    }

    static G4HepEmElemData from_json(const json& j)
    {
      G4HepEmElemData d;

      j.at("fZet").get_to(d.fZet);
      j.at("fZet13").get_to(d.fZet13);
      j.at("fZet23").get_to(d.fZet23);
      j.at("fCoulomb").get_to(d.fCoulomb);
      j.at("fLogZ").get_to(d.fLogZ);
      j.at("fZFactor1").get_to(d.fZFactor1);
      j.at("fDeltaMaxLow").get_to(d.fDeltaMaxLow);
      j.at("fDeltaMaxHigh").get_to(d.fDeltaMaxHigh);
      j.at("fILVarS1").get_to(d.fILVarS1);
      j.at("fILVarS1Cond").get_to(d.fILVarS1Cond);

      auto tmpSandiaEnergies =
        j.at("fSandiaEnergies").get<dynamic_array<double>>();
      d.fNumOfSandiaIntervals = tmpSandiaEnergies.N;
      d.fSandiaEnergies       = tmpSandiaEnergies.data;

      auto tmpSandiaCoefficients =
        j.at("fSandiaCoefficients").get<dynamic_array<double>>();
      d.fSandiaCoefficients = tmpSandiaCoefficients.data;

      j.at("fKShellBindingEnergy").get_to(d.fKShellBindingEnergy);

      return d;
    }
  };

  template <>
  struct adl_serializer<G4HepEmElementData*>
  {
    static void to_json(json& j, const G4HepEmElementData* d)
    {
      if(d == nullptr)
      {
        j = nullptr;
      }
      else
      {
        // G4HepEmElementData stores *all* elements in memory, but
        // only those with fZet +ve are used in this setup so we just persist
        // those
        for(auto& elem : *d)
        {
          if(elem.fZet > 0.0)
          {
            j.push_back(elem);
          }
        }
      }
    }

    static G4HepEmElementData* from_json(const json& j)
    {
      if(j.is_null())
      {
        return nullptr;
      }
      else
      {
        G4HepEmElementData* p = nullptr;
        AllocateElementData(&p);
        if(j.size() > p->fMaxZet + 1)
        {
          FreeElementData(&p);
          throw std::runtime_error(
            "size of JSON array larger than G4HepEmElementData array");
        }
        // Read in by element and assign to the right index
        for(const auto& e : j)
        {
          auto tmpElem       = e.get<G4HepEmElemData>();
          int i              = static_cast<int>(tmpElem.fZet);
          p->fElementData[i] = tmpElem;
        }
        return p;
      }
    }
  };
}  // namespace nlohmann

// --- G4HepEmMaterialData
namespace nlohmann
{
  // We *can* have direct to/from_json functions for G4HepEmMatData
  // as it is simple. Use of adl_serializer is *purely* for consistency
  // with other structures! (Though it also helps in construction of
  // the dynamic arrays G4HepEmMatData holds on from_json)
  template <>
  struct adl_serializer<G4HepEmMatData>
  {
    static void to_json(json& j, const G4HepEmMatData& d)
    {
      j["fG4MatIndex"]  = d.fG4MatIndex;
      j["fElementVect"] = make_span(d.fNumOfElement, d.fElementVect);
      j["fNumOfAtomsPerVolumeVect"] =
        make_span(d.fNumOfElement, d.fNumOfAtomsPerVolumeVect);
      j["fDensity"]          = d.fDensity;
      j["fDensityCorfactor"] = d.fDensityCorFactor;
      j["fElectronDensity"]  = d.fElectronDensity;
      j["fRadiationLength"]  = d.fRadiationLength;
      j["fMeanExEnergy"]     = d.fMeanExEnergy;
      j["fSandiaEnergies"] =
        make_span(d.fNumOfSandiaIntervals, d.fSandiaEnergies);
      j["fSandiaCoefficients"] =
        make_span(4 * d.fNumOfSandiaIntervals, d.fSandiaCoefficients);

      j["fZeff"]      = d.fZeff;
      j["fZeff23"]    = d.fZeff23;
      j["fZeffSqrt"]  = d.fZeffSqrt;

      j["fUMSCPar"]         = d.fUMSCPar;
      j["fUMSCStepMinPars"] = d.fUMSCStepMinPars;
      j["fUMSCTailCoeff"]   = d.fUMSCTailCoeff;
      j["fUMSCThetaCoeff"]  = d.fUMSCThetaCoeff;

    }

    static G4HepEmMatData from_json(const json& j)
    {
      G4HepEmMatData d;

      j.at("fG4MatIndex").get_to(d.fG4MatIndex);
      auto tmpElemVect = j.at("fElementVect").get<dynamic_array<int>>();
      d.fNumOfElement  = tmpElemVect.N;
      d.fElementVect   = tmpElemVect.data;

      auto tmpAtomsPerVolumeVect =
        j.at("fNumOfAtomsPerVolumeVect").get<dynamic_array<double>>();
      d.fNumOfAtomsPerVolumeVect = tmpAtomsPerVolumeVect.data;

      j.at("fDensity").get_to(d.fDensity);
      j.at("fDensityCorfactor").get_to(d.fDensityCorFactor);
      j.at("fElectronDensity").get_to(d.fElectronDensity);
      j.at("fRadiationLength").get_to(d.fRadiationLength);
      j.at("fMeanExEnergy").get_to(d.fMeanExEnergy);

      auto tmpSandiaEnergies =
        j.at("fSandiaEnergies").get<dynamic_array<double>>();
      d.fNumOfSandiaIntervals = tmpSandiaEnergies.N;
      d.fSandiaEnergies       = tmpSandiaEnergies.data;

      auto tmpSandiaCoefficients =
        j.at("fSandiaCoefficients").get<dynamic_array<double>>();
      d.fSandiaCoefficients = tmpSandiaCoefficients.data;

      j.at("fZeff").get_to(d.fZeff);
      j.at("fZeff23").get_to(d.fZeff23);
      j.at("fZeffSqrt").get_to(d.fZeffSqrt);

      j.at("fUMSCPar").get_to(d.fUMSCPar);
      j.at("fUMSCStepMinPars").get_to(d.fUMSCStepMinPars);
      j.at("fUMSCTailCoeff").get_to(d.fUMSCTailCoeff);
      j.at("fUMSCThetaCoeff").get_to(d.fUMSCThetaCoeff);

      return d;
    }
  };

  template <>
  struct adl_serializer<G4HepEmMaterialData*>
  {
    static void to_json(json& j, const G4HepEmMaterialData* d)
    {
      if(d == nullptr)
      {
        j = nullptr;
      }
      else
      {
        j["fNumG4Material"]   = d->fNumG4Material;
        j["fNumMaterialData"] = d->fNumMaterialData;
        j["fG4MatIndexToHepEmMatIndex"] =
          make_span(d->fNumG4Material, d->fG4MatIndexToHepEmMatIndex);
        j["fMaterialData"] = make_span(d->fNumMaterialData, d->fMaterialData);
      }
    }

    static G4HepEmMaterialData* from_json(const json& j)
    {
      if(j.is_null())
      {
        return nullptr;
      }
      else
      {
        auto tmpNumG4Mat   = j.at("fNumG4Material").get<int>();
        auto tmpNumMatData = j.at("fNumMaterialData").get<int>();

        // Allocate data with enough memory to hold the read in data
        G4HepEmMaterialData* d = nullptr;
        AllocateMaterialData(&d, tmpNumG4Mat, tmpNumMatData);

        auto tmpMatIndexVect = j.at("fG4MatIndexToHepEmMatIndex");
        std::copy(tmpMatIndexVect.begin(), tmpMatIndexVect.end(),
                  d->fG4MatIndexToHepEmMatIndex);
        auto tmpMatData = j.at("fMaterialData");
        std::copy(tmpMatData.begin(), tmpMatData.end(), d->fMaterialData);

        return d;
      }
    }
  };
}  // namespace nlohmann

// --- G4HepEmMatCutData
namespace nlohmann
{
  template <>
  struct adl_serializer<G4HepEmMCCData>
  {
    static void to_json(json& j, const G4HepEmMCCData& d)
    {
      j["fSecElProdCutE"]  = d.fSecElProdCutE;
      j["fSecPosProdCutE"] = d.fSecPosProdCutE;
      j["fSecGamProdCutE"] = d.fSecGamProdCutE;
      j["fLogSecGamCutE"]  = d.fLogSecGamCutE;
      j["fHepEmMatIndex"]  = d.fHepEmMatIndex;
      j["fG4MatCutIndex"]  = d.fG4MatCutIndex;
      j["fG4RegionIndex"]  = d.fG4RegionIndex;
    }

    static G4HepEmMCCData from_json(const json& j)
    {
      G4HepEmMCCData d;

      j.at("fSecElProdCutE").get_to(d.fSecElProdCutE);
      j.at("fSecPosProdCutE").get_to(d.fSecPosProdCutE);
      j.at("fSecGamProdCutE").get_to(d.fSecGamProdCutE);
      j.at("fLogSecGamCutE").get_to(d.fLogSecGamCutE);
      j.at("fHepEmMatIndex").get_to(d.fHepEmMatIndex);
      j.at("fG4MatCutIndex").get_to(d.fG4MatCutIndex);
      j.at("fG4RegionIndex").get_to(d.fG4RegionIndex);

      return d;
    }
  };

  template <>
  struct adl_serializer<G4HepEmMatCutData*>
  {
    static void to_json(json& j, const G4HepEmMatCutData* d)
    {
      if(d == nullptr)
      {
        j = nullptr;
      }
      else
      {
        j["fNumG4MatCuts"]  = d->fNumG4MatCuts;
        j["fNumMatCutData"] = d->fNumMatCutData;
        j["fG4MCIndexToHepEmMCIndex"] =
          make_span(d->fNumG4MatCuts, d->fG4MCIndexToHepEmMCIndex);
        j["fMatCutData"] = make_span(d->fNumMatCutData, d->fMatCutData);
      }
    }

    static G4HepEmMatCutData* from_json(const json& j)
    {
      if(j.is_null())
      {
        return nullptr;
      }
      else
      {
        auto tmpNumG4Cuts  = j.at("fNumG4MatCuts").get<int>();
        auto tmpNumMatCuts = j.at("fNumMatCutData").get<int>();

        // Allocate the new object using this info
        G4HepEmMatCutData* d = nullptr;
        AllocateMatCutData(&d, tmpNumG4Cuts, tmpNumMatCuts);

        auto tmpG4CutIndex = j.at("fG4MCIndexToHepEmMCIndex");
        std::copy(tmpG4CutIndex.begin(), tmpG4CutIndex.end(),
                  d->fG4MCIndexToHepEmMCIndex);

        auto tmpMCData = j.at("fMatCutData");
        std::copy(tmpMCData.begin(), tmpMCData.end(), d->fMatCutData);

        return d;
      }
    }
  };
}  // namespace nlohmann

// --- G4HepEmElectronData
namespace nlohmann
{
  template <>
  struct adl_serializer<G4HepEmElectronData*>
  {
    static void to_json(json& j, G4HepEmElectronData* d)
    {
      if(d == nullptr)
      {
        j = nullptr;
      }
      else
      {
        j["fNumMatCuts"]      = d->fNumMatCuts;
        j["fNumMaterials"]    = d->fNumMaterials;
        j["fELossLogMinEkin"] = d->fELossLogMinEkin;
        j["fELossEILDelta"]   = d->fELossEILDelta;

        j["fELossEnergyGrid"] =
          make_span(d->fELossEnergyGridSize, d->fELossEnergyGrid);

        const int nELoss = 5 * (d->fELossEnergyGridSize) * (d->fNumMatCuts);
        j["fELossData"]  = make_span(nELoss, d->fELossData);

        j["fResMacXSecStartIndexPerMatCut"] =
          make_span(d->fNumMatCuts, d->fResMacXSecStartIndexPerMatCut);
        j["fResMacXSecData"] =
          make_span(d->fResMacXSecNumData, d->fResMacXSecData);

        j["fENucLogMinEkin"] = d->fENucLogMinEkin;
        j["fENucEILDelta"]   = d->fENucEILDelta;

        j["fENucEnergyGrid"] =
          make_span(d->fENucEnergyGridSize, d->fENucEnergyGrid);

        const int nENuc = 2 * (d->fENucEnergyGridSize) * (d->fNumMaterials);
        j["fENucMacXsecData"]  = make_span(nENuc, d->fENucMacXsecData);

        const int nTr1MacXsec = 2 * (d->fELossEnergyGridSize) * (d->fNumMaterials);
        j["fTr1MacXSecData"] =
          make_span(nTr1MacXsec, d->fTr1MacXSecData);

        j["fElemSelectorIoniStartIndexPerMatCut"] =
          make_span(d->fNumMatCuts, d->fElemSelectorIoniStartIndexPerMatCut);
        j["fElemSelectorIoniData"] =
          make_span(d->fElemSelectorIoniNumData, d->fElemSelectorIoniData);

        j["fElemSelectorBremSBStartIndexPerMatCut"] =
          make_span(d->fNumMatCuts, d->fElemSelectorBremSBStartIndexPerMatCut);
        j["fElemSelectorBremSBData"] =
          make_span(d->fElemSelectorBremSBNumData, d->fElemSelectorBremSBData);

        j["fElemSelectorBremRBStartIndexPerMatCut"] =
          make_span(d->fNumMatCuts, d->fElemSelectorBremRBStartIndexPerMatCut);
        j["fElemSelectorBremRBData"] =
          make_span(d->fElemSelectorBremRBNumData, d->fElemSelectorBremRBData);
      }
    }

    static G4HepEmElectronData* from_json(const json& j)
    {
      if(j.is_null())
      {
        return nullptr;
      }
      else
      {
        G4HepEmElectronData* d = nullptr;
        AllocateElectronData(&d);

        j.at("fNumMatCuts").get_to(d->fNumMatCuts);
        j.at("fNumMaterials").get_to(d->fNumMaterials);
        j.at("fELossLogMinEkin").get_to(d->fELossLogMinEkin);
        j.at("fELossEILDelta").get_to(d->fELossEILDelta);

        auto tmpElossGrid =
          j.at("fELossEnergyGrid").get<dynamic_array<double>>();
        d->fELossEnergyGridSize = tmpElossGrid.N;
        d->fELossEnergyGrid     = tmpElossGrid.data;

        auto tmpELossData = j.at("fELossData").get<dynamic_array<double>>();
        d->fELossData     = tmpELossData.data;
        // To validate, tmpELossData == 5 * (d->fELossEnergyGridSize) *
        // (d->fNumMatCuts);
        {
          auto tmpIndex =
            j.at("fResMacXSecStartIndexPerMatCut").get<dynamic_array<int>>();
          d->fResMacXSecStartIndexPerMatCut = tmpIndex.data;
          // To validate, tmpIndex.N == d->fNumMatCuts;

          j.at("fENucLogMinEkin").get_to(d->fENucLogMinEkin);
          j.at("fENucEILDelta").get_to(d->fENucEILDelta);

          // Get the array but ignore the size (fENucEnergyGridSize) as this is a
          // const (at time of writing)
          auto tmpENucGrid =
            j.at("fENucEnergyGrid").get<dynamic_array<double>>();
          d->fENucEnergyGrid     = tmpENucGrid.data;

          auto tmpENucData = j.at("fENucMacXsecData").get<dynamic_array<double>>();
          d->fENucMacXsecData    = tmpENucData.data;

          auto tmpData = j.at("fResMacXSecData").get<dynamic_array<double>>();
          d->fResMacXSecNumData = tmpData.N;
          d->fResMacXSecData    = tmpData.data;

          auto tmpTr1Data = j.at("fTr1MacXSecData").get<dynamic_array<double>>();
          d->fTr1MacXSecData    = tmpTr1Data.data;
        }

        {
          auto tmpIndex = j.at("fElemSelectorIoniStartIndexPerMatCut")
                            .get<dynamic_array<int>>();
          d->fElemSelectorIoniStartIndexPerMatCut = tmpIndex.data;
          // To validate, tmpIndex.N == d->fNumMatCuts;

          auto tmpData =
            j.at("fElemSelectorIoniData").get<dynamic_array<double>>();
          d->fElemSelectorIoniNumData = tmpData.N;
          d->fElemSelectorIoniData    = tmpData.data;
        }

        {
          auto tmpIndex = j.at("fElemSelectorBremSBStartIndexPerMatCut")
                            .get<dynamic_array<int>>();
          d->fElemSelectorBremSBStartIndexPerMatCut = tmpIndex.data;
          // To validate, tmpIndex.N == d->fNumMatCuts;

          auto tmpData =
            j.at("fElemSelectorBremSBData").get<dynamic_array<double>>();
          d->fElemSelectorBremSBNumData = tmpData.N;
          d->fElemSelectorBremSBData    = tmpData.data;
        }

        {
          auto tmpIndex = j.at("fElemSelectorBremRBStartIndexPerMatCut")
                            .get<dynamic_array<int>>();
          d->fElemSelectorBremRBStartIndexPerMatCut = tmpIndex.data;
          // To validate, tmpIndex.N == d->fNumMatCuts;

          auto tmpData =
            j.at("fElemSelectorBremRBData").get<dynamic_array<double>>();
          d->fElemSelectorBremRBNumData = tmpData.N;
          d->fElemSelectorBremRBData    = tmpData.data;
        }

        return d;
      }
    }
  };
}  // namespace nlohmann

// --- G4HepEmSBTableData
namespace nlohmann
{
  template <>
  struct adl_serializer<G4HepEmSBTableData*>
  {
    static void to_json(json& j, const G4HepEmSBTableData* d)
    {
      if(d == nullptr)
      {
        j = nullptr;
      }
      else
      {
        j["fLogMinElEnergy"]  = d->fLogMinElEnergy;
        j["fILDeltaElEnergy"] = d->fILDeltaElEnergy;
        j["fElEnergyVect"]    = d->fElEnergyVect;
        j["fLElEnergyVect"]   = d->fLElEnergyVect;
        j["fKappaVect"]       = d->fKappaVect;
        j["fLKappaVect"]      = d->fLKappaVect;

        j["fGammaCutIndxStartIndexPerMC"] =
          make_span(d->fNumHepEmMatCuts, d->fGammaCutIndxStartIndexPerMC);

        j["fGammaCutIndices"] =
          make_span(d->fNumElemsInMatCuts, d->fGammaCutIndices);

        j["fSBStartTablesStartPerZ"] = d->fSBTablesStartPerZ;
        j["fSBTableData"] = make_span(d->fNumSBTableData, d->fSBTableData);
      }
    }

    static G4HepEmSBTableData* from_json(const json& j)
    {
      if(j.is_null())
      {
        return nullptr;
      }
      else
      {
        G4HepEmSBTableData* d = nullptr;

        // Reading arrays first so we can allocate/copy directly
        // fNumHepEmMatCuts
        auto tmpGammaCutStartIndices = j.at("fGammaCutIndxStartIndexPerMC");
        // fNumElemsInMatCuts
        auto tmpGammaCutIndices = j.at("fGammaCutIndices");
        // fNumSBTableData
        auto tmpSBTableData = j.at("fSBTableData");

        AllocateSBTableData(&d, tmpGammaCutStartIndices.size(),
                            tmpGammaCutIndices.size(), tmpSBTableData.size());

        // copy JSON arrays to newly allocated SB arrays
        std::copy(tmpGammaCutStartIndices.begin(),
                  tmpGammaCutStartIndices.end(),
                  d->fGammaCutIndxStartIndexPerMC);
        std::copy(tmpGammaCutIndices.begin(), tmpGammaCutIndices.end(),
                  d->fGammaCutIndices);
        std::copy(tmpSBTableData.begin(), tmpSBTableData.end(),
                  d->fSBTableData);

        // Now remaining data
        j.at("fLogMinElEnergy").get_to(d->fLogMinElEnergy);
        j.at("fILDeltaElEnergy").get_to(d->fILDeltaElEnergy);
        j.at("fElEnergyVect").get_to(d->fElEnergyVect);
        j.at("fLElEnergyVect").get_to(d->fLElEnergyVect);
        j.at("fKappaVect").get_to(d->fKappaVect);
        j.at("fLKappaVect").get_to(d->fLKappaVect);

        j.at("fSBStartTablesStartPerZ").get_to(d->fSBTablesStartPerZ);

        return d;
      }
    }
  };
}  // namespace nlohmann

// --- G4HepEmGammaData
namespace nlohmann
{
  template <>
  struct adl_serializer<G4HepEmGammaData*>
  {
    static void to_json(json& j, const G4HepEmGammaData* d)
    {
      if(d == nullptr)
      {
        j = nullptr;
      }
      else
      {
        /** Number of G4HepEm materials: number of G4HepEmMatData structures
         * stored in the G4HepEmMaterialData::fMaterialData array. */
        j["fNumMaterials"] = d->fNumMaterials;

        //// === Macroscopic cross section related data:
        j["fDataPerMat"] = d->fDataPerMat;
        j["fNumData0"] = d->fNumData0;
        j["fNumData1"] = d->fNumData1;

        j["fEMin0"] = d->fEMin0;
        j["fEMax0"] = d->fEMax0;
        j["fLogEMin0"] = d->fLogEMin0;
        j["fEILDelta0"] = d->fEILDelta0;

        j["fEMax1"] = d->fEMax1;
        j["fLogEMin1"] = d->fLogEMin1;
        j["fEILDelta1"] = d->fEILDelta1;

        j["fEMax2"] = d->fEMax2;
        j["fLogEMin2"] = d->fLogEMin2;
        j["fEILDelta2"] = d->fEILDelta2;

        const int macXsecDataSize = d->fNumMaterials*d->fDataPerMat;
        j["fMacXsecData"] =make_span(macXsecDataSize, d->fMacXsecData);


        //// === element selector for conversion (note: KN compton interaction
        /// do not know anything about Z)
        j["fElemSelectorConvLogMinEkin"] = d->fElemSelectorConvLogMinEkin;
        j["fElemSelectorConvEILDelta"]   = d->fElemSelectorConvEILDelta;
        j["fElemSelectorConvStartIndexPerMat"] =
          make_span(d->fNumMaterials, d->fElemSelectorConvStartIndexPerMat);

        j["fElemSelectorConvEgrid"] =
          make_span(d->fElemSelectorConvEgridSize, d->fElemSelectorConvEgrid);

        j["fElemSelectorConvData"] =
          make_span(d->fElemSelectorConvNumData, d->fElemSelectorConvData);
      }
    }

    static G4HepEmGammaData* from_json(const json& j)
    {
      if(j.is_null())
      {
        return nullptr;
      }
      else
      {
        G4HepEmGammaData* d = nullptr;
        AllocateGammaData(&d);

        j.at("fNumMaterials").get_to(d->fNumMaterials);


        j.at("fDataPerMat").get_to(d->fDataPerMat);
        j.at("fNumData0").get_to(d->fNumData0);
        j.at("fNumData1").get_to(d->fNumData1);

        j.at("fEMin0").get_to(d->fEMin0 );
        j.at("fEMax0").get_to(d->fEMax0);
        j.at("fLogEMin0").get_to(d->fLogEMin0);
        j.at("fEILDelta0").get_to(d->fEILDelta0);

        j.at("fEMax1").get_to(d->fEMax1);
        j.at("fLogEMin1").get_to(d->fLogEMin1);
        j.at("fEILDelta1").get_to(d->fEILDelta1);

        j.at("fEMax2").get_to(d->fEMax2);
        j.at("fLogEMin2").get_to(d->fLogEMin2);
        j.at("fEILDelta2").get_to(d->fEILDelta2);

        // We don't store the size of the following array, rather should
        // validate that it is expected size: d->fNumMaterials * d->fDataPerMat
        auto tmpMacXsecData = j.at("fMacXsecData").get<dynamic_array<double>>();
        d->fMacXsecData = tmpMacXsecData.data;


        j.at("fElemSelectorConvLogMinEkin")
          .get_to(d->fElemSelectorConvLogMinEkin);
        j.at("fElemSelectorConvEILDelta").get_to(d->fElemSelectorConvEILDelta);

        // size of this array is d->fNumMaterial, which we store separately for
        // now
        auto tmpConvStartIndexPerMat =
          j.at("fElemSelectorConvStartIndexPerMat").get<dynamic_array<int>>();
        d->fElemSelectorConvStartIndexPerMat = tmpConvStartIndexPerMat.data;

        auto tmpConvEgrid =
          j.at("fElemSelectorConvEgrid").get<dynamic_array<double>>();
        d->fElemSelectorConvEgridSize = tmpConvEgrid.N;
        d->fElemSelectorConvEgrid     = tmpConvEgrid.data;

        auto tmpConvData =
          j.at("fElemSelectorConvData").get<dynamic_array<double>>();
        d->fElemSelectorConvNumData = tmpConvData.N;
        d->fElemSelectorConvData    = tmpConvData.data;

        return d;
      }
    }
  };
}  // namespace nlohmann

// --- G4HepEmData
namespace nlohmann
{
  template <>
  struct adl_serializer<G4HepEmData*>
  {
    static void to_json(json& j, const G4HepEmData* d)
    {
      if(d == nullptr)
      {
        j = nullptr;
      }
      else
      {
        j["fTheMatCutData"]   = d->fTheMatCutData;
        j["fTheMaterialData"] = d->fTheMaterialData;
        j["fTheElementData"]  = d->fTheElementData;
        j["fTheElectronData"] = d->fTheElectronData;
        j["fThePositronData"] = d->fThePositronData;
        j["fTheSBTableData"]  = d->fTheSBTableData;
        j["fTheGammaData"]    = d->fTheGammaData;
      }
    }

    static G4HepEmData* from_json(const json& j)
    {
      if(j.is_null())
      {
        return nullptr;
      }
      else
      {
        G4HepEmData* d    = new G4HepEmData;
        d->fTheMatCutData = j.at("fTheMatCutData").get<G4HepEmMatCutData*>();
        d->fTheMaterialData =
          j.at("fTheMaterialData").get<G4HepEmMaterialData*>();
        d->fTheElementData = j.at("fTheElementData").get<G4HepEmElementData*>();
        d->fTheElectronData =
          j.at("fTheElectronData").get<G4HepEmElectronData*>();
        d->fThePositronData =
          j.at("fThePositronData").get<G4HepEmElectronData*>();
        d->fTheSBTableData = j.at("fTheSBTableData").get<G4HepEmSBTableData*>();
        d->fTheGammaData   = j.at("fTheGammaData").get<G4HepEmGammaData*>();
        return d;
      }
    }
  };
}  // namespace nlohmann

// --- G4HepEmState
namespace nlohmann
{
  template <>
  struct adl_serializer<G4HepEmState*>
  {
    static void to_json(json& j, const G4HepEmState* d)
    {
      if(d == nullptr)
      {
        j = nullptr;
      }
      else
      {
        j["fParameters"] = d->fParameters;
        j["fData"]       = d->fData;
      }
    }

    static G4HepEmState* from_json(const json& j)
    {
      if(j.is_null())
      {
        return nullptr;
      }
      else
      {
        G4HepEmState* d = new G4HepEmState;
        d->fParameters  = j.at("fParameters").get<G4HepEmParameters*>();
        d->fData        = j.at("fData").get<G4HepEmData*>();
        return d;
      }
    }
  };
}  // namespace nlohmann

#endif  // G4HepEmJsonSerialization_H
