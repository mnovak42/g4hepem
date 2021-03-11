#include "G4HepEmElectronData.hh"
#include "G4HepEmElementData.hh"
#include "G4HepEmGammaData.hh"
#include "G4HepEmMatCutData.hh"
#include "G4HepEmMaterialData.hh"
#include "G4HepEmSBTableData.hh"

#include "gtest/gtest.h"

// Simple check of a size-dynamic array pair
// Returns true if:
// - size is 0, pointer is nullptr
// - size is >0, pointer is not null
// Thus only confirms that a pointer exists if size is > 0
template <typename T>
bool valid_array(int size, T* data) {
  if(size == 0 && data == nullptr) {
    return true;
  }
  return (size > 0 && data != nullptr);
}

// --- G4HepEmElectronData
void G4HepEmElectronDataTester(G4HepEmElectronData* d) {
  // Construction/Allocation interfaces only ever default
  // so we just test that all dynamic memory is null
  ASSERT_NE(d, nullptr);

  EXPECT_EQ(d->fNumMatCuts, 0);
  EXPECT_EQ(d->fELossEnergyGridSize, 0);
  EXPECT_EQ(d->fELossEnergyGrid, nullptr);
  EXPECT_EQ(d->fELossData, nullptr);

  EXPECT_EQ(d->fResMacXSecNumData, 0);
  EXPECT_EQ(d->fResMacXSecStartIndexPerMatCut, nullptr);
  EXPECT_EQ(d->fResMacXSecData, nullptr);

  EXPECT_EQ(d->fElemSelectorIoniNumData, 0);
  EXPECT_EQ(d->fElemSelectorIoniStartIndexPerMatCut, nullptr);
  EXPECT_EQ(d->fElemSelectorIoniData, nullptr);

  EXPECT_EQ(d->fElemSelectorBremSBNumData, 0);
  EXPECT_EQ(d->fElemSelectorBremSBStartIndexPerMatCut, nullptr);
  EXPECT_EQ(d->fElemSelectorBremSBData, nullptr);

  EXPECT_EQ(d->fElemSelectorBremRBNumData, 0);
  EXPECT_EQ(d->fElemSelectorBremRBStartIndexPerMatCut, nullptr);
  EXPECT_EQ(d->fElemSelectorBremRBData, nullptr);
}

TEST(G4HepEmElectronData, DefaultConstruction) {
  G4HepEmElectronData d;
  G4HepEmElectronDataTester(&d);
}

TEST(G4HepEmElectronData, MakerFunction) {
  G4HepEmElectronData* d = MakeElectronData();
  G4HepEmElectronDataTester(d);
  FreeElectronData(&d);
  ASSERT_EQ(d, nullptr);
}

TEST(G4HepEmElectronData, AllocationInterface) {
  G4HepEmElectronData* d = nullptr;
  AllocateElectronData(&d);
  G4HepEmElectronDataTester(d);
  FreeElectronData(&d);
  ASSERT_EQ(d, nullptr);
}

// --- G4HepEmElementData
void G4HepEmElementDataTester(G4HepEmElementData* d) {
  ASSERT_NE(d, nullptr);
  EXPECT_NE(d->fMaxZet, 0);
  ASSERT_NE(d->fElementData, nullptr);

  // Indexing is by Z, so from 1 to fMaxZet, but with
  // fMaxZet+1 elements!
  for(int i = 0; i < d->fMaxZet+1 ; ++i) {
    EXPECT_LT(d->fElementData[i].fZet, 0.0);
  }
}

TEST(G4HepEmElementData, DefaultConstruction) {
  G4HepEmElementData d;
  EXPECT_EQ(d.fMaxZet, 0);
  EXPECT_EQ(d.fElementData, nullptr);
}

TEST(G4HepEmElementData, MakerFunction) {
  G4HepEmElementData* d = MakeElementData();
  G4HepEmElementDataTester(d);
  FreeElementData(&d);
  ASSERT_EQ(d, nullptr);
}

TEST(G4HepEmElementData, AllocationInterface) {
  G4HepEmElementData* d = nullptr;
  AllocateElementData(&d);
  G4HepEmElementDataTester(d);
  FreeElementData(&d);
  ASSERT_EQ(d, nullptr);
}

// --- G4HepEmGammaData
void G4HepEmGammaDataTester(G4HepEmGammaData* d) {
  // Construction/Allocation interfaces only ever default
  // so we just test that all dynamic memory is null
  ASSERT_NE(d, nullptr);

  EXPECT_EQ(d->fNumMaterials, 0);

  // Energy grid has a fixed size, but dynamic allocation
  EXPECT_EQ(d->fConvEnergyGridSize, 147);
  EXPECT_EQ(d->fConvEnergyGrid, nullptr);

  // Energy grid has a fixed size, but dynamic allocation
  EXPECT_EQ(d->fCompEnergyGridSize, 85);
  EXPECT_EQ(d->fCompEnergyGrid, nullptr);
  EXPECT_EQ(d->fConvCompMacXsecData, nullptr);

  EXPECT_EQ(d->fElemSelectorConvEgridSize, 0);
  EXPECT_EQ(d->fElemSelectorConvNumData, 0);
  EXPECT_EQ(d->fElemSelectorConvStartIndexPerMat, nullptr);
  EXPECT_EQ(d->fElemSelectorConvEgrid, nullptr);
  EXPECT_EQ(d->fElemSelectorConvData, nullptr);
}

TEST(G4HepEmGammaData, DefaultConstruction) {
  G4HepEmGammaData d;
  G4HepEmGammaDataTester(&d);
}

TEST(G4HepEmGammaData, MakerFunction) {
  G4HepEmGammaData* d = MakeGammaData();
  G4HepEmGammaDataTester(d);
  FreeGammaData(&d);
  ASSERT_EQ(d, nullptr);
}

TEST(G4HepEmGammaData, AllocationInterface) {
  G4HepEmGammaData* d = nullptr;
  AllocateGammaData(&d);
  G4HepEmGammaDataTester(d);
  FreeGammaData(&d);
  ASSERT_EQ(d, nullptr);
}

// --- G4HepEmMatCutData
void G4HepEmMatCutDataTester(G4HepEmMatCutData* d, int expectedG4Cuts, int expectedUsedCuts) {
  ASSERT_NE(d, nullptr);

  EXPECT_EQ(d->fNumG4MatCuts, expectedG4Cuts);
  EXPECT_PRED2(valid_array<int>, d->fNumG4MatCuts, d->fG4MCIndexToHepEmMCIndex);
  EXPECT_EQ(d->fNumMatCutData, expectedUsedCuts);
  ASSERT_PRED2(valid_array<G4HepEmMCCData>, d->fNumMatCutData, d->fMatCutData);

  for(int i = 0; i < d->fNumMatCutData; ++i) {
    // Each G4HepEmMCCData element must be default constructed
    EXPECT_EQ((d->fMatCutData[i]).fHepEmMatIndex, -1);
    EXPECT_EQ((d->fMatCutData[i]).fG4MatCutIndex, -1);
  }
}

TEST(G4HepEmMatCutData, DefaultConstruction) {
  G4HepEmMatCutData d;
  G4HepEmMatCutDataTester(&d, 0, 0);
}

TEST(G4HepEmMatCutData, MakerFunction) {
  G4HepEmMatCutData* d = MakeMatCutData(3,2);
  G4HepEmMatCutDataTester(d, 3,2);
  FreeMatCutData(&d);
  ASSERT_EQ(d, nullptr);
}

TEST(G4HepEmMatCutData, AllocationInterface) {
  G4HepEmMatCutData* d = nullptr;
  AllocateMatCutData(&d, 42, 24);
  G4HepEmMatCutDataTester(d, 42, 24);
  FreeMatCutData(&d);
  ASSERT_EQ(d, nullptr);
}

// --- G4HepEmMaterialData
void G4HepEmMatDataTester(const G4HepEmMatData& d) {
  EXPECT_EQ(d.fG4MatIndex, -1);
  EXPECT_EQ(d.fNumOfElement, 0);
  EXPECT_EQ(d.fElementVect, nullptr);
  EXPECT_EQ(d.fNumOfAtomsPerVolumeVect, nullptr);
}


void G4HepEmMaterialDataTester(G4HepEmMaterialData* d, int expectedNumMat, int expectedUsedMat) {
  ASSERT_NE(d, nullptr);

  EXPECT_EQ(d->fNumG4Material, expectedNumMat);
  EXPECT_PRED2(valid_array<int>, d->fNumG4Material, d->fG4MatIndexToHepEmMatIndex);
  ASSERT_PRED2(valid_array<G4HepEmMatData>, d->fNumMaterialData, d->fMaterialData);

  for(int i = 0; i < d->fNumMaterialData; ++i) {
    // Each G4HepEmMatData element must be default constructed
    G4HepEmMatDataTester(d->fMaterialData[i]);
  }
}

TEST(G4HepEmMaterialData, DefaultConstruction) {
  G4HepEmMaterialData d;
  G4HepEmMaterialDataTester(&d, 0, 0);
}

TEST(G4HepEmMaterialData, MakerFunction) {
  G4HepEmMaterialData* d = MakeMaterialData(5,3);
  G4HepEmMaterialDataTester(d, 5, 3);
  FreeMaterialData(&d);
  ASSERT_EQ(d, nullptr);
}

TEST(G4HepEmMaterialData, AllocationInterface) {
  G4HepEmMaterialData* d = nullptr;
  AllocateMaterialData(&d, 53, 26);
  G4HepEmMaterialDataTester(d, 53, 26);
  FreeMaterialData(&d);
  ASSERT_EQ(d, nullptr);
}


// --- G4HepEmSBTableData
void G4HepEmSBTableDataTester(G4HepEmSBTableData* d, int expectedNumCuts, int expectedNumElems, int expectedNumSBElems) {
  ASSERT_NE(d, nullptr);

  EXPECT_EQ(sizeof(d->fElEnergyVect)/sizeof(*(d->fElEnergyVect)), d->fNumElEnergy);
  EXPECT_EQ(sizeof(d->fLElEnergyVect)/sizeof(*(d->fLElEnergyVect)), d->fNumElEnergy);
  EXPECT_EQ(sizeof(d->fKappaVect)/sizeof(*(d->fKappaVect)), d->fNumKappa);
  EXPECT_EQ(sizeof(d->fLKappaVect)/sizeof(*(d->fLKappaVect)), d->fNumKappa);

  EXPECT_EQ(d->fNumHepEmMatCuts, expectedNumCuts);
  EXPECT_PRED2(valid_array<int>, d->fNumHepEmMatCuts, d->fGammaCutIndxStartIndexPerMC);

  EXPECT_EQ(d->fNumElemsInMatCuts, expectedNumElems);
  EXPECT_PRED2(valid_array<int>, d->fNumElemsInMatCuts, d->fGammaCutIndices);

  EXPECT_EQ(d->fNumSBTableData, expectedNumSBElems);
  EXPECT_PRED2(valid_array<double>, d->fNumSBTableData, d->fSBTableData);
}

TEST(G4HepEmSBTableData, DefaultConstruction) {
  G4HepEmSBTableData d;
  G4HepEmSBTableDataTester(&d, 0, 0, 0);
}

TEST(G4HepEmSBTableData, MakerFunction) {
  G4HepEmSBTableData* d = MakeSBTableData(5, 3, 5);
  G4HepEmSBTableDataTester(d, 5, 3, 5);
  FreeSBTableData(&d);
  ASSERT_EQ(d, nullptr);
}

TEST(G4HepEmSBTableData, AllocationInterface) {
  G4HepEmSBTableData* d = nullptr;
  AllocateSBTableData(&d, 53, 26, 42);
  G4HepEmSBTableDataTester(d, 53, 26, 42);
  FreeSBTableData(&d);
  ASSERT_EQ(d, nullptr);
}
