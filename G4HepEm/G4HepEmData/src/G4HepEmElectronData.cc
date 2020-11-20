
#include "G4HepEmElectronData.hh"

#include <iostream>

// NOTE: allocates only the main data structure but not the dynamic members
void AllocateElectronData (struct G4HepEmElectronData** theElectronData) {
  // clean away previous (if any)
  FreeElectronData(theElectronData);
  *theElectronData   = new G4HepEmElectronData;
  // eloss
  (*theElectronData)->fELossEnergyGrid                       = nullptr;
  (*theElectronData)->fELossData                             = nullptr;
  // mac-xsec
  (*theElectronData)->fResMacXSecStartIndexPerMatCut         = nullptr;
  (*theElectronData)->fResMacXSecData                        = nullptr;
  // elemen selectors per models:
  // - Moller-Bhabha ionisation
  (*theElectronData)->fElemSelectorIoniStartIndexPerMatCut   = nullptr;
  (*theElectronData)->fElemSelectorIoniData                  = nullptr;
  // - Seltzer-Berger model for e-/e+ bremsstrahlung
  (*theElectronData)->fElemSelectorBremSBStartIndexPerMatCut = nullptr;
  (*theElectronData)->fElemSelectorBremSBData                = nullptr;
  // - relativistic (improved Bethe-Heitler) model for e-/e+ bremsstrahlung  
  (*theElectronData)->fElemSelectorBremRBStartIndexPerMatCut = nullptr;
  (*theElectronData)->fElemSelectorBremRBData                = nullptr;

}


void FreeElectronData (struct G4HepEmElectronData** theElectronData)  {
  if (*theElectronData) {
    // eloss
    if ((*theElectronData)->fELossEnergyGrid) {
      delete[] (*theElectronData)->fELossEnergyGrid;
    }
    if ((*theElectronData)->fELossData) {
      delete[] (*theElectronData)->fELossData;
    }
    // mac-xsec
    if ((*theElectronData)->fResMacXSecData) {
      delete[] (*theElectronData)->fResMacXSecData;
    }
    if ((*theElectronData)->fResMacXSecStartIndexPerMatCut) {
      delete[] (*theElectronData)->fResMacXSecStartIndexPerMatCut;
    }
    // element selectors:
    if ((*theElectronData)->fElemSelectorIoniStartIndexPerMatCut) {
      delete[] (*theElectronData)->fElemSelectorIoniStartIndexPerMatCut;
    }
    if ((*theElectronData)->fElemSelectorIoniData) {
      delete[] (*theElectronData)->fElemSelectorIoniData;
    }
    if ((*theElectronData)->fElemSelectorBremSBStartIndexPerMatCut) {
      delete[] (*theElectronData)->fElemSelectorBremSBStartIndexPerMatCut;
    }
    if ((*theElectronData)->fElemSelectorBremSBData) {
      delete[] (*theElectronData)->fElemSelectorBremSBData;
    }
    if ((*theElectronData)->fElemSelectorBremRBStartIndexPerMatCut) {
      delete[] (*theElectronData)->fElemSelectorBremRBStartIndexPerMatCut;
    }
    if ((*theElectronData)->fElemSelectorBremRBData) {
      delete[] (*theElectronData)->fElemSelectorBremRBData;
    }
        
    delete *theElectronData;
    *theElectronData = nullptr;
  }
}