//
// M. Novak: it's practically a copy of my `G4GSPWACorrections` (without its run
//           time `GetPWACorrectionFactors` interpolation method) that could be
//           used directly if its `fDataPerMaterial` `DataPerMaterial` collection
//           would be publicly available.

#ifndef G4HepEmGSPWACorTableBuilder_HH
#define G4HepEmGSPWACorTableBuilder_HH

#include <CLHEP/Units/SystemOfUnits.h>

#include "globals.hh"

#include <vector>
#include <string>
#include <sstream>

class G4Material;
class G4Element;


class G4HepEmGSPWACorTableBuilder {
public:
  G4HepEmGSPWACorTableBuilder(G4bool iselectron=true);

 ~G4HepEmGSPWACorTableBuilder();

  void     Initialise();

  // either per material or per Z
  struct DataPerMaterial {
    std::vector<G4double>   fCorScreening;    // correction factor to Moliere screening parameter
    std::vector<G4double>   fCorFirstMoment;  // correction factor to first moment
    std::vector<G4double>   fCorSecondMoment; // correction factor to second
  };

  const DataPerMaterial* GetPWACorrectionDataPerMaterial(G4int matindx) {
    return fDataPerMaterial[matindx];
  }

private:
  void     InitDataPerElement();

  void     InitDataPerMaterials();

  void     LoadDataElement(const G4Element*);

  void     InitDataMaterial(const G4Material*);

  void     ClearDataPerElement();

  void     ClearDataPerMaterial();


// data members
private:
  G4bool   fIsElectron;
  static constexpr G4int     gMaxZet    = 98;                 // max. Z for which correction data were computed (98)
  static constexpr G4int     gNumEkin   = 31;                 // number of kinetic energy grid points for Mott correction
  static constexpr G4int     gNumBeta2  = 16;                 // \beta^2 values between [fMinBeta2-fMaxBeta2]
  static constexpr G4double  gMinEkin   =   1.*CLHEP::keV;    // minimum kinetic energy value
  static constexpr G4double  gMidEkin   = 100.*CLHEP::keV;    // kinetic energy at the border of the E_{kin}-\beta^2 grids
  static constexpr G4double  gMaxBeta2  =   0.9999;           // maximum \beta^2 value
  //
  G4double                   fMaxEkin;        // from max fMaxBeta2 = 0.9999 (~50.5889 [MeV])
  G4double                   fLogMinEkin;     // \ln[fMinEkin]
  G4double                   fInvLogDelEkin;  // 1/[\ln(fMidEkin/fMinEkin)/(fNumEkin-fNumBeta2)]
  G4double                   fMinBeta2;       // <= E_{kin}=100 [keV] (~0.300546)
  G4double                   fInvDelBeta2;    // 1/[(fMaxBeta2-fMinBeta2)/(fNumBeta2-1)]
  //
  static const std::string   gElemSymbols[];
  //
  std::vector<DataPerMaterial*>  fDataPerElement;   // size will be gMaxZet+1; won't be null only at used Z indices
  std::vector<DataPerMaterial*>  fDataPerMaterial;  // size will #materials; won't be null only at used mat. indices

};

#endif
