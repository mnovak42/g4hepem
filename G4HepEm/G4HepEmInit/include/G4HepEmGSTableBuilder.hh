//
// M. Novak: it's practically a copy of my `G4GoudsmitSaundersonTable` (without
//           its run time sampling methods) that could be used directly if its
//           `gGSMSCAngularDistributions1` and `2` angular Dtr table collections
//           would be publicly available.


#ifndef G4HepEmGSTableBuilder_HH
#define G4HepEmGSTableBuilder_HH

#include <vector>

#include "G4Types.hh"

class G4MaterialCutsCouple;

class G4HepEmGSTableBuilder {

public:
  G4HepEmGSTableBuilder();
 ~G4HepEmGSTableBuilder();

  void Initialise();

  // structure to store one GS transformed angular distribution (for a given s/lambda_el,s/lambda_elG1)
  struct GSMSCAngularDtr {
    G4int     fNumData;    // # of data points
    G4double *fUValues;    // array of transformed variables
    G4double *fParamA;     // array of interpolation parameters a
    G4double *fParamB;     // array of interpolation parameters b
  };

  void   LoadMSCData();

  const GSMSCAngularDtr* GetGSAngularDtr(G4int iLambda, G4int iQ, G4bool isFirstSet);

  // material dependent MSC parameters (computed at initialisation) regarding
  // Moliere's screening parameterΩΩΩΩ
  G4double GetMoliereBc(G4int matindx)  { return gMoliereBc[matindx];  }

  G4double GetMoliereXc2(G4int matindx) { return gMoliereXc2[matindx]; }


private:
  // initialisation of material dependent Moliere's MSC parameters
  void InitMoliereMSCParams();


 private:
   static G4bool             gIsInitialised;       // are the precomputed angular distributions already loaded in?
   static constexpr G4int    gLAMBNUM = 64;        // # L=s/lambda_el in [fLAMBMIN,fLAMBMAX]
   static constexpr G4int    gQNUM1   = 15;        // # Q=s/lambda_el G1 in [fQMIN1,fQMAX1] in the 1-st Q grid
   static constexpr G4int    gQNUM2   = 32;        // # Q=s/lambda_el G1 in [fQMIN2,fQMAX2] in the 2-nd Q grid
   static constexpr G4double gLAMBMIN = 1.0;       // minimum s/lambda_el
   static constexpr G4double gLAMBMAX = 100000.0;  // maximum s/lambda_el
   static constexpr G4double gQMIN1   = 0.001;     // minimum s/lambda_el G1 in the 1-st Q grid
   static constexpr G4double gQMAX1   = 0.99;      // maximum s/lambda_el G1 in the 1-st Q grid
   static constexpr G4double gQMIN2   = 0.99;      // minimum s/lambda_el G1 in the 2-nd Q grid
   static constexpr G4double gQMAX2   = 7.99;      // maximum s/lambda_el G1 in the 2-nd Q grid
   //
   G4double fLogLambda0;          // ln(gLAMBMIN)
   G4double fLogDeltaLambda;      // ln(gLAMBMAX/gLAMBMIN)/(gLAMBNUM-1)
   G4double fInvLogDeltaLambda;   // 1/[ln(gLAMBMAX/gLAMBMIN)/(gLAMBNUM-1)]
   G4double fInvDeltaQ1;          // 1/[(gQMAX1-gQMIN1)/(gQNUM1-1)]
   G4double fDeltaQ2;             // [(gQMAX2-gQMIN2)/(gQNUM2-1)]
   G4double fInvDeltaQ2;          // 1/[(gQMAX2-gQMIN2)/(gQNUM2-1)]
   //

   // vector to store all GS transformed angular distributions (cumputed based on the Screened-Rutherford DCS)
   static std::vector<GSMSCAngularDtr*> gGSMSCAngularDistributions1;
   static std::vector<GSMSCAngularDtr*> gGSMSCAngularDistributions2;

   //@{
   /** Precomputed \f$ b_lambda_{c} $\f and \f$ \chi_c^{2} $\f material dependent
   *   Moliere parameters that can be used to compute the screening parameter,
   *   the elastic scattering cross section (or \f$ \lambda_{e} $\f) under the
   *   screened Rutherford cross section approximation. (These are used in
   *   G4GoudsmitSaundersonMscModel if fgIsUsePWATotalXsecData is FALSE.)
   */
   static std::vector<double> gMoliereBc;
   static std::vector<double> gMoliereXc2;
};

#endif
