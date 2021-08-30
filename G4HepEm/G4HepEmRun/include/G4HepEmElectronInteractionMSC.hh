
#ifndef G4HepEmElectronInteractionMSC_HH
#define G4HepEmElectronInteractionMSC_HH

#include "G4HepEmMacros.hh"

class  G4HepEmTLData;
class  G4HepEmRandomEngine;

class  G4HepEmMSCTrackData;

struct G4HepEmData;
struct G4HepEmParameters;
struct G4HepEmElectronData;
struct G4HepEmGSTableData;

// Multiple scattering of e-/e+ based on the Goudsmit-Saunderson angular
// distributions computed by using the screened Rutherford DCS for elastic
// scattering using Moliere's screening parameters and a DPWA based correction
// to the correspondig integrated quantities.

class G4HepEmElectronInteractionMSC {
private:
  G4HepEmElectronInteractionMSC() = delete;

public:

  G4HepEmHostDevice
  static void StepLimit(G4HepEmData* hepEmData, G4HepEmParameters* hepEmPars, G4HepEmMSCTrackData* mscData,
                        double pStepLength, double ekin, double lekin, int imc, double range, double presafety,
                        bool onBoundary, bool iselectron, G4HepEmRandomEngine* rnge);

  G4HepEmHostDevice
  static void SampleScattering(G4HepEmData* hepEmData, G4HepEmMSCTrackData* mscData, double pStepLength,
                               double ekin, double eloss, int imc, bool iselectron, G4HepEmRandomEngine* rnge);


  // auxilary method for MSC step limit randomisation
  G4HepEmHostDevice
  static double RandomizeTrueStepLength(double tlimit, G4HepEmRandomEngine* rnge);

  // main auxiliary method to compute the elastic, first transport mean free paths, the
  // screening parameter and G1. PWA corrections to Q1 and G2/G1 will also be delivered.
  G4HepEmHostDevice
  static void  ComputeParameters(int imat, double ekin, double lekin, double &lambel, double &lambtr1, double &scra, double &g1,
                                 double &pwaCorToQ1, double &pwaCorToG2PerG1, const G4HepEmGSTableData* gsTable, bool iselectron);
  // auxilary method, used in the above, for interpolating the DPWA correction factors for e- or e+
  // in a given material and kinetic energy based on the data stored in the G4HepEmGSTableData.
  G4HepEmHostDevice
  static void  GetPWACorrectionFactors(double logekin, double beta2, int imat, double& pwaCorToScrA, double& pwaCorToQ1,
                                       double& pwaCorToG2PerG1, const G4HepEmGSTableData* gsTable, bool iselectron);


  // main auxiliary method for sampling angular deflections including no, single, few or multiple scattering
  G4HepEmHostDevice
  static bool   SampleAngularDeflection(double lambdaval, double expn, double qval, double scra,
                                        double &cost, double &sint, double &transfPar, double** dtrData,
                                        G4HepEmRandomEngine* rnge, G4HepEmGSTableData* gsTable, bool isfirst);
  // auxilary method used inside the above for handling the MSC scattering case
  G4HepEmHostDevice
  static double SampleMSCCosTheta(double lambdaval, double qval, double scra,
                                  double &transfPar, double** dtrData,
                                  G4HepEmRandomEngine* rnge, G4HepEmGSTableData* gsTable, bool isfirst);
  // auxilary method used inside the above for getting the pointer where the current dtr data starts
  // (the transformation parameter will also be delivered through `transfpar`)
  G4HepEmHostDevice
  static double* GetGSAngularDtr(double scra, double lambdaval, double qval,
                                double &transfpar, G4HepEmRandomEngine* rnge, G4HepEmGSTableData* gsTable);

};

#endif // G4HepEmElectronInteractionBrem_HH
