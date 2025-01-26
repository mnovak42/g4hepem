#ifndef G4HepEmElectronInteractionUMSC_HH
#define G4HepEmElectronInteractionUMSC_HH

#include "G4HepEmMacros.hh"

struct G4HepEmData;
struct G4HepEmParameters;

class  G4HepEmMSCTrackData;
class  G4HepEmRandomEngine;

/**
 * @file    G4HepEmElectronInteractionUMSC.hh
 * @class   G4HepEmElectronInteractionUMSC
 * @author  M. Novak
 * @date    2022
 *
 * @brief Urban model for multiple scattering of e-/e+ for HEP applications.
 */


class G4HepEmElectronInteractionUMSC {
private:
  G4HepEmElectronInteractionUMSC() = delete;

public:

  G4HepEmHostDevice
  static void StepLimit(G4HepEmData* hepEmData, G4HepEmParameters* hepEmPars, G4HepEmMSCTrackData* mscData,
                        double ekin, int imat, int iregion, double range, double presafety,
                        bool onBoundary, bool iselectron, G4HepEmRandomEngine* rnge);

  G4HepEmHostDevice
  static void SampleScattering(G4HepEmData* hepEmData, G4HepEmMSCTrackData* mscData, double pStepLength,
                               double preStepEkin, double preStepTr1mfp, double postStepEkin, double postStepTr1mfp,
                               int imat, bool isElectron, G4HepEmRandomEngine* rnge);




  // auxilary method for sampling Urban MSC cos(theta) in the given step (used in the above `SampleScattering`)
  G4HepEmHostDevice
  static double SampleCosineTheta(double pStepLengt, double preStepEkin, double preStepTr1mfp,
                                  double postStepEkin, double postStepTr1mfp, double umscTlimitMin,
                                  double radLength, double zeff, const double* umscTailCoeff, const double* umscThetaCoeff,
                                  bool isElectron, G4HepEmRandomEngine* rnge);

  // auxilary method for sampling cos(theta) in a simplified way: using an arbitrary pdf with correct mean and stdev
  // (used in the above `SampleCosineTheta`)
  G4HepEmHostDevice
  static double SimpleScattering(double xmeanth, double x2meanth, G4HepEmRandomEngine* rnge);

  // auxilary method for computing theta0 (used in the above `SampleCosineTheta`)
  G4HepEmHostDevice
  static double ComputeTheta0(double stepInRadLength, double postStepEkin, double preStepEkin,
                              double zeff, const double* umscThetaCoeff, bool isElectron);

  // auxilary method for computing the e+ correction to theta0 (used in the above `ComputeTheta0` but only in case of e+)
  G4HepEmHostDevice
  static double Theta0PositronCorrection(double eekin, double zeff);

  // auxilary method for sampling the lateral displacement vector (x,y,0) on a rather approximate way
  G4HepEmHostDevice
  static void   SampleDisplacement(double pStepLengt, double thePhi, G4HepEmMSCTrackData* mscData, G4HepEmRandomEngine* rnge);

};

#endif // G4HepEmElectronInteractionUMSC_HH
