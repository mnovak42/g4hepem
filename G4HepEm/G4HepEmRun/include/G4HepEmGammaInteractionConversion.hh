
#ifndef G4HepEmGammaInteractionConversion_HH
#define G4HepEmGammaInteractionConversion_HH

#include "G4HepEmMacros.hh"

#include <cmath>

class  G4HepEmTLData;
class  G4HepEmRandomEngine;
struct G4HepEmData;
struct G4HepEmElementData;



void PerformGammaConversion(G4HepEmTLData* tlData, struct G4HepEmData* hepEmData);

G4HepEmHostDevice
void SampleKinEnergies(struct G4HepEmData* hepEmData, double thePrimEkin, double theLogEkin,
           int theMCIndx, double& eKinEnergy, double& pKinEnergy, G4HepEmRandomEngine* rnge);


G4HepEmHostDevice
void SampleDirections(const double* orgGammaDir, double* secElDir, double* secPosDir,
                      const double secElEkin, const double secPosEkin, G4HepEmRandomEngine* rnge);


// Target atom selector for the above bremsstrahlung intercations in case of
// materials composed from multiple elements.
G4HepEmHostDevice
int SelectTargetAtom(const struct G4HepEmGammaData* gmData, const int imat, const double ekin,
                     const double lekin, const double urndn);

G4HepEmHostDevice
double SampleEnergyRateNoLPM(const double normCond, const double epsMin, const double epsRange, const double deltaFactor,
              const double invF10, const double invF20, const double fz, G4HepEmRandomEngine* rnge);

G4HepEmHostDevice
double SampleEnergyRateWithLPM(const double normCond, const double epsMin, const double epsRange, const double deltaFactor,
                 const double invF10, const double invF20, const double fz, G4HepEmRandomEngine* rnge,
                 const double eGamma, const double lpmEnergy, const struct G4HepEmElemData* elemData);

G4HepEmHostDevice
void ComputePhi12(const double delta, double &phi1, double &phi2);


// Compute the value of the screening function 3*PHI1(delta) - PHI2(delta):
G4HepEmHostDevice
double ScreenFunction1(const double delta);


// Compute the value of the screening function 1.5*PHI1(delta) +0.5*PHI2(delta):
G4HepEmHostDevice
double ScreenFunction2(const double delta);


// Same as ScreenFunction1 and ScreenFunction2 but computes them at once
G4HepEmHostDevice
void ScreenFunction12(const double delta, double &f1, double &f2);

#endif  // G4HepEmGammaInteractionConversion_HH
