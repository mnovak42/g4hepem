
#ifndef G4HepEmElectronEnergyLossFluctuation_HH
#define G4HepEmElectronEnergyLossFluctuation_HH

#include "G4HepEmMacros.hh"

class G4HepEmRandomEngine;

/**
 * @file    G4HepEmElectronEnergyLossFluctuation.hh
 * @class   G4HepEmElectronEnergyLossFluctuation
 * @author  M. Novak
 * @date    2022
 *
 * @brief Urban universal model for e-/e+ energy loss fluctuation (as in 25-02-2022).
 */

class G4HepEmElectronEnergyLossFluctuation {
private:
  G4HepEmElectronEnergyLossFluctuation() = delete;

public:
  G4HepEmHostDevice
  static double SampleEnergyLossFLuctuation(double ekin, double tcut, double tmax, double excEner,
                                            double  logExcEner, double stepLength, double meanELoss,
                                            G4HepEmRandomEngine* rnge);


  //
  G4HepEmHostDevice
  static double SampleGaussianLoss(double meanx, double sig2x, G4HepEmRandomEngine* rnge);

};

#endif // G4HepEmElectronEnergyLossFluctuation_HH
