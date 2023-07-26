
#ifndef G4HepEmRandomEngine_HH
#define G4HepEmRandomEngine_HH

#include "G4HepEmMacros.hh"
#include "G4HepEmMath.hh"
#include "G4HepEmConstants.hh"

#include <cmath>

/**
 * @file    G4HepEmRandomEngine.hh
 * @class   G4HepEmRandomEngine
 * @author  J. Hahnfeld
 * @date    2021
 *
 * A simple abstraction for a random number engine.
 *
 * Holds a reference to the real engine and two member functions to use the
 * engine to generate one random number or fill an array with a given size, respectively.
 *
 * When G4HepEm is compiled with support for Geant4, the reference must be a pointer
 * to an instance of `CLHEP::HepRandomEngine` for host-side use. For device-side use,
 * the user must implement a suitable reference _and_ compile/link `__device__` implementations
 * for the `flat` and `flatArray` member functions.
 *
 * For G4HepEm built in standalone mode without Geant4 support, the user must compile and
 * link in both host- and device- side implementations for the engine and member functions.
 */
class G4HepEmRandomEngine final {
public:
  G4HepEmHostDevice
  G4HepEmRandomEngine(void *object)
    : fObject(object), fIsGauss(false), fGauss(0.) { }

  /** Return a random number uniformly distributed between 0 and 1.
   */
  G4HepEmHostDevice
  double flat();
  /** Fill elements of array with random numbers uniformly distributed between 0 and 1.
   *
   *  @param [in] size Number of elements in `vect` input array
   *  @param [in][out] vect Array to fill with random numbers
   *  @pre `size` must be less than or equal to the number of elements in `vect`
   */
  G4HepEmHostDevice
  void flatArray(const int size, double* vect);

  G4HepEmHostDevice
  double Gauss(const double mean, const double stDev) {
    if (fIsGauss) {
      fIsGauss = false;
      return fGauss*stDev+mean;
    }
    double rnd[2];
    double r, v1, v2;
    do {
      flatArray(2, rnd);
      v1 = 2.*rnd[0] - 1.;
      v2 = 2.*rnd[1] - 1.;
      r = v1*v1 + v2*v2;
    } while ( r > 1.);
    const double fac = std::sqrt(-2.*G4HepEmLog(r)/r);
    fGauss   = v1*fac;
    fIsGauss = true;
    return v2*fac*stDev+mean;
  }

  G4HepEmHostDevice
  void DiscardGauss() { fIsGauss = false; }


  G4HepEmHostDevice
  int Poisson(double mean) {
    const int   border = 16;
    const double limit = 2.E+9;

    int number = 0;
    if(mean <= border) {
      const double position = flat();
      double poissonValue   = G4HepEmExp(-mean);
      double poissonSum     = poissonValue;
      while(poissonSum <= position) {
        ++number;
        poissonValue *= mean/number;
        poissonSum   += poissonValue;
      }
      return number;
    }  // the case of mean <= 16
    //
    double rnd[2];
    flatArray(2, rnd);
    const double t = std::sqrt(-2.*G4HepEmLog(rnd[0])) * std::cos(k2Pi*rnd[1]);
    double value = mean + t*std::sqrt(mean) + 0.5;
    return value < 0.     ?  0 :
           value >= limit ? static_cast<int>(limit) : static_cast<int>(value);
  }


private:
  void *fObject;

  bool fIsGauss;
  double fGauss;
};

#endif // G4HepEmRandomEngine_HH
