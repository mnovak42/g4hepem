
#ifndef G4HepEmRandomEngine_HH
#define G4HepEmRandomEngine_HH

#include "G4HepEmMacros.hh"

#include <cmath>

/**
 * @file    G4HepEmRandomEngine.hh
 * @class   G4HepEmRandomEngine
 * @author  J. Hahnfeld
 * @date    2021
 *
 * A simple abstraction for a random number engine.
 *
 * Holds a reference to the real engine and two function pointers to generate
 * one random number or fill an array with a given size, respectively.
 */
class G4HepEmRandomEngine {
public:
  typedef double (*FlatFn)(void *object);
  typedef void (*FlatArrayFn)(void *object, const int size, double* vect);

  G4HepEmHostDevice
  G4HepEmRandomEngine(void *object, FlatFn flatFn, FlatArrayFn flatArrayFn)
    : fObject(object), fFlatFn(flatFn), fFlatArrayFn(flatArrayFn),
      fIsGauss(false), fGauss(0.) { }

  G4HepEmHostDevice
  double flat() { return fFlatFn(fObject); }
  G4HepEmHostDevice
  void flatArray(const int size, double* vect) {
    fFlatArrayFn(fObject, size, vect);
  }

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
    const double fac = std::sqrt(-2.*std::log(r)/r);
    fGauss   = v1*fac;
    fIsGauss = true;
    return v2*fac*stDev+mean;
  }


private:
  void *fObject;
  FlatFn fFlatFn;
  FlatArrayFn fFlatArrayFn;

  bool fIsGauss;
  double fGauss;
};

#endif // G4HepEmRandomEngine_HH
