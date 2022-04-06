
#ifndef G4HepEmRandomEngine_HH
#define G4HepEmRandomEngine_HH

#include "G4HepEmMacros.hh"
#include "G4HepEmMath.hh"
#include "G4HepEmConstants.hh"

#include <cmath>

/**
 * Provides common functions to a derived random number engine.
 */
template <typename Derived>
struct G4HepEmEngineBase {
  G4HepEmHostDevice double Gauss(const double mean, const double stDev)
  {
    if (fIsGauss) {
      fIsGauss = false;
      return fGauss * stDev + mean;
    }
    double rnd[2];
    double r, v1, v2;
    do {
      static_cast<Derived *>(this)->flatArray(2, rnd);
      v1 = 2. * rnd[0] - 1.;
      v2 = 2. * rnd[1] - 1.;
      r  = v1 * v1 + v2 * v2;
    } while (r > 1.);
    const double fac = std::sqrt(-2. * G4HepEmLog(r) / r);
    fGauss           = v1 * fac;
    fIsGauss         = true;
    return v2 * fac * stDev + mean;
  }

  G4HepEmHostDevice void DiscardGauss() { fIsGauss = false; }

  G4HepEmHostDevice int Poisson(double mean)
  {
    const int border   = 16;
    const double limit = 2.E+9;

    int number = 0;
    if (mean <= border) {
      const double position = static_cast<Derived *>(this)->flat();
      double poissonValue   = G4HepEmExp(-mean);
      double poissonSum     = poissonValue;
      while (poissonSum <= position) {
        ++number;
        poissonValue *= mean / number;
        poissonSum += poissonValue;
      }
      return number;
    } // the case of mean <= 16
    //
    double rnd[2];
    static_cast<Derived *>(this)->flatArray(2, rnd);
    const double t = std::sqrt(-2. * G4HepEmLog(rnd[0])) * std::cos(k2Pi * rnd[1]);
    double value   = mean + t * std::sqrt(mean) + 0.5;
    return value < 0. ? 0 : value >= limit ? static_cast<int>(limit) : static_cast<int>(value);
  }

private:
  bool fIsGauss = false;
  double fGauss = 0;
};

/**
 * @file    G4HepEmRandomEngine.hh
 * @class   G4HepEmRandomEngine
 * @author  J. Hahnfeld
 * @date    2021
 *
 * A simple type erasing abstraction for a random number engine.
 *
 * Holds a reference to the real engine and two function pointers to generate
 * one random number or fill an array with a given size, respectively.
 */
class G4HepEmRandomEngine : public G4HepEmEngineBase<G4HepEmRandomEngine> {
public:
  typedef double (*FlatFn)(void *object);
  typedef void (*FlatArrayFn)(void *object, const int size, double *vect);

  G4HepEmHostDevice G4HepEmRandomEngine(void *object, FlatFn flatFn, FlatArrayFn flatArrayFn)
      : fObject(object), fFlatFn(flatFn), fFlatArrayFn(flatArrayFn)
  {
  }

  G4HepEmHostDevice double flat() { return fFlatFn(fObject); }
  G4HepEmHostDevice void flatArray(const int size, double *vect) { fFlatArrayFn(fObject, size, vect); }

private:
  void *fObject;
  FlatFn fFlatFn;
  FlatArrayFn fFlatArrayFn;
};

#endif // G4HepEmRandomEngine_HH
