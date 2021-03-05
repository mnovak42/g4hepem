
#ifndef G4HepEmMath_HH
#define G4HepEmMath_HH

#include <cmath>

#include "G4HepEmMacros.hh"

template <typename T>
G4HepEmHostDevice static inline
T G4HepEmMax(T a, T b) {
 return a > b ? a : b;
}

template <typename T>
G4HepEmHostDevice static inline
T G4HepEmMin(T a, T b) {
 return a < b ? a : b;
}

template <typename T>
G4HepEmHostDevice static inline
T G4HepEmX13(T x) {
 return std::pow(x, 1./3.);
}

#endif // G4HepEmMath_HH
