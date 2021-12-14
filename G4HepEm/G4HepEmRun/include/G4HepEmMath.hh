
#ifndef G4HepEmMath_HH
#define G4HepEmMath_HH

#include <cmath>

#include "G4HepEmExp.hh"
#include "G4HepEmLog.hh"

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

// --- Log function with VDT (G4Log) specialisations for double and float
template <typename T>
G4HepEmHostDevice inline
T G4HepEmLog(T x) {
 return std::log(x);
}
template < >
G4HepEmHostDevice inline
double G4HepEmLog(double x) {
 return VDTLog(x);
}
template < >
G4HepEmHostDevice inline
float G4HepEmLog(float x) {
 return VDTLogf(x);
}

// --- Exp function with VDT (G4Exp) specialisations for double and float
template <typename T>
G4HepEmHostDevice inline
T G4HepEmExp(T x) {
 return std::exp(x);
}
template < >
G4HepEmHostDevice inline
double G4HepEmExp(double x) {
 return VDTExp(x);
}
template < >
G4HepEmHostDevice inline
float G4HepEmExp(float x) {
 return VDTExpf(x);
}

// --- Pow(x,a) function with the VDT (G4) Exp and Log specialisations for double and float
template <typename T>
G4HepEmHostDevice inline
T G4HepEmPow(T x, T a) {
 return std::pow(x, a);
}
template < >
G4HepEmHostDevice inline
double G4HepEmPow(double x, double a) {
 return VDTExp(a*VDTLog(x));
}
template < >
G4HepEmHostDevice inline
float G4HepEmPow(float x, float a) {
 return VDTExpf(a*VDTLogf(x));
}

#endif // G4HepEmMath_HH
