
#ifndef G4HepEmConstants__HH
#define G4HepEmConstants__HH

#include "CLHEP/Units/SystemOfUnits.h"
#include "CLHEP/Units/PhysicalConstants.h"

static constexpr double kPi             = CLHEP::pi;                 // e- m_0c^2 in [MeV]
static constexpr double k2Pi            = CLHEP::twopi;              // e- m_0c^2 in [MeV]

static constexpr double kElectronMassC2 = CLHEP::electron_mass_c2;   // e- m_0c^2 in [MeV]

static constexpr double kAlpha          = CLHEP::fine_structure_const;

static constexpr double kPir02          = CLHEP::pi*CLHEP::classic_electr_radius*CLHEP::classic_electr_radius;

static constexpr double kALargeValue    = 1.0E+20;



#endif  // G4HepEmConstants__HH