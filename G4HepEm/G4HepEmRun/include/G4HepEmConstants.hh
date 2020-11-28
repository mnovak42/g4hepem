
#ifndef G4HepEmConstants__HH
#define G4HepEmConstants__HH

#include "CLHEP/Units/SystemOfUnits.h"
#include "CLHEP/Units/PhysicalConstants.h"

static constexpr double kPi             = CLHEP::pi;                 // e- m_0c^2 in [MeV]
static constexpr double k2Pi            = CLHEP::twopi;              // e- m_0c^2 in [MeV]

static constexpr double kElectronMassC2 = CLHEP::electron_mass_c2;   // e- m_0c^2 in [MeV]

static constexpr double kAlpha          = CLHEP::fine_structure_const;

/** \f$ \pi r_0^2\f$ */ 
static constexpr double kPir02          = CLHEP::pi*CLHEP::classic_electr_radius*CLHEP::classic_electr_radius;

/** Migdal's constant: \f$ 4\pi r_0*[\hbar c/(m_0c^2)]^2 \f$ */
static constexpr double kMigdalConst    = 4.0 * CLHEP::pi * CLHEP::classic_electr_radius * CLHEP::electron_Compton_length * CLHEP::electron_Compton_length;

/** LPM constant: \f$ \alpha(m_0c^2)^2/(4\pi*\hbar c) \f$ */
static constexpr double kLPMconstant    = CLHEP::fine_structure_const * CLHEP::electron_mass_c2 * CLHEP::electron_mass_c2 / (4.0 * CLHEP::pi * CLHEP::hbarc);


static constexpr double kALargeValue    = 1.0E+20;



#endif  // G4HepEmConstants__HH