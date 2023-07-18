
#ifndef G4HepEmConstants__HH
#define G4HepEmConstants__HH


// CLHEP::pi
static constexpr double kPi                = 3.1415926535897931e+00;
static constexpr double k2Pi               = 2.0 * kPi;

// e- m_0c^2 in [MeV] CLHEP::electron_mass_c2
static constexpr double kElectronMassC2    = 5.1099890999999997e-01;
static constexpr double kInvElectronMassC2 = 1.0 / kElectronMassC2;

// CLHEP::fine_structure_const
static constexpr double kAlpha             = 7.2973525653052150e-03;

/** \f$ \pi r_0^2\f$ */
// CLHEP::pi*CLHEP::classic_electr_radius*CLHEP::classic_electr_radius
static constexpr double kPir02             = 2.4946724123674787e-23;

/** Migdal's constant: \f$ 4\pi r_0*[\hbar c/(m_0c^2)]^2 \f$ */
// 4.0 * CLHEP::pi * CLHEP::classic_electr_radius * CLHEP::electron_Compton_length * CLHEP::electron_Compton_length
static constexpr double kMigdalConst       = 5.2804955733859579e-30;

/** LPM constant: \f$ \alpha(m_0c^2)^2/(4\pi*\hbar c) \f$ */
// CLHEP::fine_structure_const * CLHEP::electron_mass_c2 * CLHEP::electron_mass_c2 / (4.0 * CLHEP::pi * CLHEP::hbarc)
static constexpr double kLPMconstant       = 7.6843819381368661e+05;

static constexpr double kALargeValue       = 1.0E+20;


#endif  // G4HepEmConstants__HH
