
#ifndef G4HepEmParameters_HH
#define G4HepEmParameters_HH

/**
 * @file    G4HepEmParameters.hh
 * @struct  G4HepEmParameters
 * @author  M. Novak
 * @date    2020
 *
 * @brief Physics configuration related parameters.
 *
 * Collection of physics modelling related configuration parameters used in 
 * ``G4HepEm`` at initialization- and run-time.
 *
 * A single instance of this structure is created and stored in the `master`
 * G4HepEmRunManager when its InitializeGlobal() method is invoked by calling
 * the InitHepEmParameters() function declared in the G4HepEmParamatersInit
 * header file. This method extracts information (mainly) from the G4EmParameters 
 * singletone object.
 */
 
struct G4HepEmParameters { 
  /** \f$e^-/e^+\f$ tracking (kinetic) energy cut in Geant4 internal energy units: 
    * \f$e^-/e^+\f$ tracks are stopped when their energy drops below this threshold, 
    * their kinetic energy is deposited and annihilation to two \f$\gamma\f$-s interaction
    * is invoked for in case of \f$e^+\f$.*/
  double fElectronTrackingCut;
  
  // The configuration of the kinetic energy grid of the energy loss related tables:
  /** Minimum of the kinetic energy grid used to build the sub-(secondary-production)threshold 
    * related energy loss quantity tables such as the *restricted stopping power*, *range* and 
    * *inverse range* tables. */
  double fMinLossTableEnergy;
  /** Maximum of the kinetic energy grid for loss tables.*/ 
  double fMaxLossTableEnergy;
  /** Number of bins (equally spaced on log scale) of the loss table kinetic energy grid. */
  int    fNumLossTableBins;
  
  /** The *final range* parameter of the sub-threshold energy loss related step limit function.*/
  double fFinalRange;
  /** The *rover range* parameter of the sub-threshold energy loss related step limit function.*/
  double fDRoverRange;
  /** Maximum allowed *linear* energy loss along step due to sub-threshold (continuous) energy losses 
    * given as fraction of the intial kinetic energy. Proper integral is used to compute the mean energy loss 
    * when the energy loss, according to linear approximation, is over this threshold.*/
  double fLinELossLimit;
  
  /** Kinetic energy limit between the two (Seltzer-Berger and Relativistic) models for bremsstrahlung photon emission
    * in case of \f$e^-/e^+\f$ primary particles.*/
  double fElectronBremModelLim;
  
};

#endif // G4HepEmParameters_HH
