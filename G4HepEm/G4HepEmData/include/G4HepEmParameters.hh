
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
 * singletone object. Therefore, the default values given here will be updated
 * during the initialisation.
 */

/** Parameters per detector region. */
struct G4HepEmRegionParmeters {
  /** The *final range* parameter of the sub-threshold energy loss related step limit function.*/
  double fFinalRange = 1.0;
  /** The *rover range* parameter of the sub-threshold energy loss related step limit function.*/
  double fDRoverRange = 0.2;
  /** Maximum allowed *linear* energy loss along step due to sub-threshold (continuous) energy losses
    * given as fraction of the intial kinetic energy. Proper integral is used to compute the mean energy loss
    * when the energy loss, according to linear approximation, is over this threshold.*/
  double fLinELossLimit = 0.01;

  /** MSC range and safety factor parameters */
  double fMSCRangeFactor  = 0.04;
  double fMSCSafetyFactor = 0.6;
  /** Flag to indicate if the non-default, simplified `fMinimal` MSC step limit should be used.*/
  bool   fIsMSCMinimalStepLimit = false;

  /** Flag to indicate if energy loss fluctuation should be used.*/
  bool   fIsELossFluctuation = true;

  /** Flag to indicate if the combined MSC + Transportation process is allowed for multiple steps. */
  bool   fIsMultipleStepsInMSCTrans = true;
};


struct G4HepEmParameters {
  /** \f$e^-/e^+\f$ tracking (kinetic) energy cut in Geant4 internal energy units:
    * \f$e^-/e^+\f$ tracks are stopped when their energy drops below this threshold,
    * their kinetic energy is deposited and annihilation to two \f$\gamma\f$-s interaction
    * is invoked for in case of \f$e^+\f$.*/
  double fElectronTrackingCut = 0.001;

  // The configuration of the kinetic energy grid of the energy loss related tables:
  /** Minimum of the kinetic energy grid used to build the sub-(secondary-production)threshold
    * related energy loss quantity tables such as the *restricted stopping power*, *range* and
    * *inverse range* tables. */
  double fMinLossTableEnergy = 0.0001; // 100 eV
  /** Maximum of the kinetic energy grid for loss tables.*/
  double fMaxLossTableEnergy = 1.0E+08; // 100 TeV
  /** Number of bins (equally spaced on log scale) of the loss table kinetic energy grid. */
  int    fNumLossTableBins   = 84;

  /** Kinetic energy limit between the two (Seltzer-Berger and Relativistic) models for bremsstrahlung photon emission
    * in case of \f$e^-/e^+\f$ primary particles.*/
  double fElectronBremModelLim = 1000; // 1 GeV

  /** Number of detector regions */
  int fNumRegions = 0;
  /** A `G4HepEmRegionParmeters` array for the individual detector regions. */
  G4HepEmRegionParmeters* fParametersPerRegion = nullptr; //[fNumRegions]

#ifdef G4HepEm_CUDA_BUILD
  G4HepEmRegionParmeters* fParametersPerRegion_gpu = nullptr; //[fNumRegions]
#endif  // G4HepEm_CUDA_BUILD




  /** The *final range* parameter of the sub-threshold energy loss related step limit function.*/
  double fFinalRange = 1.0;
  /** The *rover range* parameter of the sub-threshold energy loss related step limit function.*/
  double fDRoverRange = 0.2;
  /** Maximum allowed *linear* energy loss along step due to sub-threshold (continuous) energy losses
    * given as fraction of the intial kinetic energy. Proper integral is used to compute the mean energy loss
    * when the energy loss, according to linear approximation, is over this threshold.*/
  double fLinELossLimit = 0.01;

  // MSC range and safety factor parameters
  double fMSCRangeFactor  = 0.04;
  double fMSCSafetyFactor = 0.6;


};

/** Function that ...*/
void InitG4HepEmParameters (struct G4HepEmParameters* theHepEmParams);

/** Function that ...*/
void FreeG4HepEmParameters (struct G4HepEmParameters* theHepEmParams);


#ifdef G4HepEm_CUDA_BUILD
  /** Function that makes the `G4HepEmRegionParmeters` array member of `G4HepEmParameters`
    * available on the device (the host side `_gpu` pointer will refer to the device side array).*/
  void CopyG4HepEmParametersToGPU(struct G4HepEmParameters* onCPU);

  /** Function that ...*/
  void FreeG4HepEmParametersOnGPU(struct G4HepEmParameters* onCPU);
#endif  // G4HepEm_CUDA_BUILD



#endif // G4HepEmParameters_HH
