
#ifndef G4HepEmElectronManager_HH
#define G4HepEmElectronManager_HH

#include "G4HepEmMacros.hh"

struct G4HepEmData;
struct G4HepEmParameters;
struct G4HepEmElectronData;

class  G4HepEmTLData;
class  G4HepEmElectronTrack;
class  G4HepEmTrack; 

/**
 * @file    G4HepEmElectronManager.hh
 * @struct  G4HepEmElectronManager
 * @author  M. Novak
 * @date    2020
 *
 * @brief The top level run-time manager for e-/e+ transport simulations.
 *
 * This manager can provide the information regarding how far a given e-/e+ particle
 * goes along its original direction till it's needed to be stopped again because
 * some physics interaction(s) needs to be performed. It is also responsible to
 * perform the required interaction(s) as well.
 *
 * The two methods, through wich this manager acts on the particles, are the
 * G4HepEmElectronManager::HowFar() and G4HepEmElectronManager::Perform(). The
 * first provides the information regarding how far the particle can go, along its
 * original direction, till its next stop due to physics interaction(s).
 * The second can be used to perform the corresponding physics interaction(s).
 * All physics interactions, relevant for HEP detector simulatios, such as
 * `ionisation`, `bremsstrahlung`, `Coulomb scattering` are considered for e-/e+
 * with `annihilation` in addition for e+, including both their continuous, discrete
 * and at-rest parts pespectively. The accuracy of the models, used to describe
 * these interactions, are also compatible to those used by HEP detector simulations.
 *
 * Each G4HepEmRunManager has its own member from this manager for e-/e+ transport.
 * However, a single object could alos be used and shared by all the worker run
 * managers since this G4HepEmElectronManager is stateless. All the state and
 * thread related infomation (e.g. primary/secondary tracks or the thread local
 * random engine) are stored in the G4HepEmTLData input argument, that is also
 * used to deliver the effect of the actions of this manager (i.e. written into
 * the tracks stored in the input G4HepEmTLData argument).
 */

class G4HepEmElectronManager {

public:

  /** Functions that provides the information regarding how far a given e-/e+ particle goes.
    *
    * This functions provides the information regarding how far a given e-/e+ particle goes
    * till it's needed to be stopped again because some physics interaction(s) needs to be performed.
    * The input/primary e-/e+ particle track is provided through the G4HepEmTLData input argument. The
    * The computed physics step lenght is written directly into the input track. There is no any local
    * (state) variable used in the computation.
    *
    * @param hepEmData pointer to the top level, global, G4HepEmData structure.
    * @param hepEmPars pointer to the global, G4HepEmParameters structure.
    * @param tlData    pointer to a worker-local, G4HepEmTLData object. The corresonding object
    *   is assumed to contain all the required input information in its primary G4HepEmTLData::fElectronTrack
    *   member. This member is also used to deliver the results of the function call, i.e. the computed physics
    *   step limit is written into the G4HepEmTLData::fElectronTrack (in its fGStepLength member).
    */
  void   HowFar(struct G4HepEmData* hepEmData, struct G4HepEmParameters* hepEmPars, G4HepEmTLData* tlData);

  /** Function that provides the information regarding how far a given e-/e+ particle goes.
    *
    * This function provides the information regarding how far a given e-/e+ particle goes
    * till it's needed to be stopped again because some physics interaction(s) needs to be performed.
    * The input/primary e-/e+ particle track is provided as G4HepEmElectronTrack which must have sampled
    * `number-of-interaction-left`. The computed physics step length is written directly into the input
    * track. There is no local (state) variable used in the computation.
    *
    * @param hepEmData pointer to the top level, global, G4HepEmData structure.
    * @param hepEmPars pointer to the global, G4HepEmParameters structure.
    * @param theElTrack pointer to the input information of the track. The data structure must have all entries
    *   `number-of-interaction-left` sampled and is also used to deliver the results of the function call, i.e.
    *   the computed physics step limit is written into its fGStepLength member.
    */
  G4HepEmHostDevice
  void   HowFar(struct G4HepEmData* hepEmData, struct G4HepEmParameters* hepEmPars, G4HepEmElectronTrack* theElTrack);

  /** Functions that performs all continuous physics interactions for a given e-/e+ particle.
    *
    * This functions can be invoked when the particle is propagated to its post-step point to perform all
    * continuous physics interactions. The input/primary e-/e+ particle track is provided through as
    * G4HepEmElectronTrack. There is no local (state) variable used in the computation.
    *
    * @param hepEmData pointer to the top level, global, G4HepEmData structure.
    * @param hepEmPars pointer to the global, G4HepEmParameters structure.
    * @param theElTrack pointer to the input information of the track. All the results of this function call,
    *   i.e. the primary particle's energy updated to its post-interaction(s), are also delivered through this
    *   object.
    * @return boolean whether the particle was stopped
    */
  G4HepEmHostDevice
  bool PerformContinuous(struct G4HepEmData* hepEmData, struct G4HepEmParameters* hepEmPars, G4HepEmElectronTrack* theElTrack);

  /** Function to check if a delta interaction happens instead of the discrete process.
    *
    * @param hepEmData pointer to the top level, global, G4HepEmData structure.
    * @param hepEmPars pointer to the global, G4HepEmParameters structure.
    * @param theTrack pointer to the input information of the track.
    * @param rand number drawn at random
    * @return boolean whether a delta interaction happens
    */
  G4HepEmHostDevice
  bool CheckDelta(struct G4HepEmData* hepEmData, G4HepEmTrack* theTrack, double rand);

  /** Functions that performs all physics interactions for a given e-/e+ particle.
    *
    * This functions can be invoked when the particle is propagated to its post-step point to perform all
    * physics interactions. The input/primary e-/e+ particle track is provided through the G4HepEmTLData input
    * argument. The post-interaction(s) primary track and the secondary tracks are also provided through this
    * G4HepEmTLData input argument. There is no any local (state) variable used in the computation.
    *
    * @param hepEmData pointer to the top level, global, G4HepEmData structure.
    * @param hepEmPars pointer to the global, G4HepEmParameters structure.
    * @param tlData    pointer to a worker-local, G4HepEmTLData object. The corresonding object
    *   is assumed to contain all the required input information in its primary G4HepEmTLData::fElectronTrack
    *   member. All the results of this function call, i.e. the primary particle updated to its post-interaction(s)
    *   state as well as the possible secondary particles, are also delivered through this G4HepEmTLData.
    */
  void   Perform(struct G4HepEmData* hepEmData, struct G4HepEmParameters* hepEmPars, G4HepEmTLData* tlData);



private:

  /**
    * Auxiliary function that evaluates and provides the `restricted range` for the given kinetic energy
    * and material-cuts combination.
    *
    * @param elData pointer to the global e-/e+ data structure that contains the corresponding `Energy Loss` related data.
    * @param imc    index of the ``G4HepEm`` material-cuts in which the range is required
    * @param ekin   kinetic energy of the e-/e+ at which the range is required
    * @param lekin  logarithm of the above kinetic energy
    * @return `Restricted range` value, interpolated at the given e-/e+ kinetic energy in the given material-cuts based on
    *   the corresponding (discrete) `Energy Loss` data provded as input.
    */

  G4HepEmHostDevice
  double  GetRestRange(const struct G4HepEmElectronData* elData, const int imc, const double ekin, const double lekin);

  G4HepEmHostDevice
  double  GetRestDEDX(const struct G4HepEmElectronData* elData, const int imc, const double ekin, const double lekin);

  G4HepEmHostDevice
  double  GetInvRange(const struct G4HepEmElectronData* elData, int imc, double range);

  /*
  double  GetRestMacXSecIoni(struct G4HepEmElectronData* elData, const int imc, const double ekin, const double lekin);

  double  GetRestMacXSecBrem(struct G4HepEmElectronData* elData, const int imc, const double ekin, const double lekin);
  */

  G4HepEmHostDevice
  double  GetRestMacXSec(const struct G4HepEmElectronData* elData, const int imc, const double ekin, const double lekin, bool isioni);

  G4HepEmHostDevice
  double  GetRestMacXSecForStepping(const struct G4HepEmElectronData* elData, const int imc, double ekin, double lekin, bool isioni);

  //void    GetRestMacXSecs(struct G4HepEmElectronData* elData, const int imc, const double ekin, const double lekin, double& mxIoni, double& mxBrem);
  G4HepEmHostDevice
  double ComputeMacXsecAnnihilation(const double ekin, const double elctronDensity);

  G4HepEmHostDevice
  double ComputeMacXsecAnnihilationForStepping(const double ekin, const double elctronDensity);

};


#endif // G4HepEmElectronManager_HH
