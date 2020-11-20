

#ifndef G4HepEmProcess_HH
#define G4HepEmProcess_HH 

#include "G4VProcess.hh"

class  G4HepEmRunManager;  

class  G4ParticleChangeForLoss;

//
// The connection (first version) of the G4HepEm functionalities to Geant4 through
// the implementation of the Geant4 G4VProcess interface.
//
// A single instance of this process needs to be assigned to the e- and/or to 
// the e+ and/or the gamma particle in the G4 physics list of the G4 application.
// Then when running the application, all the physics realted information for 
// these partciles is provided to the Geant4 tracking by G4HepEm instead the 
// native Geant4 processes. 
//
// The G4VProcess is implemented as pure continuous physics process. Since the 
// the corresponding 'AlongStepDoIt' interface method is called in each step by
// the G4 stepping loop (after the transporation), this ensures that all (continous, 
// discrete and even at reast) interactions can be implemented in a single 
// G4HepEm method. (`PostStep` and `AtRest` DoIt methods not called at each step!)
//
// This G4 process interface implementation has a G4HepEmRunManager member that 
// is the top level interface to all G4HepEm functionalities. All the infomation, 
// required by the G4 tracking, i.e. `physical interaction length` and `do it`,  
// is provided through this G4HepEmRunManager member in the appropriate `AlongStep`
// versions of these two above interafce methods.
//
// NOTE: one instance should be assigned to all particles (see the TestEm3 example 
//       'PhysListHepEm:::ConstructProcess()' interface method for example)
// NOTE: do not assigne the same process twise i.e. both normal G4 and G4HepEm.
//


class G4HepEmProcess : public G4VProcess {
public:
   G4HepEmProcess();
  ~G4HepEmProcess();
  
  

  
   // Used for the initialization: this method is invoked by the process manager
   // whenever cross section tables needs to be rebuilt e.g. when a new matrial 
   // has been added. So this will be used for all process related initialization
   // including the process-global and process local inits.
   void BuildPhysicsTable(const G4ParticleDefinition&) override;
   
   // 
   // Everything is done along-step:
   //
   // Returns with the (geometrical) step lenght proposed by all physics 
   // processes (including all continuous, discrete and at-rest). 
   G4double AlongStepGetPhysicalInteractionLength(const G4Track& track,
                                                  G4double  previousStepSize,
                                                  G4double  currentMinimumStep,
                                                  G4double& proposedSafety,
                                                  G4GPILSelection* selection) override;
   // Performs all neccessary interactions (including all continuous, discrete 
   // and at-rest)
   G4VParticleChange* AlongStepDoIt(const G4Track&, const G4Step&) override; 

   // Interface method called by G4 tracking before a new track (primary of popped 
   // up from the secondary track stack) starts to be inserted into the stepping 
   // loop.  
   void StartTracking(G4Track*) override;
   
 
   // Everything is done along-step: there are no discrete or at-rest interaction
   G4double PostStepGetPhysicalInteractionLength(const G4Track&, G4double, G4ForceCondition*) override {
     return -1.0; 
   }
   G4VParticleChange* PostStepDoIt(const G4Track&, const G4Step&) override { 
     return nullptr;
   }

   G4double AtRestGetPhysicalInteractionLength(const G4Track&, G4ForceCondition* ) override { 
     return -1.0; 
   }
   G4VParticleChange* AtRestDoIt(const G4Track&, const G4Step& ) override {
     return nullptr;
   }
   
   
   
   void StreamInfo(std::ostream& out, const G4ParticleDefinition& part) const;



private:
  // the top level interface to the G4HepEm functionalities  
  G4HepEmRunManager*       fTheG4HepEmRunManager;

  G4ParticleChangeForLoss* fParticleChangeForLoss;

};

#endif 