
#ifndef G4HepEmElectronInteractionBrem_HH
#define G4HepEmElectronInteractionBrem_HH

class  G4HepEmTLData;
struct G4HepEmData;
struct G4HepEmElectronData;


// Bremsstrahlung interaction based on the numerical Seltzer-Berger DCS for the 
// emitted photon energy. 
// Used between 100 eV - 1 GeV primary e-/e+ kinetic energies.
// NOTE: the core part i.e. sampling the emitted photon energy is different than 
//       that in the G4SeltzerBergerModel. I implemented here my rejection free,
//       memory effcicient (tables only per Z and not per mat-cuts) sampling. 
//       Rejection is used only to account dielectric supression and e+ correction. 
void PerformElectronBremSB(G4HepEmTLData* tlData, struct G4HepEmData* hepEmData, bool iselectron);

// Bremsstrahlung interaction based on the Bethe-Heitler DCS with modifications
// such as screening and Coulomb corrections, emission in the field of the atomic
// electrons and LPM suppression.  
// Used between 1 GeV - 100 TeV primary e-/e+ kinetic energies.
void PerformElectronBremRB(G4HepEmTLData* tlData, struct G4HepEmData* hepEmData);



// Target atom selector for the above bremsstrahlung intercations in case of 
// materials composed from multiple elements.
int SelectTargetAtomBrem(const struct G4HepEmElectronData* elData, const int imc, const double ekin, 
                         const double lekin, const double urndn, const bool isbremSB);

// Simple linear search (with step of 3!) used in the photon energy sampling part 
// of the SB (Seltzer-Berger) brem model.
int LinSearch(const double* vect, const int size, const double val);


void EvaluateLPMFunctions(double& funcXiS, double& funcGS, double& funcPhiS, const double egamma, 
                     const double etotal, const double elpm, const double z23, 
                     const double ilVarS1, const double ilVarS1Cond, const double densityCor);

// LPM functions G(s) and Phi(s) over an s-value grid of: ds=0.05 on [0:2.0] (2x41)
const double kFuncLPM[] = {
  0.0000E+00, 0.0000E+00,  6.9163E-02, 2.5747E-01,  2.0597E-01, 4.4573E-01,
  3.5098E-01, 5.8373E-01,  4.8095E-01, 6.8530E-01,  5.8926E-01, 7.6040E-01,
  6.7626E-01, 8.1626E-01,  7.4479E-01, 8.5805E-01,  7.9826E-01, 8.8952E-01,
  8.4003E-01, 9.1338E-01,  8.7258E-01, 9.3159E-01,  8.9794E-01, 9.4558E-01,
  9.1776E-01, 9.5640E-01,  9.3332E-01, 9.6483E-01,  9.4560E-01, 9.7143E-01,
  9.5535E-01, 9.7664E-01,  9.6313E-01, 9.8078E-01,  9.6939E-01, 9.8408E-01,
  9.7444E-01, 9.8673E-01,  9.7855E-01, 9.8888E-01,  9.8191E-01, 9.9062E-01,
  9.8467E-01, 9.9204E-01,  9.8695E-01, 9.9321E-01,  9.8884E-01, 9.9417E-01,
  9.9042E-01, 9.9497E-01,  9.9174E-01, 9.9564E-01,  9.9285E-01, 9.9619E-01,
  9.9379E-01, 9.9666E-01,  9.9458E-01, 9.9706E-01,  9.9526E-01, 9.9739E-01,
  9.9583E-01, 9.9768E-01,  9.9632E-01, 9.9794E-01,  9.9674E-01, 9.9818E-01,
  9.9710E-01, 9.9839E-01,  9.9741E-01, 9.9857E-01,  9.9767E-01, 9.9873E-01,
  9.9790E-01, 9.9887E-01,  9.9809E-01, 9.9898E-01,  9.9826E-01, 9.9909E-01,
  9.9840E-01, 9.9918E-01,  9.9856E-01, 9.9926E-01
}; 


#endif // G4HepEmElectronInteractionBrem_HH