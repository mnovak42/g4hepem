#ifndef G4HepEmGammaInteractionPhotoelectric_HH
#define G4HepEmGammaInteractionPhotoelectric_HH

#include "G4HepEmConstants.hh"
#include "G4HepEmMacros.hh"

class  G4HepEmTLData;
class  G4HepEmRandomEngine;
struct G4HepEmData;


class G4HepEmGammaInteractionPhotoelectric {
private:
  G4HepEmGammaInteractionPhotoelectric() = delete;

public:
  static void Perform(G4HepEmTLData* tlData, struct G4HepEmData* hepEmData);

  G4HepEmHostDevice
  static double SelectElementBindingEnergy(const struct G4HepEmData* hepEmData, const int imc, const double mxsec, const double ekin, G4HepEmRandomEngine* rnge);

  G4HepEmHostDevice
  static void SamplePhotoElectronDirection(const double theGammaE, const double* theGammaDir, double* theDir, G4HepEmRandomEngine* rnge);
};

#endif // G4HepEmGammaInteractionPhotoelectric_HH
