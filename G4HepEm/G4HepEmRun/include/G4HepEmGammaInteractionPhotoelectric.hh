#ifndef G4HepEmGammaInteractionPhotoelectric_HH
#define G4HepEmGammaInteractionPhotoelectric_HH

#include "G4HepEmMacros.hh"

class  G4HepEmTLData;
struct G4HepEmData;


class G4HepEmGammaInteractionPhotoelectric {
private:
  G4HepEmGammaInteractionPhotoelectric() = delete;

public:
  static void Perform(G4HepEmTLData* tlData, struct G4HepEmData* hepEmData);
};

#endif // G4HepEmGammaInteractionPhotoelectric_HH
