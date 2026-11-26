#ifndef G4HepEmParamatersInit_HH
#define G4HepEmParamatersInit_HH

struct G4HepEmParamaters;
struct G4HepEmRegionParmeters;

void InitHepEmParameters(struct G4HepEmParameters* hepEmPars);

// Sets the default values of one region from G4EmParameters and the global tracking cuts.
void InitHepEmRegionParameters(const struct G4HepEmParameters* hepEmPars, struct G4HepEmRegionParmeters& rDat);

// Extends the per-region array to all regions in the G4RegionStore; regions created
// after `InitHepEmParameters` get the values of `defaults`. Returns true if it was extended.
bool ExtendHepEmRegionParameters(struct G4HepEmParameters* hepEmPars,
                                 const struct G4HepEmRegionParmeters& defaults);

#endif // G4HepEmParamaters_HH
