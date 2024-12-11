#ifndef G4HepEmWoodcockHelper_h
#define G4HepEmWoodcockHelper_h 1

#include "G4Navigator.hh"
#include "globals.hh"

class G4VSolid;
class G4MaterialCutsCouple;
class G4Material;
class G4LogicalVolume;
class G4VPhysicalVolume;
class G4Track;

class G4HepEmData;
class G4HepEmMatCutData;
class G4HepEmGammaTrack;

#include <map>

class G4HepEmWoodcockHelper {
public:
    G4HepEmWoodcockHelper();
   ~G4HepEmWoodcockHelper();

    // Returns true only if at least one Woodcock tracking region has been found
    G4bool Initialize(std::vector<std::string>& wdtRegionNames, struct G4HepEmMatCutData* hepEmMatCutData, G4VPhysicalVolume* worldVolume);


    void     SetKineticEnergyLimit(G4double val) { fWDTKineticEnergyLimit = val; }
    G4double GetKineticEnergyLimit() { return fWDTKineticEnergyLimit; }


    // Checks if this step will be done in a WDT region with high enough kinetic energy.
    // Finds the root volume of the region in chich this step will be done and
    // sets the following fileds: fWDTTransform, fWDTHepEmIMC, fWDTSolid, fWDTCouple
    // Returns `false` is the step is not in a WDT region or the energy is too low.
    G4bool  FindWDTVolume(int regionID, const G4Track& aTrack);

    // Keeps "Woodock" tracking inside an envelop, found and set up in `FindWDTVolume`,
    // till the photon gets close to this volume boundary (returns `true`) or
    // reaches a point of interaction (returns `false`). The accumulated step length
    // during the tracking is written to the field of the input G4HepEmTrack while
    // all relevant pre-step point infomatin is set to thier post-step point value
    // (as several volume boundaries might have been crossed between the pre- and
    // post step points and interaction can only happen at the post step point if any).
    G4bool KeepTracking(const struct G4HepEmData* theHepEmData, G4HepEmGammaTrack* theGammaTrack, G4Track& aTrack);


private:

   void ClearData();

   void FindWDTMaterial(G4LogicalVolume* lvol, double& maxDensity, G4Material** maxDensityMat);


   // One `WDTDataPerRootLogVol` data is structured for each root logical volume
   // of a Woodcock tracking reagion: with a pointer to its solid, to the mat.-
   // cuts couple of the "heaviest" material in this branch, i.e. below the root
   // logical volume, and its corresponding G4HepEm material-cuts couple index.
   struct WDTDataPerRootLogVol {
     WDTDataPerRootLogVol()
     : fSolid(nullptr), fG4Couple(nullptr), fG4CoupleHepEmIndex(-1) {}
     WDTDataPerRootLogVol(G4VSolid* solid, G4MaterialCutsCouple* couple, G4int hepEmIMC)
     : fSolid(solid),   fG4Couple(couple),  fG4CoupleHepEmIndex(hepEmIMC) {}
     G4VSolid*              fSolid;              // solid of the root logical vol.
     G4MaterialCutsCouple*  fG4Couple;           // couple with the heaviest material
     G4int                  fG4CoupleHepEmIndex; // G4HepEm mat-cut index of that
   };

   // All `WDTDataPerRootLogVol` that belongs to one Woodcock tracking region, i.e.
   // as many as root logical volume of that region. Indexed by the ID of the
   // root logical volume (i.e. that's the key).
   struct WDTDataForARegion {
     std::map<G4int, WDTDataPerRootLogVol*> fWDTDataRegion;
   };

   // Woodcock tracking related data for all regions where Woodcock tracking was
   // requested by giving the name of the regions (and a detector region with that
   // name has been found). Indexed by the G4Region ID (i.e. that's the key).
   // All these data are initialised when the `Initialize` method is invoked.
   std::map<G4int, WDTDataForARegion*> fWDTData;

   // Some data that are used during the Woodcock tracking and their values are
   // set accoring to the root logical volume inside which the actual tracking is
   // performed. Most of these values are set based on the `WDTDataPerRootLogVol`,
   // (stored for all root logical volumes of all Woodcock tracking regions)
   // inside the `FindRootVolume` method.
   // NOTE: none of these objects are owned
   G4VSolid*             fWDTSolid;
   G4MaterialCutsCouple* fWDTCouple;
   G4int                 fWDTHepEmIMC;
   G4AffineTransform     fWDTTransform;  // transformation of the actual phy. vol.

   // A navigator used to locate points (not to mess with the navigator for tarcking)
   G4Navigator           fWDTNavigator;

   // A kinetic energy limit below which Woodcock tracking is turned off.
   G4double              fWDTKineticEnergyLimit;
};

#endif // G4HepEmWoodcockHelper
