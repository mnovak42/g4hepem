#ifndef G4HepEmMSCTrackData_HH
#define G4HepEmMSCTrackData_HH

#include "G4HepEmMacros.hh"

// A simple structure that encapsulates MSC related data for a single e-/e+ track.
//
// This object stores all the MSC related information that an e-/e+ a track needs
// to keep and propagate between the different parts of the simulation step related
// to MSC. G4HepEmElectronTrack contains an instance of this.

#include <cmath>

class G4HepEmMSCTrackData {

public:
  G4HepEmHostDevice
  G4HepEmMSCTrackData() { ReSet(); }

  G4HepEmHostDevice
  G4HepEmMSCTrackData(const G4HepEmMSCTrackData& o) {
    CopyFrom(o);
  }

  G4HepEmHostDevice
  G4HepEmMSCTrackData& operator=(const G4HepEmMSCTrackData& o) {
    if (this != &o) {
      CopyFrom(o);
    }
    return *this;
  }

  G4HepEmHostDevice
  void SetDisplacement(double x, double y, double z) {
    fDisplacement[0] = x;
    fDisplacement[1] = y;
    fDisplacement[2] = z;
  }
  G4HepEmHostDevice
  double* GetDisplacement() { return fDisplacement; }

  G4HepEmHostDevice
  void SetNewDirection(double x, double y, double z) {
    fDirection[0] = x;
    fDirection[1] = y;
    fDirection[2] = z;
  }
  G4HepEmHostDevice
  double* GetDirection() { return fDirection; }


  // reset all member values
  G4HepEmHostDevice
  void ReSet() {
    fLambtr1              = 0.;

    fTrueStepLength       = 0.;
    fZPathLength          = 0.;
    fDisplacement[0]      = 0.;
    fDisplacement[1]      = 0.;
    fDisplacement[2]      = 0.;
    fDirection[0]         = 0.;
    fDirection[1]         = 0.;
    fDirection[2]         = 1.;

    fInitialRange         = 1.0e+21;
    fDynamicRangeFactor   = 0.04;   // fr will be set in the MSC step limit
    fTlimitMin            = 1.0E-7; // tlimitmin 10*0.01 [nm] 1.0E-7[mm]

    fPar1                 = -1.;
    fPar2                 =  0.;
    fPar3                 =  0.;

    fIsNoScatteringInMSC  = false;
    fIsDisplace           = false;
    fIsFirstStep          = true;
    fIsActive             = false;
  }

  // Helper to be used in the copy constructor and assigment
  G4HepEmHostDevice
  void CopyFrom(const G4HepEmMSCTrackData& o) {
    fLambtr1              = o.fLambtr1;

    fTrueStepLength       = o.fTrueStepLength;
    fZPathLength          = o.fZPathLength;
    fDisplacement[0]      = o.fDisplacement[0];
    fDisplacement[1]      = o.fDisplacement[1];
    fDisplacement[2]      = o.fDisplacement[2];
    fDirection[0]         = o.fDirection[0];
    fDirection[1]         = o.fDirection[1];
    fDirection[2]         = o.fDirection[2];

    fInitialRange         = o.fInitialRange;
    fDynamicRangeFactor   = o.fDynamicRangeFactor;
    fTlimitMin            = o.fTlimitMin;

    fPar1                 = o.fPar1;
    fPar2                 = o.fPar2;
    fPar3                 = o.fPar3;

    fIsNoScatteringInMSC  = o.fIsNoScatteringInMSC;
    fIsDisplace           = o.fIsDisplace;
    fIsFirstStep          = o.fIsFirstStep;
    fIsActive             = o.fIsActive;
  }




public:
  double fLambtr1;            // first transport mfp

  double fTrueStepLength;     // the true, i.e. physical step Length
  double fZPathLength;        // projection of the transport distance along the org. dir.
  double fDisplacement[3];    // the displacement vector
  double fDirection[3];       // direction proposed by MSC

  double fInitialRange;       // initial range value (entering in the volume)
  double fDynamicRangeFactor; // dynamic range factor i.e. `fr`
  double fTlimitMin;          // minimum true step length i.e. `tlimitmin`

  double fPar1;               // parameters used in the true - > geom conversion
  double fPar2;
  double fPar3;

  bool   fIsNoScatteringInMSC; // indicates that no scattering happend
  bool   fIsDisplace;          // indicates that displacement needs to be done
  bool   fIsFirstStep;         // first step with this particle

  bool   fIsActive;

};

#endif // G4HepEmMSCTrackData
