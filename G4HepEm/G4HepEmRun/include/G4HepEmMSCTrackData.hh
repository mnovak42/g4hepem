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
    fLambel               = o.fLambel;
    fLambtr1              = o.fLambtr1;
    fScra                 = o.fScra;
    fG1                   = o.fG1;
    fPWACorToQ1           = o.fPWACorToQ1;
    fPWACorToG2PerG1      = o.fPWACorToG2PerG1;

    fTrueStepLength       = o.fTrueStepLength;
    fZPathLength          = o.fZPathLength;
    fDisplacement[0]      = o.fDisplacement[0];
    fDisplacement[1]      = o.fDisplacement[1];
    fDisplacement[2]      = o.fDisplacement[2];
    fDirection[0]         = o.fDirection[0];
    fDirection[1]         = o.fDirection[1];
    fDirection[2]         = o.fDirection[2];

    fInitialRange         = o.fInitialRange;

    fPar1                 = o.fPar1;
    fPar2                 = o.fPar2;
    fPar3                 = o.fPar3;

    fIsNoScatteringInMSC  = o.fIsNoScatteringInMSC;
    fIsNoDisplace         = o.fIsNoDisplace;
    fIsFirstStep          = o.fIsFirstStep;
    fIsActive             = o.fIsActive;
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
    fLambel               = 0.;
    fLambtr1              = 0.;
    fScra                 = 0.;
    fG1                   = 0.;
    fPWACorToQ1           = 1.;
    fPWACorToG2PerG1      = 1.;

    fTrueStepLength       = 0.;
    fZPathLength          = 0.;
    fDisplacement[0]      = 0.;
    fDisplacement[1]      = 0.;
    fDisplacement[2]      = 0.;
    fDirection[0]         = 0.;
    fDirection[1]         = 0.;
    fDirection[2]         = 1.;

    fInitialRange         = 1.0e+21;

    fPar1                 = -1.;
    fPar2                 =  0.;
    fPar3                 =  0.;

    fIsNoScatteringInMSC  = false;
    fIsNoDisplace         = false;
    fIsFirstStep          = true;
    fIsActive             = false;
  }

public:
  double fLambel;             // elastic mfp
  double fLambtr1;            // first transport mfp
  double fScra;               // screening parameter
  double fG1;                 // fisrt stransport coefficient
  double fPWACorToQ1;         // DPWA correction to Q1
  double fPWACorToG2PerG1;    // DPWA correction to the G2/G1 ratio

  double fTrueStepLength;     // the true, i.e. physical step Length
  double fZPathLength;        // projection of the transport distance along the org. dir.
  double fDisplacement[3];    // the displacement vector
  double fDirection[3];       // direction proposed by MSC

  double fInitialRange;       // initial range value (entering in the volume)

  double fPar1;               // parameters used in the true - > geom conversion
  double fPar2;
  double fPar3;

  bool   fIsNoScatteringInMSC; // indicates that no scattering happend
  bool   fIsNoDisplace;        // indicates that displacement won't be used
  bool   fIsFirstStep;         // first step with this particle

  bool   fIsActive;

};

#endif // G4HepEmMSCTrackData
