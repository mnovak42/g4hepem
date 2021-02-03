
#ifndef G4HepEmTrack_HH
#define G4HepEmTrack_HH

#include "G4HepEmMacros.hh"

// A simple track structure for neutral particles.
//
// This object stores all the basic information (e.g. position, directin, etc)
// a track needs to keep and propagate between the different component of the
// simulation.
// Both the G4HepEmElectronTrack and G4HepEmGammaTrack contain an instance of
// this. While the G4HepEmTrack is already sufficient to store all information
// for gamma particle tracks the G4HepEmElectronTrack contains additional members
// speacial for charged particle tracks and their simualtion.

#include <cmath>

class G4HepEmTrack {

public:
  G4HepEmHostDevice
  G4HepEmTrack() { ReSet(); }

  G4HepEmHostDevice
  G4HepEmTrack(const G4HepEmTrack& o) {
    fPosition[0]   = o.fPosition[0];
    fPosition[1]   = o.fPosition[1];
    fPosition[2]   = o.fPosition[2];

    fDirection[0]  = o.fDirection[0];
    fDirection[1]  = o.fDirection[1];
    fDirection[2]  = o.fDirection[2];

    fEKin          = o.fEKin;
    fLogEKin       = o.fLogEKin;

    fCharge        = o.fCharge;

    fEDeposit      = o.fEDeposit;

    fGStepLength   = o.fGStepLength;

    fMFPs[0]       = o.fMFPs[0];
    fMFPs[1]       = o.fMFPs[1];
    fMFPs[2]       = o.fMFPs[2];

    fNumIALeft[0] = o.fNumIALeft[0];
    fNumIALeft[1] = o.fNumIALeft[1];
    fNumIALeft[2] = o.fNumIALeft[2];

    fID           = o.fID;
    fIDParent     = o.fIDParent;

    fMCIndex      = o.fMCIndex;

    fPIndxWon     = o.fPIndxWon;

    fOnBoundary   = o.fOnBoundary;
  }

  // Position
  G4HepEmHostDevice
  void    SetPosition(double* posv) {
    fPosition[0] = posv[0];
    fPosition[1] = posv[1];
    fPosition[2] = posv[2];
  }

  G4HepEmHostDevice
  void    SetPosition(double x, double y, double z) {
    fPosition[0] = x;
    fPosition[1] = y;
    fPosition[2] = z;
  }

  G4HepEmHostDevice
  double* GetPosition() { return fPosition; }

  // Direction
  G4HepEmHostDevice
  void    SetDirection(double* dirv) {
    fDirection[0] = dirv[0];
    fDirection[1] = dirv[1];
    fDirection[2] = dirv[2];
  }

  G4HepEmHostDevice
  void    SetDirection(double x, double y, double z) {
    fDirection[0] = x;
    fDirection[1] = y;
    fDirection[2] = z;
  }

  G4HepEmHostDevice
  double* GetDirection() {return fDirection; }

  // Kinetic energy
  G4HepEmHostDevice
  void    SetEKin(double ekin)  {
    fEKin    = ekin;
    fLogEKin = 100.0;
  }

  // !!! should be used only with special caution !!!
  G4HepEmHostDevice
  void    SetLEKin(double lekin)  { fLogEKin = lekin; }
  G4HepEmHostDevice
  void    SetEKin(double ekin, double lekin)  {
    fEKin    = ekin;
    fLogEKin = lekin;
  }



  G4HepEmHostDevice
  double  GetEKin()    const { return fEKin; }
  G4HepEmHostDevice
  double  GetLogEKin() {
    if (fLogEKin > 99.0) {
      fLogEKin = (fEKin > 0.) ? std::log(fEKin) : -30;
    }
    return fLogEKin;
  }

  // Charge
  G4HepEmHostDevice
  void    SetCharge(double ch) { fCharge = ch; }

  G4HepEmHostDevice
  double  GetCharge() const { return fCharge; }

  //Energy deposit
  G4HepEmHostDevice
  void    SetEnergyDeposit(double val) { fEDeposit  = val; }
  G4HepEmHostDevice
  void    AddEnergyDeposit(double val) { fEDeposit += val; }
  G4HepEmHostDevice
  double  GetEnergyDeposit()  const    { return fEDeposit; }

  G4HepEmHostDevice
  void    SetGStepLength(double gsl)   { fGStepLength = gsl;  }
  G4HepEmHostDevice
  double  GetGStepLength()    const    { return fGStepLength; }

  // Macroscopic cross section
  G4HepEmHostDevice
  void    SetMFP(double val, int pindx) { fMFPs[pindx] = val; }
  G4HepEmHostDevice
  double  GetMFP(int pindx)   const     { return fMFPs[pindx]; }
  G4HepEmHostDevice
  double* GetMFP()                      { return fMFPs; }

  // Number of intercation left for the processes with mac-xsec above
  G4HepEmHostDevice
  void    SetNumIALeft(double val, int pindx) {fNumIALeft[pindx] = val; }
  G4HepEmHostDevice
  double  GetNumIALeft(int pindx) const       {return fNumIALeft[pindx]; }
  G4HepEmHostDevice
  double* GetNumIALeft()                      {return fNumIALeft; }


  // ID
  G4HepEmHostDevice
  void    SetID(int id) { fID = id;   }
  G4HepEmHostDevice
  int     GetID() const { return fID; }

  // Parent ID
  G4HepEmHostDevice
  void    SetParentID(int id) { fIDParent = id;   }
  G4HepEmHostDevice
  int     GetParentID() const { return fIDParent; }


  G4HepEmHostDevice
  void    SetMCIndex(int imc) { fMCIndex = imc;  }
  G4HepEmHostDevice
  int     GetMCIndex() const { return fMCIndex; }


  G4HepEmHostDevice
  void    SetWinnerProcessIndex(int ip) { fPIndxWon = ip; }
  G4HepEmHostDevice
  int     GetWinnerProcessIndex() const { return fPIndxWon; }

  G4HepEmHostDevice
  void    SetOnBoundary(bool val)  { fOnBoundary = val;  }
  G4HepEmHostDevice
  bool    GetOnBoundary() const { return fOnBoundary; }

  // Reset all member values
  G4HepEmHostDevice
  void ReSet() {
    fPosition[0]  = 0.0;
    fPosition[1]  = 0.0;
    fPosition[2]  = 0.0;

    fDirection[0] = 0.0;
    fDirection[1] = 0.0;
    fDirection[2] = 0.0;

    fEKin         = 0.0;
    fLogEKin      = 100.0;

    fCharge       = 0.0;

    fEDeposit     = 0.0;

    // step length along the original direction
    fGStepLength  = 0.0;

    fMFPs[0]      = -1.0;
    fMFPs[1]      = -1.0;

    fNumIALeft[0] = -1.0;
    fNumIALeft[1] = -1.0;

    fID           =  -1;
    fIDParent     =  -1;

    fMCIndex      =  -1;

    fPIndxWon     =  -1;

    fOnBoundary   = false;
  }



private:

  double   fPosition[3];
  double   fDirection[3];
  double   fEKin;
  double   fLogEKin;
  double   fCharge;
  double   fEDeposit;
  double   fGStepLength;   // step length along the original direction (straight line)
  double   fMFPs[3];       // pair, compton, photo-electric in case of photon
  double   fNumIALeft[3];  // ioni, brem, (e+-e- annihilation) in case of e- (e+)

  int      fID;
  int      fIDParent;

  int      fMCIndex;

  int      fPIndxWon; // 0-pair, 1-compton, 2-photo-electric for photon
                      // 0-ioni, 1-brem,  (2-annihilation) for e- (e+)
  bool     fOnBoundary;
};


#endif // G4HepEmTrack_HH
