
#ifndef G4HepEmTrack_HH
#define G4HepEmTrack_HH

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
  G4HepEmTrack() { ReSet(); }
  
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
  void    SetPosition(double* posv) { 
    fPosition[0] = posv[0];
    fPosition[1] = posv[1];
    fPosition[2] = posv[2];
  }

  void    SetPosition(double x, double y, double z) { 
    fPosition[0] = x;
    fPosition[1] = y;
    fPosition[2] = z;
  }
  
  double* GetPosition() { return fPosition; }
  
  // Direction
  void    SetDirection(double* dirv) {
    fDirection[0] = dirv[0];
    fDirection[1] = dirv[1];
    fDirection[2] = dirv[2];
  }

  void    SetDirection(double x, double y, double z) {
    fDirection[0] = x;
    fDirection[1] = y;
    fDirection[2] = z;
  }
  
  double* GetDirection() {return fDirection; }
  
  // Kinetic energy
  void    SetEKin(double ekin)  { 
    fEKin    = ekin;
    fLogEKin = 100.0;
  }
  // !!! should be used only with special caution !!!
  void    SetLEKin(double lekin)  { fLogEKin = lekin; }
  
  double  GetEKin()    const { return fEKin; }
  double  GetLogEKin() {
    if (fLogEKin > 99.0) { 
      fLogEKin = (fEKin > 0.) ? std::log(fEKin) : -30;
    }
    return fLogEKin;
  }
    
  // Charge
  void    SetCharge(double ch) { fCharge = ch; }
  
  double  GetCharge() const { return fCharge; }
  
  //Energy deposit
  void    SetEnergyDeposit(double val) { fEDeposit  = val; }
  void    AddEnergyDeposit(double val) { fEDeposit += val; }
  double  GetEnergyDeposit()  const    { return fEDeposit; }
  
  void    SetGStepLength(double gsl)   { fGStepLength = gsl;  }
  double  GetGStepLength()    const    { return fGStepLength; }
  
  // Macroscopic cross section
  void    SetMFP(double val, int pindx) { fMFPs[pindx] = val; }
  double  GetMFP(int pindx)   const     { return fMFPs[pindx]; }
  double* GetMFP()                      { return fMFPs; }
  
  // Number of intercation left for the processes with mac-xsec above
  void    SetNumIALeft(double val, int pindx) {fNumIALeft[pindx] = val; }
  double  GetNumIALeft(int pindx) const       {return fNumIALeft[pindx]; }
  double* GetNumIALeft()                      {return fNumIALeft; }
  
  
  // ID 
  void    SetID(int id) { fID = id;   }
  int     GetID() const { return fID; }
  
  // Parent ID
  void    SetParentID(int id) { fIDParent = id;   }
  int     GetParentID() const { return fIDParent; }
  
  
  void    SetMCIndex(int imc) { fMCIndex = imc;  }
  int     GetMCIndex() const { return fMCIndex; }
  
  
  void    SetWinnerProcessIndex(int ip) { fPIndxWon = ip; }
  int     GetWinnerProcessIndex() const { return fPIndxWon; }   
  
  void    SetOnBoundary(bool val)  { fOnBoundary = val;  }
  bool    GetOnBoundary() const { return fOnBoundary; }
  
  // Reset all member values
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