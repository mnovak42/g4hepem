
#ifndef G4HepEmSBBremTableBuilder_HH
#define G4HepEmSBBremTableBuilder_HH

// #g4 includes
#include "globals.hh"
#include "G4String.hh"

#include <vector>

// forward declar
class G4MaterialCutsCouple;

class G4HepEmSBBremTableBuilder {

public:
   // CTR/DTR
   G4HepEmSBBremTableBuilder();

  ~G4HepEmSBBremTableBuilder();

   // loads and init sampling tables: lowe/highe are the low/high energy usage
   // limits of the corresponding Seltzerberger-model.
   void Initialize(const G4double lowe, const G4double highe);

   // clean away all sampling tables and makes ready for re-initialisation
   void ClearSamplingTables();


   // used only for development: print out table related information
    void Dump();

// data structure definitions
public:

   // Sampling-Table point: describes one [E_i],[kappa_j] point
   struct STPoint {
     G4double fCum;    // value of the cumulative function
     G4double fParA;   // rational function approximation based interp. parameter
     G4double fParB;   // rational function approximation based interp. parameter
   };

   // Sampling-Table: describes one [E_j] e- energy point i.e. one Table
   struct STable {
     // cumulative values for the kappa-cuts: kappa_cut_i=E_gamma_cut_i/E_el_j
     std::vector<G4double> fCumCutValues;
     // as many STPoint-s as kappa values
     std::vector<STPoint>  fSTable;
   };

   // Sampling-Tables for a given Z:
   // describes all tables (i.e. for all e- energies) for a given element (Z)
   struct SamplingTablePerZ {
     SamplingTablePerZ() : fNumGammaCuts(0), fMinElEnergyIndx(-1), fMaxElEnergyIndx(-1) {}
     size_t                fNumGammaCuts;     // number of gamma-cut for this
     G4int                 fMinElEnergyIndx;  // max(i) such E_i <= E for all E
     G4int                 fMaxElEnergyIndx;  // min(i) such E_i >= E for all E
     std::vector<STable*>  fTablesPerEnergy;  // as many table as e-ekin grid point
     //the different gamma-cut values that are defined for this element(Z) and ln
     std::vector<G4double> fGammaECuts;
     std::vector<G4double> fLogGammaECuts;
     // the couple index element stores the corresponding (sorted) gamma-cut index
     std::vector<size_t>   fMatCutIndxToGamCutIndx;
     // temporary vector to store some indecis during initialisation
     std::vector< std::vector<size_t> >   fGamCutIndxToMatCutIndx;
   };


// access of data structures 
    const SamplingTablePerZ* GetSamplingTablesForZ(int iz) { return fSBSamplingTables[iz]; }
    const double*            GetElEnergyVect()             { return fElEnergyVect.data(); }
    const double*            GetKappaVect()                { return fKappaVect.data(); }


private:

  void  BuildSamplingTables();

  void  InitSamplingTables();

  void  LoadSTGrid();

  void  LoadSamplingTables(G4int iz);

  void  ReadCompressedFile(const G4String &fname, std::istringstream &iss);


//  // simple linear search: most of the time faster than anything in our case
//  G4int LinSearch(const std::vector<STPoint>& vect,
//                  const G4int size,
//                  const G4double val);

public:

  // pre-prepared sampling tables are available:
  G4int                           fMaxZet;      // max Z number
  G4int                           fNumElEnergy; // # e- kine (E_k) per Z
  G4int                           fNumKappa;    // # red. photon eners per E_k

  // min/max electron kinetic energy usage limits
  G4double                        fUsedLowEenergy;
  G4double                        fUsedHighEenergy;
  G4double                        fLogMinElEnergy;
  G4double                        fILDeltaElEnergy;

  // e- kinetic energy and reduced photon energy grids and tehir logarithms
  std::vector<G4double>           fElEnergyVect;
  std::vector<G4double>           fLElEnergyVect;
  std::vector<G4double>           fKappaVect;
  std::vector<G4double>           fLKappaVect;

  // container to store samplingtables per Z (size is fMaxZet+1)
  std::vector<SamplingTablePerZ*> fSBSamplingTables;

};

#endif
