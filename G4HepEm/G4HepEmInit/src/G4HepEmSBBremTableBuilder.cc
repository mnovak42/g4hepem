#include "G4HepEmSBBremTableBuilder.hh"

#include "G4SystemOfUnits.hh"

#include "G4Material.hh"
#include "G4ProductionCutsTable.hh"
#include "G4MaterialCutsCouple.hh"

#include "G4Log.hh"
#include "G4Exp.hh"

#include "zlib.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>

G4HepEmSBBremTableBuilder::G4HepEmSBBremTableBuilder()
 : fMaxZet(-1), fNumElEnergy(-1), fNumKappa(-1), fUsedLowEenergy(-1.),
   fUsedHighEenergy(-1.), fLogMinElEnergy(-1.), fILDeltaElEnergy(-1.)
{}

G4HepEmSBBremTableBuilder::~G4HepEmSBBremTableBuilder()
{
  ClearSamplingTables();
}

void G4HepEmSBBremTableBuilder::Initialize(const G4double lowe, const G4double highe)
{
  fUsedLowEenergy  = lowe;
  fUsedHighEenergy = highe;
  BuildSamplingTables();
  InitSamplingTables();
//  Dump();
}


void G4HepEmSBBremTableBuilder::BuildSamplingTables() {
  // claer
  ClearSamplingTables();
  LoadSTGrid();
  // First elements and gamma cuts data structures need to be built:
  // loop over all material-cuts and add gamma cut to the list of elements
  const G4ProductionCutsTable
  *thePCTable = G4ProductionCutsTable::GetProductionCutsTable();
  // a temporary vector to store one element
  std::vector<size_t> vtmp(1,0);
  size_t numMatCuts = thePCTable->GetTableSize();
  for (size_t imc=0; imc<numMatCuts; ++imc) {
    const G4MaterialCutsCouple *matCut = thePCTable->GetMaterialCutsCouple(imc);
    if (!matCut->IsUsed()) {
      continue;
    }
    const G4Material*           mat = matCut->GetMaterial();
    const G4ElementVector* elemVect = mat->GetElementVector();
    const G4int              indxMC = matCut->GetIndex();
    const G4double gamCut = (*(thePCTable->GetEnergyCutsVector(0)))[indxMC];
    if (gamCut>=fUsedHighEenergy) {
      continue;
    }
    const size_t numElems = elemVect->size();
    for (size_t ielem=0; ielem<numElems; ++ielem) {
      const G4Element *elem = (*elemVect)[ielem];
      const G4int izet = std::max(std::min(fMaxZet, elem->GetZasInt()),1);
      if (!fSBSamplingTables[izet]) {
        // create data structure but do not load sampling tables yet: will be
        // loaded after we know what are the min/max e- energies where sampling
        // will be needed during the simulation for this Z
        // LoadSamplingTables(izet);
        fSBSamplingTables[izet] = new SamplingTablePerZ();
      }
      // add current gamma cut to the list of this element data (only if this
      // cut value is still not tehre)
      const std::vector<double> &cVect = fSBSamplingTables[izet]->fGammaECuts;
      size_t indx = std::find(cVect.begin(), cVect.end(), gamCut)-cVect.begin();
      if (indx==cVect.size()) {
        vtmp[0] = imc;
        fSBSamplingTables[izet]->fGamCutIndxToMatCutIndx.push_back(vtmp);
        fSBSamplingTables[izet]->fGammaECuts.push_back(gamCut);
        fSBSamplingTables[izet]->fLogGammaECuts.push_back(G4Log(gamCut));
        ++fSBSamplingTables[izet]->fNumGammaCuts;
      } else {
        fSBSamplingTables[izet]->fGamCutIndxToMatCutIndx[indx].push_back(imc);
      }
    }
  }
}

void G4HepEmSBBremTableBuilder::InitSamplingTables() {
  const size_t numMatCuts = G4ProductionCutsTable::GetProductionCutsTable()
                            ->GetTableSize();
  for (G4int iz=1; iz<fMaxZet+1; ++iz) {
    SamplingTablePerZ* stZ = fSBSamplingTables[iz];
    if (!stZ) continue;
    // Load-in sampling table data:
    LoadSamplingTables(iz);
    // init data
    for (G4int iee=0; iee<fNumElEnergy; ++iee) {
      if (!stZ->fTablesPerEnergy[iee])
        continue;
      const G4double elEnergy = fElEnergyVect[iee];
      // 1 indicates that gamma production is not possible at this e- energy
      stZ->fTablesPerEnergy[iee]->fCumCutValues.resize(stZ->fNumGammaCuts,1.);
      // sort gamma cuts and other members accordingly
      for (size_t i=0; i<stZ->fNumGammaCuts-1; ++i) {
        for (size_t j=i+1; j<stZ->fNumGammaCuts; ++j) {
          if (stZ->fGammaECuts[j]<stZ->fGammaECuts[i]) {
            G4double dum0                   = stZ->fGammaECuts[i];
            G4double dum1                   = stZ->fLogGammaECuts[i];
            std::vector<size_t>   dumv      = stZ->fGamCutIndxToMatCutIndx[i];
            stZ->fGammaECuts[i]             = stZ->fGammaECuts[j];
            stZ->fLogGammaECuts[i]          = stZ->fLogGammaECuts[j];
            stZ->fGamCutIndxToMatCutIndx[i] = stZ->fGamCutIndxToMatCutIndx[j];
            stZ->fGammaECuts[j]             = dum0;
            stZ->fLogGammaECuts[j]          = dum1;
            stZ->fGamCutIndxToMatCutIndx[j] = dumv;
          }
        }
      }
      // set couple indices to store the corresponding gamma cut index
      stZ->fMatCutIndxToGamCutIndx.resize(numMatCuts,-1);
      for (size_t i=0; i<stZ->fGamCutIndxToMatCutIndx.size(); ++i) {
        for (size_t j=0; j<stZ->fGamCutIndxToMatCutIndx[i].size(); ++j) {
          stZ->fMatCutIndxToGamCutIndx[stZ->fGamCutIndxToMatCutIndx[i][j]] = i;
        }
      }
      // clear temporary vector
      for (size_t i=0; i<stZ->fGamCutIndxToMatCutIndx.size(); ++i) {
        stZ->fGamCutIndxToMatCutIndx[i].clear();
      }
      stZ->fGamCutIndxToMatCutIndx.clear();
      //  init for each gamma cut that are below the e- energy
      for (size_t ic=0; ic<stZ->fNumGammaCuts; ++ic) {
        const G4double gamCut = stZ->fGammaECuts[ic];
        if (elEnergy>gamCut) {
          // find lower kappa index; compute the 'xi' i.e. cummulative value for
          // gamCut/elEnergy
          const G4double cutKappa = std::max(1.e-12, gamCut/elEnergy);
          const G4int       iKLow = (cutKappa>1.e-12)
          ? std::lower_bound(fKappaVect.begin(), fKappaVect.end(), cutKappa)
            - fKappaVect.begin() -1
          : 0;
          const STPoint* stpL = &(stZ->fTablesPerEnergy[iee]->fSTable[iKLow]);
          const STPoint* stpH = &(stZ->fTablesPerEnergy[iee]->fSTable[iKLow+1]);
          const G4double pA   = stpL->fParA;
          const G4double pB   = stpL->fParB;
          const G4double etaL = stpL->fCum;
          const G4double etaH = stpH->fCum;
          const G4double alph = G4Log(cutKappa/fKappaVect[iKLow])
                               /G4Log(fKappaVect[iKLow+1]/fKappaVect[iKLow]);
          const G4double dum  = pA*(alph-1.)-1.-pB;
          G4double val = etaL;
          if (alph==0.) {
            stZ->fTablesPerEnergy[iee]->fCumCutValues[ic] = val;
          } else {
            val = -(dum+std::sqrt(dum*dum-4.*pB*alph*alph))/(2.*pB*alph);
            val = val*(etaH-etaL)+etaL;
            stZ->fTablesPerEnergy[iee]->fCumCutValues[ic] = val;
          }
        }
      }
    }
  }
}

// should be called only from LoadSamplingTables(G4int) and once
void G4HepEmSBBremTableBuilder::LoadSTGrid() {
  char* path = std::getenv("G4LEDATA");
  if (!path) {
//    G4Exception("G4HepEmSBBremTableBuilder::LoadSTGrid()","em0006",
//                FatalException, "Environment variable G4LEDATA not defined");
    return;
  }
  const G4String fname =  G4String(path) + "/brem_SB/SBTables/grid";
  std::ifstream infile(fname,std::ios::in);
  if (!infile.is_open()) {
//    G4String msgc = "Cannot open file: " + fname;
//    G4Exception("G4HepEmSBBremTableBuilder::LoadSTGrid()","em0006",
//                FatalException, msgc.c_str());
    return;
  }
  // get max Z, # electron energies and # kappa values
  infile >> fMaxZet;
  infile >> fNumElEnergy;
  infile >> fNumKappa;
  // allocate space for the data and get them in:
  // (1.) first eletron energy grid
  fElEnergyVect.resize(fNumElEnergy);
  fLElEnergyVect.resize(fNumElEnergy);
  for (G4int iee=0; iee<fNumElEnergy; ++iee) {
    G4double  dum;
    infile >> dum;
    fElEnergyVect[iee]  = dum*CLHEP::MeV;
    fLElEnergyVect[iee] = G4Log(fElEnergyVect[iee]);
  }
  // (2.) then the kappa grid
  fKappaVect.resize(fNumKappa);
  fLKappaVect.resize(fNumKappa);
  for (G4int ik=0; ik<fNumKappa; ++ik) {
    infile >> fKappaVect[ik];
    fLKappaVect[ik] = G4Log(fKappaVect[ik]);
  }
  // (3.) set size of the main container for sampling tables
  fSBSamplingTables.resize(fMaxZet+1,nullptr);
  // init electron energy grid related variables: use accurate values !!!
//  fLogMinElEnergy   = G4Log(fElEnergyVect[0]);
//  fILDeltaElEnergy  = 1./G4Log(fElEnergyVect[1]/fElEnergyVect[0]);
  const G4double elEmin  = 100.0*CLHEP::eV; //fElEnergyVect[0];
  const G4double elEmax  =  10.0*CLHEP::GeV;//fElEnergyVect[fNumElEnergy-1];
  fLogMinElEnergy  = G4Log(elEmin);
  fILDeltaElEnergy = 1./(G4Log(elEmax/elEmin)/(fNumElEnergy-1.0));
  // reset min/max energies if needed
  fUsedLowEenergy  = std::max(fUsedLowEenergy ,elEmin);
  fUsedHighEenergy = std::min(fUsedHighEenergy,elEmax);
  //
  infile.close();
}

void G4HepEmSBBremTableBuilder::LoadSamplingTables(G4int iz) {
  // check if grid needs to be loaded first
  if (fMaxZet<0) {
    LoadSTGrid();
  }
  // load data for a given Z only once
  iz = std::max(std::min(fMaxZet, iz),1);
  char* path = std::getenv("G4LEDATA");
  if (!path) {
//    G4Exception("G4HepEmSBBremTableBuilder::LoadSamplingTables()","em0006",
//                FatalException, "Environment variable G4LEDATA not defined");
    return;
  }
  const G4String fname =  G4String(path) + "/brem_SB/SBTables/sTableSB_"
                        + std::to_string(iz);
  std::istringstream infile(std::ios::in);
  // read the compressed data file into the stream
  ReadCompressedFile(fname, infile);
  // the SamplingTablePerZ object was already created, set size of containers
  // then load sampling table data for each electron energies
  SamplingTablePerZ* zTable = fSBSamplingTables[iz];
  //
  // Determine min/max elektron kinetic energies and indices
  const G4double minGammaCut = zTable->fGammaECuts[ std::min_element(
                 std::begin(zTable->fGammaECuts),std::end(zTable->fGammaECuts))
                -std::begin(zTable->fGammaECuts)];
  const G4double elEmin = std::max(fUsedLowEenergy, minGammaCut);
  const G4double elEmax = fUsedHighEenergy;
  // find low/high elecrton energy indices where tables will be needed
  // low:
  zTable->fMinElEnergyIndx = 0;
  if (elEmin>=fElEnergyVect[fNumElEnergy-1]) {
    zTable->fMinElEnergyIndx = fNumElEnergy-1;
  } else {
    zTable->fMinElEnergyIndx = std::lower_bound(fElEnergyVect.begin(),
                        fElEnergyVect.end(), elEmin) - fElEnergyVect.begin() -1;
  }
  // high:
  zTable->fMaxElEnergyIndx = 0;
  if (elEmax>=fElEnergyVect[fNumElEnergy-1]) {
    zTable->fMaxElEnergyIndx = fNumElEnergy-1;
  } else {
    // lower + 1
    zTable->fMaxElEnergyIndx = std::lower_bound(fElEnergyVect.begin(),
                           fElEnergyVect.end(), elEmax) - fElEnergyVect.begin();
  }
  // protect
  if (zTable->fMaxElEnergyIndx<=zTable->fMinElEnergyIndx) {
    return;
  }
  // load sampling tables that are needed: file is already in the stream
  zTable->fTablesPerEnergy.resize(fNumElEnergy, nullptr);
  for (G4int iee=0; iee<fNumElEnergy; ++iee) {
    // go over data that are not needed
    if (iee<zTable->fMinElEnergyIndx || iee>zTable->fMaxElEnergyIndx) {
      for (G4int ik=0; ik<fNumKappa; ++ik) {
        G4double dum;
        infile >> dum; infile >> dum; infile >> dum;
      }
    } else { // load data that are needed
      zTable->fTablesPerEnergy[iee] = new STable();
      zTable->fTablesPerEnergy[iee]->fSTable.resize(fNumKappa);
      for (G4int ik=0; ik<fNumKappa; ++ik) {
        STPoint &stP = zTable->fTablesPerEnergy[iee]->fSTable[ik];
        infile >> stP.fCum;
        infile >> stP.fParA;
        infile >> stP.fParB;
      }
    }
  }
}

// clean away all sampling tables and make ready to re-install
void G4HepEmSBBremTableBuilder::ClearSamplingTables() {
  for (G4int iz=0; iz<fMaxZet+1; ++iz) {
    if (fSBSamplingTables[iz]) {
      for (G4int iee=0; iee<fNumElEnergy; ++iee) {
        if (fSBSamplingTables[iz]->fTablesPerEnergy[iee]) {
          fSBSamplingTables[iz]->fTablesPerEnergy[iee]->fSTable.clear();
          fSBSamplingTables[iz]->fTablesPerEnergy[iee]->fCumCutValues.clear();
        }
      }
      fSBSamplingTables[iz]->fTablesPerEnergy.clear();
      fSBSamplingTables[iz]->fGammaECuts.clear();
      fSBSamplingTables[iz]->fLogGammaECuts.clear();
      fSBSamplingTables[iz]->fMatCutIndxToGamCutIndx.clear();
      //
      delete fSBSamplingTables[iz];
      fSBSamplingTables[iz] = nullptr;
    }
  }
  fSBSamplingTables.clear();
  fElEnergyVect.clear();
  fLElEnergyVect.clear();
  fKappaVect.clear();
  fLKappaVect.clear();
  fMaxZet = -1;
}

void G4HepEmSBBremTableBuilder::Dump() {
  G4cerr<< "\n  =====   Dumping ===== \n" << G4endl;
  for (G4int iz=0; iz<fMaxZet+1; ++iz) {
    if (fSBSamplingTables[iz]) {
      G4cerr<< "   ----> There are " << fSBSamplingTables[iz]->fNumGammaCuts
            << " g-cut for Z = " << iz << G4endl;
      for (size_t ic=0; ic<fSBSamplingTables[iz]->fGammaECuts.size(); ++ic)
        G4cerr<< "        i = " << ic << "  "
              << fSBSamplingTables[iz]->fGammaECuts[ic] << G4endl;
    }
  }
}


// uncompress one data file into the input string stream
void G4HepEmSBBremTableBuilder::ReadCompressedFile(const G4String &fname,
                                       std::istringstream &iss) {
  std::string *dataString = nullptr;
  std::string compfilename(fname+".z");
  // create input stream with binary mode operation and positioning at the end
  // of the file
  std::ifstream in(compfilename, std::ios::binary | std::ios::ate);
  if (in.good()) {
     // get current position in the stream (was set to the end)
     int fileSize = in.tellg();
     // set current position being the beginning of the stream
     in.seekg(0,std::ios::beg);
     // create (zlib) byte buffer for the data
     Bytef *compdata = new Bytef[fileSize];
     while(in) {
        in.read((char*)compdata, fileSize);
     }
     // create (zlib) byte buffer for the uncompressed data
     uLongf complen    = (uLongf)(fileSize*4);
     Bytef *uncompdata = new Bytef[complen];
     while (Z_OK!=uncompress(uncompdata, &complen, compdata, fileSize)) {
        // increase uncompressed byte buffer
        delete[] uncompdata;
        complen   *= 2;
        uncompdata = new Bytef[complen];
     }
     // delete the compressed data buffer
     delete [] compdata;
     // create a string from the uncompressed data (will be deleted by the caller)
     dataString = new std::string((char*)uncompdata, (long)complen);
     // delete the uncompressed data buffer
     delete [] uncompdata;
  } else {
//    std::string msg = "  Problem while trying to read "
//                      + compfilename + " data file.\n";
//    G4Exception("G4HepEmSBBremTableBuilder::ReadCompressedFile","em0006",
//                FatalException,msg.c_str());
    return;
  }
  // create the input string stream from the data string
  if (dataString) {
    iss.str(*dataString);
    in.close();
    delete dataString;
  }
}
