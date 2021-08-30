
#include "G4HepEmGSTableBuilder.hh"


#include "G4PhysicalConstants.hh"
#include "G4Log.hh"
#include "G4Exp.hh"

#include "G4MaterialTable.hh"
#include "G4Material.hh"
#include "G4MaterialCutsCouple.hh"

#include "G4String.hh"

#include <fstream>
#include <cstdlib>
#include <cmath>

#include <iostream>
#include <iomanip>

G4bool G4HepEmGSTableBuilder::gIsInitialised = false;

std::vector<G4HepEmGSTableBuilder::GSMSCAngularDtr*> G4HepEmGSTableBuilder::gGSMSCAngularDistributions1;
std::vector<G4HepEmGSTableBuilder::GSMSCAngularDtr*> G4HepEmGSTableBuilder::gGSMSCAngularDistributions2;
//
std::vector<double> G4HepEmGSTableBuilder::gMoliereBc;
std::vector<double> G4HepEmGSTableBuilder::gMoliereXc2;


G4HepEmGSTableBuilder::G4HepEmGSTableBuilder() {
  // set initial values: final values will be set in the Initialize method
  fLogLambda0         = 0.;              // will be set properly at init.
  fLogDeltaLambda     = 0.;              // will be set properly at init.
  fInvLogDeltaLambda  = 0.;              // will be set properly at init.
  fInvDeltaQ1         = 0.;              // will be set properly at init.
  fDeltaQ2            = 0.;              // will be set properly at init.
  fInvDeltaQ2         = 0.;              // will be set properly at init.
}

G4HepEmGSTableBuilder::~G4HepEmGSTableBuilder() {
  for (size_t i=0; i<gGSMSCAngularDistributions1.size(); ++i) {
    if (gGSMSCAngularDistributions1[i]) {
      delete [] gGSMSCAngularDistributions1[i]->fUValues;
      delete [] gGSMSCAngularDistributions1[i]->fParamA;
      delete [] gGSMSCAngularDistributions1[i]->fParamB;
      delete gGSMSCAngularDistributions1[i];
    }
  }
  gGSMSCAngularDistributions1.clear();
  for (size_t i=0; i<gGSMSCAngularDistributions2.size(); ++i) {
    if (gGSMSCAngularDistributions2[i]) {
      delete [] gGSMSCAngularDistributions2[i]->fUValues;
      delete [] gGSMSCAngularDistributions2[i]->fParamA;
      delete [] gGSMSCAngularDistributions2[i]->fParamB;
      delete gGSMSCAngularDistributions2[i];
    }
  }
  gGSMSCAngularDistributions2.clear();
  gIsInitialised = false;
}

void G4HepEmGSTableBuilder::Initialise() {
  G4double lLambdaMin = G4Log(gLAMBMIN);
  G4double lLambdaMax = G4Log(gLAMBMAX);
  fLogLambda0         = lLambdaMin;
  fLogDeltaLambda     = (lLambdaMax-lLambdaMin)/(gLAMBNUM-1.);
  fInvLogDeltaLambda  = 1./fLogDeltaLambda;
  fInvDeltaQ1         = 1./((gQMAX1-gQMIN1)/(gQNUM1-1.));
  fDeltaQ2            = (gQMAX2-gQMIN2)/(gQNUM2-1.);
  fInvDeltaQ2         = 1./fDeltaQ2;
  // load precomputed angular distributions and set up several values used during the sampling
  // these are particle independet => they go to static container: load them only onece
  if (!gIsInitialised) {
    // load pre-computed GS angular distributions (computed based on Screened-Rutherford DCS)
    LoadMSCData();
    gIsInitialised = true;
  }
  InitMoliereMSCParams();
}



const G4HepEmGSTableBuilder::GSMSCAngularDtr*
G4HepEmGSTableBuilder::GetGSAngularDtr(G4int iLambda, G4int iQ, G4bool isFirstSet) {
  const G4int numQ = isFirstSet ? gQNUM1 : gQNUM2;
  const G4int indx = iLambda*numQ + iQ;
  return isFirstSet ? gGSMSCAngularDistributions1[indx] : gGSMSCAngularDistributions2[indx];
}


void G4HepEmGSTableBuilder::LoadMSCData() {
  char* path = std::getenv("G4LEDATA");
  if (!path) {
    G4Exception("G4HepEmGSTableBuilder::LoadMSCData()","em0006",
		FatalException,
		"Environment variable G4LEDATA not defined");
    return;
  }
  //
  gGSMSCAngularDistributions1.resize(gLAMBNUM*gQNUM1,nullptr);
  const G4String str1 = G4String(path) + "/msc_GS/GSGrid_1/gsDistr_";
  for (G4int il=0; il<gLAMBNUM; ++il) {
    G4String fname = str1 + std::to_string(il);
    std::ifstream infile(fname,std::ios::in);
    if (!infile.is_open()) {
      G4String msgc = "Cannot open file: " + fname;
      G4Exception("G4HepEmGSTableBuilder::LoadMSCData()","em0006",
	 	  FatalException, msgc.c_str());
      return;
    }
    for (G4int iq=0; iq<gQNUM1; ++iq) {
      GSMSCAngularDtr *gsd = new GSMSCAngularDtr();
      infile >> gsd->fNumData;
      gsd->fUValues = new G4double[gsd->fNumData]();
      gsd->fParamA  = new G4double[gsd->fNumData]();
      gsd->fParamB  = new G4double[gsd->fNumData]();
      G4double ddummy;
      infile >> ddummy; infile >> ddummy;
      for (G4int i=0; i<gsd->fNumData; ++i) {
        infile >> gsd->fUValues[i];
        infile >> gsd->fParamA[i];
        infile >> gsd->fParamB[i];
      }
      gGSMSCAngularDistributions1[il*gQNUM1+iq] = gsd;
    }
    infile.close();
  }
  //
  // second grid
  gGSMSCAngularDistributions2.resize(gLAMBNUM*gQNUM2,nullptr);
  const G4String str2 = G4String(path) + "/msc_GS/GSGrid_2/gsDistr_";
  for (G4int il=0; il<gLAMBNUM; ++il) {
    G4String fname = str2 + std::to_string(il);
    std::ifstream infile(fname,std::ios::in);
    if (!infile.is_open()) {
      G4String msgc = "Cannot open file: " + fname;
      G4Exception("G4HepEmGSTableBuilder::LoadMSCData()","em0006",
	 	  FatalException, msgc.c_str());
      return;
    }
    for (G4int iq=0; iq<gQNUM2; ++iq) {
      G4int numData;
      infile >> numData;
      if (numData>1) {
        GSMSCAngularDtr *gsd = new GSMSCAngularDtr();
        gsd->fNumData = numData;
        gsd->fUValues = new G4double[gsd->fNumData]();
        gsd->fParamA  = new G4double[gsd->fNumData]();
        gsd->fParamB  = new G4double[gsd->fNumData]();
        double ddummy;
        infile >> ddummy; infile >> ddummy;
        for (G4int i=0; i<gsd->fNumData; ++i) {
          infile >> gsd->fUValues[i];
          infile >> gsd->fParamA[i];
          infile >> gsd->fParamB[i];
        }
        gGSMSCAngularDistributions2[il*gQNUM2+iq] = gsd;
      } else {
        gGSMSCAngularDistributions2[il*gQNUM2+iq] = nullptr;
      }
    }
    infile.close();
  }
}



// compute material dependent Moliere MSC parameters at initialisation
void G4HepEmGSTableBuilder::InitMoliereMSCParams() {
   const G4double const1   = 7821.6;      // [cm2/g]
   const G4double const2   = 0.1569;      // [cm2 MeV2 / g]
   const G4double finstrc2 = 5.325135453E-5; // fine-structure const. square

   G4MaterialTable* theMaterialTable = G4Material::GetMaterialTable();
   // get number of materials in the table
   size_t numMaterials = theMaterialTable->size();
   // make sure that we have long enough vectors
   if(gMoliereBc.size()<numMaterials) {
     gMoliereBc.resize(numMaterials);
     gMoliereXc2.resize(numMaterials);
   }
   G4double xi   = 1.0;
   G4int    maxZ = 200;
   //
   for (size_t imat=0; imat<numMaterials; ++imat) {
     const G4Material*      theMaterial     = (*theMaterialTable)[imat];
     const G4ElementVector* theElemVect     = theMaterial->GetElementVector();
     const G4int            numelems        = theMaterial->GetNumberOfElements();
     //
     const G4double*        theNbAtomsPerVolVect  = theMaterial->GetVecNbOfAtomsPerVolume();
     G4double               theTotNbAtomsPerVol   = theMaterial->GetTotNbOfAtomsPerVolume();
     //
     G4double zs = 0.0;
     G4double zx = 0.0;
     G4double ze = 0.0;
     G4double sa = 0.0;
     //
     for(G4int ielem = 0; ielem < numelems; ielem++) {
       G4double zet = (*theElemVect)[ielem]->GetZ();
       if (zet>maxZ) {
         zet = (G4double)maxZ;
       }
       G4double iwa  = (*theElemVect)[ielem]->GetN();
       G4double ipz  = theNbAtomsPerVolVect[ielem]/theTotNbAtomsPerVol;
       G4double dum  = ipz*zet*(zet+xi);
       zs           += dum;
       ze           += dum*(-2.0/3.0)*G4Log(zet);
       zx           += dum*G4Log(1.0+3.34*finstrc2*zet*zet);
       sa           += ipz*iwa;
     }
     G4double density = theMaterial->GetDensity()*CLHEP::cm3/CLHEP::g; // [g/cm3]
     //
     gMoliereBc[theMaterial->GetIndex()]  = const1*density*zs/sa*G4Exp(ze/zs)/G4Exp(zx/zs);  //[1/cm]
     gMoliereXc2[theMaterial->GetIndex()] = const2*density*zs/sa;  // [MeV2/cm]
     // change to Geant4 internal units of 1/length and energ2/length
     gMoliereBc[theMaterial->GetIndex()]  *= 1.0/CLHEP::cm;
     gMoliereXc2[theMaterial->GetIndex()] *= CLHEP::MeV*CLHEP::MeV/CLHEP::cm;
   }
}
