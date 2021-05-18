#include "G4HepEmDataJsonIO.hh"

#include "G4HepEmDataJsonIOImpl.hh"

#include <iostream>

// Serialize parameters to os, return true if successful, false otherwise
bool G4HepEmParametersToJson(std::ostream& os, const G4HepEmParameters* data) {
  json jout;
  // Pointers need more work in adl_serializer to cope with both
  // const and non-const pointers. We take, for now, the dumb way, which
  // is to call the adl_serializer _directly_ (rather than just `json j = data`)
  nlohmann::adl_serializer<G4HepEmParameters*>::to_json(jout, data);
  os << jout;
  return true;
}

// Deserialize data from is, return new instance if successful, nullptr otherwise
G4HepEmParameters* G4HepEmParametersFromJson(std::istream& is) {
  json jin;
  is >> jin;
  G4HepEmParameters* inData = jin.get<G4HepEmParameters*>();
  return inData;
}

// Serialize data to os, return true if successful, false otherwise
bool G4HepEmDataToJson(std::ostream& os, const G4HepEmData* data) {
  json jout;
  // Pointers need more work in adl_serializer to cope with both
  // const and non-const pointers. We take, for now, the dumb way, which
  // is to call the adl_serializer _directly_ (rather than just `json j = data`)
  nlohmann::adl_serializer<G4HepEmData*>::to_json(jout, data);
  os << jout;
  return true;
}

// Deserialize data from is, return new instance if successful, nullptr otherwise
G4HepEmData* G4HepEmDataFromJson(std::istream& is) {
  json jin;
  is >> jin;
  G4HepEmData* inData = jin.get<G4HepEmData*>();
  return inData;
}

// Serialize data to os, return true if successful, false otherwise
bool G4HepEmStateToJson(std::ostream& os, const G4HepEmState* data) {
  json jout;
  // Pointers need more work in adl_serializer to cope with both
  // const and non-const pointers. We take, for now, the dumb way, which
  // is to call the adl_serializer _directly_ (rather than just `json j = data`)
  nlohmann::adl_serializer<G4HepEmState*>::to_json(jout, data);
  os << jout;
  return true;
}

// Deserialize data from is, return new instance if successful, nullptr otherwise
G4HepEmState* G4HepEmStateFromJson(std::istream& is) {
  json jin;
  is >> jin;
  G4HepEmState* inData = jin.get<G4HepEmState*>();
  return inData;
}