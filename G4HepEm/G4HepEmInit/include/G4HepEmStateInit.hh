#ifndef G4HepEmStateInit_HH
#define G4HepEmStateInit_HH

struct G4HepEmState;

/**
 * @file    G4HepEmStateInit.hh
 * @author  B. Morgan
 * @date    2021
 *
 * @brief Function to initialize ``G4HepEmState`` data structure and members
 */

/**
 * Initialize a G4HepEmState struct with parameters and data from Geant4
 *
 * The input struct's pointers will be set to the newly initialized `G4HepEmParameters`
 * and `G4HepEmData` instances. These will be constructed using the underlying
 * parameters and data from Geant4.
 *
 * Any existing data held by the input struct will not be freed
 *
 * @param[in,out] hepEmState pointer to G4HepEmState struct to initialize
 *
 * @pre Underlying Geant4 data required by G4HepEm must be initialized
 *
 */
void InitG4HepEmState(struct G4HepEmState* hepEmState);

#endif // G4HepEmStateInit_HH