#ifndef G4HepEmDataJsonIO_HH
#define G4HepEmDataJsonIO_HH

#include <iosfwd>

struct G4HepEmParameters;
struct G4HepEmData;
struct G4HepEmState;

/**
 * @file    G4HepEmDataJsonIO.hh
 * @author  B. Morgan
 * @date    2021
 *
 * @brief Functions to (de)serialize ``G4HepEMData`` data structures to/from JSON
 */

/**
 * Write a ``G4HepEmParameters` object to an output stream as JSON text
 *
 * @param[in,out] os output stream to serialize to
 * @param[in] params ``G4HepEmParameters to serialize
 *
 * @pre params must not be nullptr
 *
 * @return true if the serialization completed correctly
 */
bool G4HepEmParametersToJson(std::ostream& os, const G4HepEmParameters* params);

/**
 * Create a new ``G4HepEmParameters`` instance from an input stream of JSON data
 *
 * @param[in] is input stream to read data from
 * @return pointer to newly constructed ``G4HepEmParameters`` instance
 *
 * @post return value is ``nullptr`` if the data could not be read correctly
 */
G4HepEmParameters* G4HepEmParametersFromJson(std::istream& is);

/**
 * Write a ``G4HepEmData`` object to an output stream as JSON text
 *
 * @param[in,out] os output stream to serialize to
 * @param[in] data ``G4HepEmData`` to serialize
 *
 * @pre data must not be nullptr
 *
 * @return true if the serialization completed correctly
 */
bool G4HepEmDataToJson(std::ostream& os, const G4HepEmData* data);

/**
 * Create a new ``G4HepEmData`` instance from an input stream of JSON data
 *
 * @param[in] is input stream to read data from
 *
 * @return pointer to newly constructed ``G4HepEMData`` instance
 *
 * @post return value is ``nullptr`` if the data could not be read correctly
 */
G4HepEmData* G4HepEmDataFromJson(std::istream& is);

/**
 * Write a ``G4HepEmState`` object to an output stream as JSON text
 *
 * @param[in,out] os output stream to serialize to
 * @param[in] data ``G4HepEmState`` to serialize
 *
 * @pre data must not be nullptr
 *
 * @return true if the serialization completed correctly
 */
bool G4HepEmStateToJson(std::ostream& os, const G4HepEmState* data);

/**
 * Create a new ``G4HepEmState`` instance from an input stream of JSON data
 *
 * @param[in] is input stream to read data from
 *
 * @return pointer to newly constructed ``G4HepEMState`` instance
 *
 * @post return value is ``nullptr`` if the data could not be read correctly
 */
G4HepEmState* G4HepEmStateFromJson(std::istream& is);



#endif // G4HepEmDataDataJsonIO_HH
