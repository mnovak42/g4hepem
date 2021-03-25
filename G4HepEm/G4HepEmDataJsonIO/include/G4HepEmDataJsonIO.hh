#ifndef G4HepEmDataJsonIO_HH
#define G4HepEmDataJsonIO_HH

#include <iosfwd>

class G4HepEmData;

/**
 * @file    G4HepEmDataJsonIO.hh
 * @author  B. Morgan
 * @date    2021
 *
 * @brief Functions to (de)serialize ``G4HepEMData`` data structures to/from JSON
 */

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
 * Create a new ``G4HepEMData`` instance from an input stream of JSON data
 *
 * @param[in] is input stream to read data from
 *
 * @return pointer to newly constructed ``G4HepEMData`` instance
 *
 * @post return value is ``nullptr`` if the data could not be read correctly
 */
G4HepEmData* G4HepEmDataFromJson(std::istream& is);

#endif // G4HepEmDataDataJsonIO_HH