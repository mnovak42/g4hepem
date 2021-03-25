# Testing Export and Import of G4HepEM data

This test constructs a *"fake"* ``Geant4`` setup using its NIST pre-defined materials to create either

- a "full" setup of >300 material-cuts couples
- a "simple" setup of two materials, three volumes, and two ``G4Region``s with different
  secondary production thresholds.

After initialising ``G4HepEm``, the constructed ``G4HepEmData`` object is serialized to
JSON format in a file on disk. A new ``G4HepEm`` instance is then constructed by deserializing
this file. Both instances are compared numerically (including all sub structures and data), the
test only passing if they are equal. No CUDA/Device operations are used as the serialization is
a pure Host side operation.

The test may be run in `full` mode via (from the build directory):

```
$ ./Testing/DataImportExport/TestDataImportExport full
```

and in `simple` mode:

```
$ ./Testing/DataImportExport/TestDataImportExport simple
```

No pretty-formatting is performed on the output `.json` files.

