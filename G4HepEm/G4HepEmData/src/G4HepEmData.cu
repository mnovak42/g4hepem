
#include "G4HepEmData.hh"
#include <cassert>
#include <iostream>

#include "G4HepEmMatCutData.hh"
#include "G4HepEmMaterialData.hh"
#include "G4HepEmElementData.hh"
#include "G4HepEmElectronData.hh"

void CopyG4HepEmDataToGPU (struct G4HepEmData* onCPU) {
  // Deep copy each members represented by their pointers.  

  // 1. Copy the G4HepEmMatCutData member and set the device ptr.
  CopyMatCutDataToGPU ( onCPU->fTheMatCutData, &(onCPU->fTheMatCutData_gpu) );

//  FreeMatCutDataOnGPU( &(onCPU->fTheMatCutData_gpu) );

  // 2. Copy the G4HepEmMaterialData member and set the device ptr.
  CopyMaterialDataToGPU ( onCPU->fTheMaterialData, &(onCPU->fTheMaterialData_gpu) );

//  FreeMaterialDataOnGPU( &(onCPU->fTheMaterialData_gpu) );
  
  // 3. Copy the G4HepEmElementData member and set the device ptr.
  CopyElementDataToGPU ( onCPU->fTheElementData, &(onCPU->fTheElementData_gpu) );

//    FreeElementDataOnGPU( &(onCPU->fTheElementData_gpu) );


  // 4. Copy electron data to the GPU
  CopyElectronDataToDevice( onCPU->fTheElectronData, &(onCPU->fTheElectronData_gpu));


  // 5. Copy positron data to the GPU
  CopyElectronDataToDevice( onCPU->fThePositronData, &(onCPU->fThePositronData_gpu));

}
