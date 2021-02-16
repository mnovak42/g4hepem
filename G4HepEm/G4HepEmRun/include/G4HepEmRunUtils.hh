
#ifndef G4HepEmRunUtils_HH
#define G4HepEmRunUtils_HH

#include "G4HepEmMacros.hh"

// Roate the direction [u,v,w] given in the scattering frame to the lab frame.
// Details: scattering is described relative to the [0,0,1] direction (i.e. scattering
// frame). Therefore, after the new direction is computed relative to this [0,0,1]
// original direction, the real original direction [u1,u2,u3] in the lab frame
// needs to be accounted and the final new direction, i.e. in the lab frame is
// computed.
G4HepEmHostDevice
void RotateToReferenceFrame(double &u, double &v, double &w, const double* refDir);

G4HepEmHostDevice
void RotateToReferenceFrame(double* dir, const double* refDir);



// get spline interpolation over a log-spaced xgrid previously prepared by
// PrepareSpline (separate storrage of ydata and second deriavtive)
// use the improved, robust spline interpolation that I put in G4 10.6
G4HepEmHostDevice
double GetSplineLog(int ndata, double* xdata, double* ydata, double* secderiv, double x, double logx, double logxmin, double invLDBin);

// get spline interpolation over a log-spaced xgrid previously prepared by
// PrepareSpline (compact storrage of ydata and second deriavtive in ydata)
// use the improved, robust spline interpolation that I put in G4 10.6
G4HepEmHostDevice
double GetSplineLog(int ndata, double* xdata, double* ydata, double x, double logx, double logxmin, double invLDBin);

// get spline interpolation over a log-spaced xgrid previously prepared by
// PrepareSpline (compact storrage of xdata, ydata and second deriavtive in data)
// use the improved, robust spline interpolation that I put in G4 10.6
G4HepEmHostDevice
double GetSplineLog(int ndata, double* data, double x, double logx, double logxmin, double invLDBin);


// get spline interpolation over any xgrid: idx = i such  xdata[i] <= x < xdata[i+1]
// and x >= xdata[0] and x<xdata[ndata-1]
// PrepareSpline (separate storrage of ydata and second deriavtive)
// use the improved, robust spline interpolation that I put in G4 10.6
G4HepEmHostDevice
double GetSpline(double* xdata, double* ydata, double* secderiv, double x, int idx, int step=1);

// get spline interpolation if it was prepared with compact storrage of ydata
// and second deriavtive in ydata
// use the improved, robust spline interpolation that I put in G4 10.6
G4HepEmHostDevice
double GetSpline(double* xdata, double* ydata, double x, int idx);

// get spline interpolation if it was prepared with compact storrage of xdata,
// ydata and second deriavtive in data
G4HepEmHostDevice
double GetSpline(double* data, double x, int idx);


// finds the lower index of the x-bin in an ordered, increasing x-grid such
// that x[i] <= x < x[i+1]
G4HepEmHostDevice
int    FindLowerBinIndex(double* xdata, int num, double x, int step=1);


#endif // G4HepEmRunUtils_HH
