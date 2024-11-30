
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

// get spline interpolation of y(x) between (x1, x2) given y_N = y(x_N), y''N(x_N)
G4HepEmHostDevice
double GetSpline(double x1, double x2, double y1, double y2, double secderiv1, double secderiv2, double x);

// get linear interpolation of y(x) between (x1, x2) given y_1 = y(x_N)
G4HepEmHostDevice
double GetLinear(double x1, double x2, double y1, double y2, double x);

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


// get linear interpolation over a log-spaced xgrid previously prepared by
// PrepareSpline; compact storrage of xdata, ydata in data with a single `y`
// at each `x` (x_i,y_i, x_i+1,y_i+1,...)
G4HepEmHostDevice
double GetLinearLog(int ndata, double* data, double x, double logx, double logxmin, double invLDBin);

// same as above but having 2 `y` data now at each `x` (x_i,y1_i,y2_i, x_i+1,y_i+1,y2_i+1,,...)
// `iwhich=1` interpolates `y1` while `iwhich=2` `y2`
G4HepEmHostDevice
double GetLinearLog2(int ndata, double* data, double x, double logx, double logxmin, double invLDBin, int iwhich);
// same as above but interpolate sthe 2 data at once
G4HepEmHostDevice
void GetLinearLog2(int ndata, double* data, double x, double logx, double logxmin, double invLDBin, double res[2]);

/*
// same as GetLinearLog2 but general for N `y` data at each `x` with `shift=N+1`
// - the special case above with `N=2`: `shift=N+1=3` and this the same as the above
// - the special case having `N=1`: `shift=1+1=2` and with `which=1` this the same as the above GetLinearLog
G4HepEmHostDevice
double GetLinearLogN(int ndata, double* data, double x, double logx, double logxmin, double invLDBin, int iwhich, int shift);
*/

// get spline interpolation over a log-spaced xgrid previously prepared by
// PrepareSpline; compact storrage of xdata, ydata in data with 4 `y` values
// and their second derivatives at each `x` (x_i, y1_i,y1_i'',y2_i,y2_i'',y3_i,y3_i'',y4_i,y4_i'', x_i+1...)
// `iwhich=1` interpolates `y1`; `iwhich=2` `y2`, `iwhich=3` `y3`, `iwhich=4` `y4`
G4HepEmHostDevice
double GetSplineLog4(int ndata, double* data, double x, double logx, double logxmin, double invLDBin, int iwhich);
//same as above but interpolates all the 4 data at once
G4HepEmHostDevice
void GetSplineLog4(int ndata, double* data, double x, double logx, double logxmin, double invLDBin, double res[4]);

/*
// same as GetSplineLog4 but general for N `y` data and their second derivatives at each `x`
// - the special case above with `N=4`: `shift=2xN +1=9` and this the same as the above
// - the special case having `N=1`: `shift=2xN +1=3` and with `which=1` this the same as the above GetSplineLog with the single `double* data` array
G4HepEmHostDevice
double GetSplineLogN(int ndata, double* data, double x, double logx, double logxmin, double invLDBin, int iwhich, int shift);
*/




// finds the lower index of the x-bin in an ordered, increasing x-grid such
// that x[i] <= x < x[i+1]
G4HepEmHostDevice
int    FindLowerBinIndex(double* xdata, int num, double x, int step=1);


#endif // G4HepEmRunUtils_HH
