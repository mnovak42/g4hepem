
#include "G4HepEmInitUtils.hh"

#include <cmath>
#include <algorithm>

G4HepEmInitUtils& G4HepEmInitUtils::Instance() {
  static G4HepEmInitUtils instance;
  return instance;
}


void G4HepEmInitUtils::GLIntegral(int npoints, double* abscissas, double* weights, 
                                  double min, double max) {
  const double kPi      = 3.14159265358979323846;
  const double kEpsilon = 1.0E-13;
  double xm,xl,z,z1,p1,p2,p3,pp;
  int m = (int)(0.5*(npoints + 1.));
  xm    = 0.5*(max+min);
  xl    = 0.5*(max-min);
  for (int i=1; i<=m; ++i) {
    z = std::cos(kPi*(i-0.25)/(npoints+0.5));
    do {
      p1 =1.0;
      p2 =0.0;
      for (int j=1; j<=npoints; ++j) {
        p3 = p2;
        p2 = p1;
        p1 = ((2.0*j-1.0)*z*p2-(j-1.0)*p3)/(j);
      }
      pp = npoints*(z*p1-p2)/(z*z-1.0);
      z1 = z;
      z  = z1-p1/pp;
    } while (std::fabs(z-z1)>kEpsilon);
    abscissas[i-1]          = xm-xl*z;
    abscissas[npoints+1-i-1] = xm+xl*z;
    weights[i-1]            = 2.0*xl/((1.0-z*z)*pp*pp);
    weights[npoints+1-i-1]  = weights[i-1];
  }                  
}


void G4HepEmInitUtils::PrepareSpline(int npoint, double* xdata, double* ydata, double* secderiv) {
  int     n   = npoint-1;
  double *u   = new double[n];
  double  p   = 0.0;
  double  sig = 0.0;

  u[1] = ((ydata[2]-ydata[1])/(xdata[2]-xdata[1]) - (ydata[1]-ydata[0])/(xdata[1]-xdata[0]));
  u[1] = 6.0*u[1]*(xdata[2]-xdata[1]) / ((xdata[2]-xdata[0])*(xdata[2]-xdata[0]));

  // Decomposition loop for tridiagonal algorithm. secderiv[i]
  // and u[i] are used for temporary storage of the decomposed factors.
  secderiv[0] = 0.0;
  secderiv[1] = (2.0*xdata[1]-xdata[0]-xdata[2]) / (2.0*xdata[2]-xdata[0]-xdata[1]);
  for (int i=2; i<n-1; ++i) {
    sig = (xdata[i]-xdata[i-1]) / (xdata[i+1]-xdata[i-1]);
    p   = sig*secderiv[i-1] + 2.0;
    secderiv[i] = (sig - 1.0)/p;
    u[i] = (ydata[i+1]-ydata[i])/(xdata[i+1]-xdata[i]) - (ydata[i]-ydata[i-1])/(xdata[i]-xdata[i-1]);
    u[i] = (6.0*u[i]/(xdata[i+1]-xdata[i-1])) - sig*u[i-1]/p;
  }

  sig    = (xdata[n-1]-xdata[n-2]) / (xdata[n]-xdata[n-2]);
  p      = sig*secderiv[n-3] + 2.0;
  u[n-1] = (ydata[n]-ydata[n-1])/(xdata[n]-xdata[n-1]) - (ydata[n-1]-ydata[n-2])/(xdata[n-1]-xdata[n-2]);
  u[n-1] = 6.0*sig*u[n-1]/(xdata[n]-xdata[n-2]) - (2.0*sig - 1.0)*u[n-2]/p;

  p      = (1.0+sig) + (2.0*sig-1.0)*secderiv[n-2];
  secderiv[n-1] = u[n-1]/p;

  // The back-substitution loop for the triagonal algorithm of solving
  // a linear system of equations.
  for (int k=n-2; k>1; --k) {
    secderiv[k] *= (secderiv[k+1] - u[k]*(xdata[k+1]-xdata[k-1])/(xdata[k+1]-xdata[k]));
  }
  secderiv[n]  = (secderiv[n-1] - (1.0-sig)*secderiv[n-2])/sig;
  sig          = 1.0 - ((xdata[2]-xdata[1])/(xdata[2]-xdata[0]));
  secderiv[1] *= (secderiv[2] - u[1]/(1.0-sig));
  secderiv[0]  = (secderiv[1] - sig*secderiv[2])/(1.0-sig);

  // delete auxilary array
  delete[] u;  
}                


void G4HepEmInitUtils::PrepareSpline(int npoint, double* xdata, double* ydata) {
  double* secderiv = new double[npoint];
  double*        y = new double[npoint];
  for (int i=0; i<npoint; ++i) {
    y[i] = ydata[2*i];
  }
  PrepareSpline(npoint, xdata, ydata, secderiv);
  for (int i=0; i<npoint; ++i) {
    ydata[2*i+1] = secderiv[i];
  }
  delete[] secderiv;
  delete[] y;
}


// use the improved, robust spline interpolation that I put in G4 10.6
double G4HepEmInitUtils::GetSplineLog(int ndata, double* xdata, double* ydata, double* secderiv, double x, double logx, double logxmin, double invLDBin) {
  // make sure that $x \in  [x[0],x[ndata-1]]$
  const double xv = std::max(xdata[0], std::min(xdata[ndata-1], x));
  // compute the lowerindex of the x bin (idx \in [0,N-2] will be guaranted)
  const int   idx = (int)std::max(0., std::min((logx-logxmin)*invLDBin, ndata-2.));
  // perform the interpolation
  const double x1 = xdata[idx];
  const double x2 = xdata[idx+1];
  const double dl = x2-x1;
  // note: all corner cases of the previous methods are covered and eventually
  //       gives b=0/1 that results in y=y0\y_{N-1} if e<=x[0]/e>=x[N-1] or
  //       y=y_i/y_{i+1} if e<x[i]/e>=x[i+1] due to small numerical errors
  const double  b = std::max(0., std::min(1., (xv - x1)/dl));
  //
  const double os = 0.166666666667; // 1./6.
  const double  a = 1.0 - b;
  const double c0 = (a*a*a-a)*secderiv[idx];
  const double c1 = (b*b*b-b)*secderiv[idx+1];
  return a*ydata[idx] + b*ydata[idx+1] + (c0+c1)*dl*dl*os;  
}

// same as above but both ydata and secderiv are stored in ydata array
double G4HepEmInitUtils::GetSplineLog(int ndata, double* xdata, double* ydata, double x, double logx, double logxmin, double invLDBin) {
  // make sure that $x \in  [x[0],x[ndata-1]]$
  const double xv = std::max(xdata[0], std::min(xdata[ndata-1], x));
  // compute the lowerindex of the x bin (idx \in [0,N-2] will be guaranted)
  const int   idx = (int)std::max(0., std::min((logx-logxmin)*invLDBin, ndata-2.));
  const int  idx2 = 2*idx;
  // perform the interpolation
  const double x1 = xdata[idx];
  const double x2 = xdata[idx+1];
  const double dl = x2-x1;
  // note: all corner cases of the previous methods are covered and eventually
  //       gives b=0/1 that results in y=y0\y_{N-1} if e<=x[0]/e>=x[N-1] or
  //       y=y_i/y_{i+1} if e<x[i]/e>=x[i+1] due to small numerical errors
  const double  b = std::max(0., std::min(1., (xv - x1)/dl));
  //
  const double os = 0.166666666667; // 1./6.
  const double  a = 1.0 - b;
  const double c0 = (a*a*a-a)*ydata[idx2+1];
  const double c1 = (b*b*b-b)*ydata[idx2+3];
  return a*ydata[idx2] + b*ydata[idx2+2] + (c0+c1)*dl*dl*os;  
}


// same as above but all xdata, ydata and secderiv are stored in data array
double G4HepEmInitUtils::GetSplineLog(int ndata, double* data, double x, double logx, double logxmin, double invLDBin) {
  // make sure that $x \in  [x[0],x[ndata-1]]$
  const double xv = std::max(data[0], std::min(data[3*(ndata-1)], x));
  // compute the lowerindex of the x bin (idx \in [0,N-2] will be guaranted)
  const int   idx = (int)std::max(0., std::min((logx-logxmin)*invLDBin, ndata-2.));
  const int  idx3 = 3*idx;
  // perform the interpolation
  const double x1 = data[idx3];
  const double x2 = data[idx3+3];
  const double dl = x2-x1;
  // note: all corner cases of the previous methods are covered and eventually
  //       gives b=0/1 that results in y=y0\y_{N-1} if e<=x[0]/e>=x[N-1] or
  //       y=y_i/y_{i+1} if e<x[i]/e>=x[i+1] due to small numerical errors
  const double  b = std::max(0., std::min(1., (xv - x1)/dl));
  //
  const double os = 0.166666666667; // 1./6.
  const double  a = 1.0 - b;
  const double c0 = (a*a*a-a)*data[idx3+2];
  const double c1 = (b*b*b-b)*data[idx3+5];
  return a*data[idx3+1] + b*data[idx3+4] + (c0+c1)*dl*dl*os;  
}





// this is used for getting inverse-range on host
double G4HepEmInitUtils::GetSpline(double* xdata, double* ydata, double* secderiv, double x, int idx, int step) {
  // perform the interpolation
  const double x1 = xdata[step*idx];
  const double x2 = xdata[step*(idx+1)];
  const double dl = x2-x1;
  // note: all corner cases of the previous methods are covered and eventually
  //       gives b=0/1 that results in y=y0\y_{N-1} if e<=x[0]/e>=x[N-1] or
  //       y=y_i/y_{i+1} if e<x[i]/e>=x[i+1] due to small numerical errors
  const double  b = std::max(0., std::min(1., (x - x1)/dl));
  //
  const double os = 0.166666666667; // 1./6.
  const double  a = 1.0 - b;
  const double c0 = (a*a*a-a)*secderiv[idx];
  const double c1 = (b*b*b-b)*secderiv[idx+1];
  return a*ydata[idx] + b*ydata[idx+1] + (c0+c1)*dl*dl*os;  
}

// same as above but both ydata and secderiv are stored in ydata array
double G4HepEmInitUtils::GetSpline(double* xdata, double* ydata, double x, int idx) {
  const int  idx2 = 2*idx;
  // perform the interpolation
  const double x1 = xdata[idx];
  const double x2 = xdata[idx+1];
  const double dl = x2-x1;
  // note: all corner cases of the previous methods are covered and eventually
  //       gives b=0/1 that results in y=y0\y_{N-1} if e<=x[0]/e>=x[N-1] or
  //       y=y_i/y_{i+1} if e<x[i]/e>=x[i+1] due to small numerical errors
  const double  b = std::max(0., std::min(1., (x - x1)/dl));
  //
  const double os = 0.166666666667; // 1./6.
  const double  a = 1.0 - b;
  const double c0 = (a*a*a-a)*ydata[idx2+1];
  const double c1 = (b*b*b-b)*ydata[idx2+3];
  return a*ydata[idx2] + b*ydata[idx2+2] + (c0+c1)*dl*dl*os;  
}

// same as above but both xdata, ydata and secderiv are stored in data array
double G4HepEmInitUtils::GetSpline(double* data, double x, int idx) {
  const int  idx3 = 3*idx;
  // perform the interpolation
  const double x1 = data[idx3];
  const double x2 = data[idx3+3];
  const double dl = x2-x1;
  // note: all corner cases of the previous methods are covered and eventually
  //       gives b=0/1 that results in y=y0\y_{N-1} if e<=x[0]/e>=x[N-1] or
  //       y=y_i/y_{i+1} if e<x[i]/e>=x[i+1] due to small numerical errors
  const double  b = std::max(0., std::min(1., (x - x1)/dl));
  //
  const double os = 0.166666666667; // 1./6.
  const double  a = 1.0 - b;
  const double c0 = (a*a*a-a)*data[idx3+2];
  const double c1 = (b*b*b-b)*data[idx3+5];
  return a*data[idx3+1] + b*data[idx3+4] + (c0+c1)*dl*dl*os;  
}

// this is used to get index for inverse range on host
// NOTE: it is assumed that x[0] <= x and x < x[step*(num-1)]
// step: the delta with which   the x values are located in xdata (i.e. =1 by default)
int    G4HepEmInitUtils::FindLowerBinIndex(double* xdata, int num, double x, int step) {
  // Perform a binary search to find the interval val is in
  int ml = -1;
  int mu = num-1;
  while (std::abs(mu-ml)>1) {
    int mav = 0.5*(ml+mu);
    if (x<xdata[step*mav]) {  mu = mav; }
    else                   {  ml = mav; }
  }
  return mu-1;
}



