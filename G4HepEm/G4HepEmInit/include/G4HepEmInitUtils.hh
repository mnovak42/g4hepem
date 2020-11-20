
#ifndef G4HepEmInitUtils_HH
#define G4HepEmInitUtils_HH

//
// Utility methods used during the initialisation.
//

class G4HepEmInitUtils {
public:
  // public access method
  static G4HepEmInitUtils& Instance();

  // delete copy CTR and assigment operators
  G4HepEmInitUtils(const G4HepEmInitUtils&) = delete;
  G4HepEmInitUtils& operator=(const G4HepEmInitUtils&) = delete;

  // Fills the input vector arguments with `npoints` GL integral abscissas and 
  // weights values (on [0,1] by default)
  void GLIntegral(int npoints, double* abscissas, double* weights, double min=0.0, double max=1.0);
                  
  
  
  // receives `npoint` x,y data arrays and fills the `secderiv` array with the 
  // second derivatives that can be used for a spline interpolation
  void   PrepareSpline(int npoint, double* xdata, double* ydata, double* secderiv);

  // same as above, but both the ydata and second derivatives are stored in the 
  // ydata array (compact [...,y_i, sd_i, y_{i+1}, sd_{i+1},...] with 2x`npoint`)
  void   PrepareSpline(int npoint, double* xdata, double* ydata);                


  // get spline interpolation over a log-spaced xgrid previously prepared by 
  // PrepareSpline (separate storrage of ydata and second deriavtive)  
  // use the improved, robust spline interpolation that I put in G4 10.6
  double GetSplineLog(int ndata, double* xdata, double* ydata, double* secderiv, double x, double logx, double logxmin, double invLDBin);

  // get spline interpolation over a log-spaced xgrid previously prepared by 
  // PrepareSpline (compact storrage of ydata and second deriavtive in ydata)  
  // use the improved, robust spline interpolation that I put in G4 10.6
  double GetSplineLog(int ndata, double* xdata, double* ydata, double x, double logx, double logxmin, double invLDBin);

  // get spline interpolation over a log-spaced xgrid previously prepared by 
  // PrepareSpline (compact storrage of xdata, ydata and second deriavtive in data)  
  // use the improved, robust spline interpolation that I put in G4 10.6
  double GetSplineLog(int ndata, double* data, double x, double logx, double logxmin, double invLDBin);


  // get spline interpolation over any xgrid: idx = i such  xdata[i] <= x < xdata[i+1]
  // and x >= xdata[0] and x<xdata[ndata-1]
  // PrepareSpline (separate storrage of ydata and second deriavtive)  
  // use the improved, robust spline interpolation that I put in G4 10.6
  double GetSpline(double* xdata, double* ydata, double* secderiv, double x, int idx, int step=1);

  // get spline interpolation if it was prepared with compact storrage of ydata 
  // and second deriavtive in ydata
  // use the improved, robust spline interpolation that I put in G4 10.6
  double GetSpline(double* xdata, double* ydata, double x, int idx);

  // get spline interpolation if it was prepared with compact storrage of xdata,
  // ydata and second deriavtive in data
  double GetSpline(double* data, double x, int idx);
  
  
  // finds the lower index of the x-bin in an ordered, increasing x-grid such 
  // that x[i] <= x < x[i+1]
  int    FindLowerBinIndex(double* xdata, int num, double x, int step=1);

                  
private:
  // private CTR
  G4HepEmInitUtils() {}

}; 

#endif //  G4HepEmInitUtils_HH