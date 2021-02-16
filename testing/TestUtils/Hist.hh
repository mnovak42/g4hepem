
#ifndef HIST_HH
#define HIST_HH

// A simple histogram just for testing

#include <iostream>
#include <string>
#include <cstdio>

class Hist {
public:
  Hist(double min, double max, int numbin)
  {
    fMin     = min;
    fMax     = max;
    fNumBins = numbin;
    fDelta   = (fMax - fMin) / (numbin);
    fx       = new double[fNumBins];
    fy       = new double[fNumBins];
    for (int i = 0; i < fNumBins; ++i) {
      fx[i] = fMin + i * fDelta;
      fy[i] = 0.0;
    }
    fSum = 0.0;
  }

  Hist(double min, double max, double delta)
  {
    fMin     = min;
    fMax     = max;
    fDelta   = delta;
    fNumBins = (int)((fMax - fMin) / (delta)) + 1.0;
    fx       = new double[fNumBins];
    fy       = new double[fNumBins];
    for (int i = 0; i < fNumBins; ++i) {
      fx[i] = fMin + i * fDelta;
      fy[i] = 0.0;
    }
    fSum = 0.0;
  }

  void Fill(double x)
  {
    int indx = (int)((x - fMin) / fDelta);
    if (indx < 0) {
      std::cerr << "\n ***** ERROR in Hist::FILL  =>  x = " << x << " < fMin = " << fMin << std::endl;
      exit(1);
    }

    fy[indx] += 1.0;
  }

  void Fill(double x, double w)
  {
    int indx = (int)((x - fMin) / fDelta);
    if (indx < 0) {
      std::cerr << "\n ***** ERROR in Hist::FILL  =>  x = " << x << " < fMin = " << fMin << std::endl;
      exit(1);
    }

    fy[indx] += 1.0 * w;
  }


  void Write(const std::string& fname, double norm) {
    FILE *f = fopen(fname.c_str(), "w");
    for (int i = 0; i < GetNumBins(); ++i) {
      fprintf(f, "%d\t%.8g\t%.8g\n", i, GetX()[i] + 0.5 * GetDelta(), GetY()[i] * norm);
    }
    fclose(f);
  }


  int GetNumBins() const { return fNumBins; }
  double GetDelta() const { return fDelta; }
  double *GetX() const { return fx; }
  double *GetY() const { return fy; }

private:
  double *fx;
  double *fy;
  double fMin;
  double fMax;
  double fDelta;
  double fSum;
  int fNumBins;
};

#endif //HIST_HH
