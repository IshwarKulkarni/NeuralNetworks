#include "LinearAlgebra.hxx"
#include "Regression.hxx"
#include "DataSets.hxx"
#include "imageprocessing/ImageGeneric.hxx"
#include "MINST-ImageReader.hxx"
#include "utils/SimpleMatrix.hxx"


#include <iostream>
#include <map>
#include <vector>
#include <array>
#include <cmath>

using namespace std;
using namespace Utils;
using namespace PPMIO;
using namespace Logging;
using namespace StringUtils;
using namespace SimpleMatrix;


float ValidationRation = 0.03f, TestRatio = 0.05f;

void main()
{
    Log << setprecision(7);
    Log.Tee(cout);
    Log.ResetIntervealLogging(false);

    srand(int(time(0)));
    
    const unsigned N = 3; // = degree + 1
    const double alpha = 0.01;
    const double stoppingAccuracy = 0.03; //3%;
    const unsigned MaxIter = 500;
    const unsigned M = 400;


    array<double, 3> OpCoeff, IpCoeff = { 3,2,4 }; // sum(Coeff[i]^i*x)

    auto Fire = [&](double ip, array<double, N>::iterator b, array<double, N>::iterator e) {
        double out = 0;
        unsigned i = 0;

        for (auto it = b; it != e; ++i)
            out += ip * pow(*it, i),
            ++i;

        return out;
    };
    
    std::vector<double> Ipts(M);
    std::generate(Ipts.begin(), Ipts.end(), Rand(10, -10));
    
    unsigned Iter = 0;
    do
    {
        for (auto& x : Ipts)
        {
            double hx = Fire(x, OpCoeff.begin(), OpCoeff.end()),
                y = Fire(x, IpCoeff.begin(), IpCoeff.end());

            double diff = hx - y;
        }
        
    } while (
        Iter++ < MaxIter &&
        !std::equal(IpCoeff.begin(), IpCoeff.end(), OpCoeff.begin(), [&](double d1, double d2)
    { return abs(2*(d1 - d2)/(d1 + d2)) < stoppingAccuracy; }));



}
