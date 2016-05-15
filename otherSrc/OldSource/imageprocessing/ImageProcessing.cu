#include "Logging.hxx"
#include "Matrix.hxx"
#include "ImageGeneric.hxx"
#include "ImageProcessing.hxx"
#include "CudaMatrix.cuh"
#include <iostream>
#include <fstream>

using namespace std;
using namespace Logging;

typedef ImageGeneric<Matrix2D<float>> FloatImage;
typedef ImageGeneric<CuMatrix2D<float>> CuFloatImage;

void run(char* name_cstr)
{
    cudaSetDevice(0);
    const auto& name = StringUtils::establishEndsWith(name_cstr,".ppm" );
    
    ImageProcessing::EdgeEnhance(FloatImage(name)).Write("imEdE");
    
    CuFloatImage cuImg(name);
    const auto& edges = ImageProcessing::EdgeEnhance(cuImg);

    ColorVector vec;
    GenerateHeatMap(*edges.GetChannel(0), vec).WriteAsImage("heatmap");


    //Log    << cuEdE(0).GetMaxElem() << LogFlush() <<  LogEndl
    //        << cuEdE(0).GetMaxElem() << LogEndl ;
}

int main(int args, char** argv)
{
    run(argv[1]);
    //CHECK_UNDEAD_PTRS();
    return 0;
}
