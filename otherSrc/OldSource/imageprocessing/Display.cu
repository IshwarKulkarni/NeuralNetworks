
#include <glew.h>
#include "Color.hxx"
#include "Filter.hxx"
#include "ColorImage.hxx"
#include "ImageProcessing.hxx"
#include "CudaMatrix.cuh"
#include "CudaDisplay.hxx"
#include "ImageGeneric.hxx"

int main2(int argc, char** argv)
{
    assert( Orange ==  (Color)( (uint) (Orange) ) );

    CudaDisplay::DisplayState state(512,512);

    CudaDisplay::StartCUDADisplay( state ,argc, argv);


    float2 topLeft, bottomRight;
    topLeft.x = -2.5, topLeft.y = 1.1;
    bottomRight.x = 1, bottomRight.y = -1.1;

    uint w = 1024 , h = 768;

    FractalOperator<uint> mandelbrot(topLeft, bottomRight, w, h, 256);

    LaunchConfig(w,h,4);

    cudaPitchedPtr ptr;
    ptr.xsize = w; ptr.ysize = h;

    cudaPitchedAllocCopy<uint>(ptr);
    cudaMemset2D(ptr.ptr,ptr.pitch,0,ptr.xsize,ptr.ysize);

    auto* devFun = cudaAllocCopy(&mandelbrot);

    Create <<<gridSize,blockSize>>> (devFun, w, h, (uint*)ptr.ptr);
    CUDA_CHECK();

    auto hPtr = cudaCopyOut<uint>(ptr);
    CUDA_CHECK();


    ColorVector colorVec(hPtr, hPtr+ ptr.xsize*ptr.ysize);

    ImageIO::WriteImage("Fractal",  make_pair(ptr.xsize,ptr.ysize), 3,  (uchar*)(&(colorVec[0])),  true,  ImageIO::PPM);

    delete[] hPtr;
    cudaDeviceReset();
    return 0;
}