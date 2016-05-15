#ifndef __CUDA_OPERATORS_CUH__
#define __CUDA_OPERATORS_CUH__

#include "cuda_runtime.h"
#include <cfloat>
#include "utils/CudaUtils.hxx"

enum Oper
{
    OperAdd, 
    OperSub, 
    OperMul,
    OperDiv,
    OperPow
};

template<typename T, Oper op>
struct CuAlgebraicOperScalar
{
    T Scalar;
    __host__ CuAlgebraicOperScalar(T s):Scalar(s){}
    
    template <Oper op> __forceinline__ __device__ T Operation(T t, T u) const;
    template<> __forceinline__ __device__ T Operation<OperAdd>(T t, T u) const {return t + u; }
    template<> __forceinline__ __device__ T Operation<OperSub>(T t, T u) const {return t - u; }
    template<> __forceinline__ __device__ T Operation<OperMul>(T t, T u) const {return t * u; }
    template<> __forceinline__ __device__ T Operation<OperDiv>(T t, T u) const {return t / u; }
    template<> __forceinline__ __device__ T Operation<OperPow>(T t, T u) const { return std::pow(t,u); } 
                                                                         
    __forceinline__ __device__  T Apply(uint x, uint y, T Val) const
    {
        return Operation<op>(Val,Scalar);
    }
};

template<typename T, Oper op>
struct CuAlgebraicOper1Tex
{
    cudaTextureObject_t T1;
    CuAlgebraicOper1Tex(cudaTextureObject_t t1):T1(t1){} // we sdhould actually accept matrix and fetch tex, because owners should destroy

    template <Oper op> __forceinline__ __device__ T Operation(T t, T u) const;
    template<> __forceinline__ __device__ T Operation<OperAdd>(T t, T u) const { return t + u; }
    template<> __forceinline__ __device__ T Operation<OperSub>(T t, T u) const { return t - u; }
    template<> __forceinline__ __device__ T Operation<OperMul>(T t, T u) const { return t * u; }
    template<> __forceinline__ __device__ T Operation<OperDiv>(T t, T u) const { return t / u; }
    template<> __forceinline__ __device__ T Operation<OperPow>(T t, T u) const { return std::pow(t,u); } 

    __forceinline__ __device__ T Apply(uint x, uint y, T Val) const
    {
        T val1;
        tex2D(&val1, T1, x, y);
        return Operation<op>(Val,val1);
    }

    ~CuAlgebraicOper1Tex()
    {
        cudaDestroyTextureObject(T1);
    }

};

#define CUDA_CONSTANT_SIZE (512)
static __constant__ float CUDA_CONST_AREA[CUDA_CONSTANT_SIZE];

template<typename T>
__inline__ __device__ T fetch2D(uint x, uint y,cudaTextureObject_t MatTex)
{
    int4 v=tex1Dfetch(tex_x_double2, i);

    return make_double2(__hiloint2double(v.y, v.x),__hiloint2double(v.w, v.z));
} 

__device__ __inline__
void tex2D(Color* val, cudaTextureObject_t cudaTexMatTex, uint imX, uint imY)
{
}

template<typename T>
struct ConvolutionOperator
{
    uint KernelHeight, KernelWidth, KernelSize;
    uint  Height, Width;
    size_t Pitch;
    cudaTextureObject_t MatTex;
    float* KernelCostPtr;  
    // TODO : this ^ will leak! If I put cudaFree in desctructor: 
    //      1> When this operator is copied to device, if the host object dies, this will be freed.
    //      2> cudaFree() will free entire CUDA_CONSTANT_SIZE byte chunk.
    // I need a far more sophisticated memory module for constants.
     
    ConvolutionOperator(uint elemPitch, uint w, uint h, cudaTextureObject_t tex,const Matrix2D<float>& kernel):
        KernelHeight (kernel.Height()),
        KernelWidth  (kernel.Width()),
        Pitch        (elemPitch),
        Width         (w),
        Height       (h),
        MatTex       (tex),
        KernelSize   (KernelHeight*KernelWidth)
    {
        uint kernelSize = sizeof(float)*KernelHeight*KernelWidth ;
        THROW_IF( kernelSize  >= CUDA_CONSTANT_SIZE, CUDASizeException, "The kernel does not fit in the constant area, change implem");

        cudaMemcpyToSymbol(CUDA_CONST_AREA,kernel.GetData(), kernelSize);
    } 
    
    __forceinline__ __device__ 
    void Apply(uint x, uint y, T* result) const 
    {
        T sum(0);
        for ( uint kx = 0; kx < KernelWidth;  ++kx )
        for ( uint ky = 0; ky < KernelHeight; ++ky )
            {
                uint imX = x+ kx - KernelWidth/2, imY = y + ky - KernelHeight/2;
                uint kp = IMAD(KernelWidth,ky,kx);
                T val = 0;
                tex2D(&val, MatTex, imX, imY);
                sum += (val * CUDA_CONST_AREA[kp]);
            }
        result[x + Pitch*y] = sum;
    }
};

static __constant__ __device__ Color    InterpolateColors[32];
static __device__   Color    Interpolate(float value, float minElement, float range, uint numColors )
{
    --numColors; // now this is numer intervals
    float     relative = (value - minElement)/range ;
    uint    index     = (uint) (relative * numColors); // floor
    relative -=  float(index)/numColors; relative *= numColors;
     return (InterpolateColors[index]*(1.f-relative) + InterpolateColors[index+1]*(relative));
}

static Color StdPallete[] = { Black, Blue, Yellow, Red, White };


template<typename T>
struct FractalOperator
{
    float2 Start;
    uint Width, Height;
    float deltaR, deltaI ;
    short MaxIters;
    uint NumColors;

    FractalOperator(float2 s, float2 e, cudaPitchedPtr ptr, uint max_iters, Color* pallete = StdPallete, uint numColors = ARRAY_LENGTH(StdPallete) ):
        Start(s),
        Width(ptr.pitch/sizeof(T)),
        Height(ptr.ysize),
        MaxIters(max_iters),
        deltaR( (s.x - e.x) / ptr.xsize),
        deltaI( (s.y - e.y) / ptr.ysize),
        NumColors(numColors)
    {
        
    }

    FractalOperator(float2 s, float2 e, uint width, uint height, uint max_iters, Color* pallete = StdPallete, uint numColors = 4/*ARRAY_LENGTH(StdPallete)*/ ):
        Start(s),
        Width(width),
        Height(height),
        MaxIters(max_iters),
        deltaR ( (s.x - e.x) / width),
        deltaI ( (s.y - e.y) / height),
        NumColors(numColors)
    {
        Color sp[] =  { Black, Blue, Red, Orange, White};
        NumColors = ARRAY_LENGTH(sp);
        cudaMemcpyToSymbol(InterpolateColors, sp, sizeof(sp));
    } 


    __device__ 
    void Apply(uint x, uint y, T* result) const 
    {
        float2 zn = Start;
        zn.x -= (x * deltaR) , zn.y -= (y * deltaI);
        float2 c = zn;
        uint iter = 0 ;
        do
        {
            float t = zn.x*zn.x - zn.y*zn.y + c.x;
            zn.y = 2*zn.x*zn.y + c.y;
            zn.x = t;

        }while(++iter < MaxIters  && ( zn.x*zn.x + zn.y*zn.y < 50)) ;

        result[x + Width*y] = Interpolate(iter%MaxIters, 0, MaxIters, NumColors);
    }

};

#endif