/*
Copyright (c) Ishwar R. Kulkarni
All rights reserved.

This file is part of NeuralNetwork Project by
Ishwar Kulkarni , see https://github.com/IshwarKulkarni/NeuralNetworks

If you so desire, you can copy, redistribute and/or modify this source
along with  rest of the project. However any copy/redistribution,
including but not limited to compilation to binaries, must carry
this header in its entirety. A note must be made about the origin
of your copy.

NeuralNetwork is being distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
FITNESS FOR A PARTICULAR PURPOSE.

*/

#include "CudaLayers.cuh"

#ifdef _DEBUG
#define CNN_DEBUG 1
#define CNN_PRINT 1
#else
#define CNN_PRINT 0
#define CNN_DEBUG 0
#endif

using namespace CudaSimpleMatrix;
using namespace std::chrono;
using namespace CudaUtils;
using namespace Logging;
using namespace std;
using namespace Vec;

CudaConvolutionLayer::CudaConvolutionLayer(
    Size3 kSize, unsigned numKernels, 
    Size3 ipSize, Size3 opSize, Size2 stride, ActivationId act) :
	KSz(kSize),
	IpSz(ipSize),
    Input(IpSz),
	OpSz(opSize),
	Stride(stride),
	NumKernels(numKernels),
	Act(act),
    Kernels({ (KSz() + 1)*numKernels, 1 }),
	Results({ OpSz.x, OpSz.y*OpSz.z }),
    Grads({ OpSz.x, OpSz.y*OpSz.z }), PGrads({ IpSz})
{              //blockIdx = OpIdx
	FwdPassKLP = { VEC32DIM(OpSz), dim3(1,KSz.y, KSz.z), KSz.y*KSz.z*sizeof(float_t) };

    kSize.x = iDivUp(kSize.x, stride.x), kSize.y = iDivUp(kSize.y, stride.y);
    BwdPassKLP = { VEC32DIM(IpSz), VEC32DIM(kSize), kSize()*sizeof(float_t) };

	if (!LaunchChecker::Check(FwdPassKLP) || !LaunchChecker::Check(BwdPassKLP))
		throw std::runtime_error("Kernels too big/numerous for launching");

	//if (!IpInConst && LaunchChecker::ConstCheck(OpSz.z*(KSz()  + 1 )) ) 
	//	throw std::runtime_error("Kernels too big/numerous for constant mem size");

	if (!IpInConst)
        throw std::runtime_error("Kernels too big/numerous for constant mem size");
}

__device__ void sumByZ()
{
    extern __shared__ float_t smem[];
    __syncthreads();

    unsigned zsize = CudaUtils::NextPowerOf2(blockDim.z) / 2;
    for (unsigned int s = zsize; s > 0; s >>= 1) {
        if (threadIdx.z < s)
        {
            smem[threadIdx.z + threadIdx.y * blockDim.z] +=
                ((threadIdx.z + s < blockDim.z) ? smem[threadIdx.y*blockDim.z + threadIdx.z + s] : 0);
            __syncthreads();
        }
    }

    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0)
    {
        float_t sum = 0;
#pragma unroll
        for (unsigned i = 0; i < blockDim.x*blockDim.y; ++i) // TODO : is there a point in doing this parallely
            sum += smem[i];

        smem[0] = sum;
    }
    __syncthreads();
}

__global__ 
void Conv_forwardPass( Size3 kSz, Size3 ipSz, Size2 stride, ActivationId act, bool Padded, float_t* weights, float_t* ip, float_t* res)
{
	unsigned y = threadIdx.y, z = threadIdx.z;
	extern __shared__ float_t smem[];

    unsigned KernelNum = blockIdx.z; 
    
    float_t prod = 0, bias = weights[KernelNum * (kSz() + 1) + kSz()];

    weights += KernelNum * (kSz() + 1) + LinOffset(kSz, { 0, y, z });
    
    int ipy = blockIdx.y * stride.y + y, ipx = blockIdx.x * stride.x; // ipz = z;
    if (Padded) ipy -= kSz.y / 2, ipx -= kSz.x / 2;

    if (ipy >= 0 && ipy < ipSz.y) { 

        unsigned xs = 0, xe = kSz.x;
        if (ipx < 0) xs = -ipx;
        if (ipx + xe >= ipSz.x) xe = ipSz.x - ipx;
        
        unsigned ioff = LinOffset(ipSz, { size_t(MAX(ipx,0)), size_t(ipy), z });
        ip += ioff;
#pragma unroll
        for (unsigned x = xs; x < xe; ++x, ++ip, ++weights)
            prod += (*ip * *weights);
    }
    smem[y*kSz.z + z] = prod;

    sumByZ();

    if (y == 0 && z == 0)
        res[LinOffset(gridDim, { blockIdx.x, blockIdx.y, KernelNum })] = Activate(act, smem[0] + bias);
}

CudaSimpleMatrix::CudaMatrix<float_t> CudaConvolutionLayer::ForwardPass(float_t* input, bool Padded)
{
    Input.Copy(input);
    
    Conv_forwardPass<<<EXPAND_KLP(FwdPassKLP)>>> (KSz, IpSz, Stride, Act, Padded, Kernels.devData, Input.devData, Results.devData);

    CUDA_CHECK_SYNCH;
	return Results;
}

__global__ 
void Conv_makeGrads(float_t* grads, float_t * op, ActivationId act)
{
    unsigned i = threadIdx.x + blockDim.x * blockIdx.x;
    grads[i] *= ActivatePrime(act, op[i]);
}

__global__ 
void Conv_backwardPass(float_t* grads, float_t* weights, Size3 kSz, Size3 gSize, bool padded, Size2 stride, float_t* pGrads)
{
    Size3 k = DIM32SIZE(threadIdx);
    k *= stride; //k += Size3(1, 1, 0);
    unsigned z = blockIdx.z;
    weights += (kSz()* z);

    extern __shared__ float_t smem[];

    unsigned wOffset = LinOffset(kSz, k);
    
    Size3 g = { (blockIdx.x - k.x)/stride.x + padded*(kSz.x / 2) , (blockIdx.y - k.y) / stride.y + padded*(kSz.y / 2), z };
    
    if (g.x < gSize.x && g.y < gSize.y && g.z < gSize.z)
        smem[wOffset] = grads[LinOffset(gSize, g)] * weights[wOffset];
    else
        smem[wOffset] = 0; 

    if (blockIdx.x == gridDim.x / 2 && blockIdx.y == gridDim.y / 2)
        printf("%d %d\t%d %d\t%d %d | stride: %d,%d \n", blockIdx.x, blockIdx.y, k.x, k.y, g.x, g.y, stride.x, stride.y);

    sumByZ();
    if (k.x == 0 && k.y == 0 && k.z == 0)
        pGrads[LinOffset(DIM32SIZE(gridDim), DIM32SIZE(blockIdx))] = smem[0];
}

CudaSimpleMatrix::CudaMatrix<float_t> CudaConvolutionLayer::BackwardPass(float_t* backError, float_t e, bool Padded, float_t* grads)
{
    Grads.Copy(backError);
    
    Conv_makeGrads <<< Results.size.y, Results.size.x >>> (Grads.devData, Results.devData, Act);

    Conv_backwardPass<<< EXPAND_KLP(BwdPassKLP) >>>(
        Grads.devData, Kernels.devData, KSz, Results.size, Padded, Stride, PGrads.devData);

    CUDA_CHECK_SYNCH;

    return PGrads;
}


CudaConvolutionLayer::~CudaConvolutionLayer()
{
	Kernels .Clear();
	Results	.Clear();
	Input	.Clear();
}