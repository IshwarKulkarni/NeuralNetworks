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
	Results({ OpSz.x, OpSz.y*OpSz.z })
{              //blockIdx = OpIdx
	FwdPassKLP = { VEC32DIM(OpSz), dim3(1,KSz.y, KSz.z), KSz.y*KSz.z*sizeof(float_t) };

	if (!LaunchChecker::Check(FwdPassKLP) )
		throw std::runtime_error("Kernels too big/numerous for launching");

	//if (!IpInConst && LaunchChecker::ConstCheck(OpSz.z*(KSz()  + 1 )) ) 
	//	throw std::runtime_error("Kernels too big/numerous for constant mem size");

	if (!IpInConst)throw std::runtime_error("Kernels too big/numerous for constant mem size");
}

template<bool Padded>
__global__ void forwardPass(
    Size3 kSz, Size3 ipSz, 
    Size3 opSz, Size2 stride, ActivationId act, 
    float_t* weights, float_t* ip, float_t* res)
{
	unsigned y = threadIdx.y, z = threadIdx.z;
	extern __shared__ float_t smem[];

    unsigned KernelNum = blockIdx.z; 
    
    float_t prod = 0, bias = weights[KernelNum * (kSz() + 1) + kSz()];

    weights += KernelNum * (kSz() + 1) + LinOffset(kSz, { 0, y, z });
    res     +=  LinOffset(opSz, { blockIdx.x, blockIdx.y, KernelNum });
    

    int ipy = blockIdx.y * stride.y + y, ipx = blockIdx.x * stride.x; // ipz = z;
    if (Padded) ipy -= kSz.y / 2, ipx -= kSz.x / 2;

    if (ipy >= 0 && ipy < ipSz.y) { 

        unsigned xs = 0, xe = kSz.x;
        if (ipx < 0) xs = -ipx;
        if (ipx + xe >= ipSz.x) xe = ipSz.x - ipx;
        
        ip += LinOffset(ipSz, {size_t(MAX(ipx,0)), size_t(ipy), z});
#pragma unroll
        for (unsigned x = xs; x < xe; ++x, ++ip, ++weights)
            prod += (*ip * *weights);
    }
    smem[y*kSz.z + z] = prod;

    __syncthreads();

    unsigned zsize = CudaUtils::NextPowerOf2(blockDim.z); // fold z plane in half // sideways reduction
    for (unsigned int s = zsize / 2; s > 0; s >>= 1) {
        if (z < s)
            smem[z + y * blockDim.z] += ((z + s < blockDim.z) ? smem[y*blockDim.z + z +s] : 0);
        __syncthreads();
    }

    if (y == 0 && z == 0)
    {
        float_t dot = smem[0] + bias;
        for (unsigned i = 1; i < blockDim.y; ++i)
            dot += smem[i * blockDim.z];

        *res = Activate(act, dot);
    }
}

CudaSimpleMatrix::CudaMatrix<float_t> CudaConvolutionLayer::ForwardPass(float_t* input)
{
    Input.Copy(input);
    
    forwardPass <true> <<< EXPAND_KLP(FwdPassKLP) >>> (KSz, IpSz, OpSz, Stride, Act, 
        Kernels.devData, Input.devData, Results.devData);

    CUDA_CHECK_SYNCH;
	return Results;
}


CudaConvolutionLayer::~CudaConvolutionLayer()
{
	Kernels .Clear();
	Results	.Clear();
	Input	.Clear();
}