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
#define NN_DEBUG 1
#define NN_PRINT 1
#else
#define NN_PRINT 0
#define NN_DEBUG 0
#endif

using namespace CudaSimpleMatrix;
using namespace std::chrono;
using namespace CudaUtils;
using namespace std;

// TODO : 1. Backprop of layers that don't fit pgrads in smem
// TODO : 1.a Foward pass grid size can be larger than 111 and smaller than Numneurons,1,1
// TODO : 2. Batched updates , this applies to all of NeuralNetwork project
// TODO : 3. Move weights to Constant area.

CudaFullyConnectedLayer::CudaFullyConnectedLayer(unsigned numInputs, unsigned numNeurons, ActivationId actId) :
	NumNeurons(numNeurons),
    NumInputs(numInputs), Act(actId),
    Weights ({ NumInputs + 1, NumNeurons }),
	Results ({ NumNeurons, 1 }),
	Grads   ({ NumNeurons, 1 }),
	LGrads  ({ NumNeurons, 1 }),
    PGrads  ({ NumInputs, NumNeurons }),
    Input   ({ NumInputs, 1 })
{
	FwdPassKLP = { dim3(), VEC22DIM(Weights.size), Weights.size()*sizeof(float_t) };
	if( !(FwdSingleBlock = LaunchChecker::Check(FwdPassKLP)) )
		FwdPassKLP = { dim3(NumNeurons) , dim3(NumInputs + 1), (NumInputs + 1)*sizeof(float_t) };

	BwdPassKLP = { dim3(), dim3(NumInputs + 1, NumNeurons, 1), sizeof(float_t) * NumInputs * NumNeurons };
	
	if (!LaunchChecker::Check(FwdPassKLP) || !(BwdSingleBlock = LaunchChecker::Check(BwdPassKLP)))
		throw std::runtime_error("Layer too large for GPU");

}

CudaFullyConnectedLayer::~CudaFullyConnectedLayer()
{
	Weights.Clear();
	Results.Clear();
	LGrads.Clear();
	Grads.Clear();
	Input.Clear();
	PGrads.Clear();
}

__global__ void FC_forwardPass( unsigned numInputs, ActivationId act,  
                            const float_t* ip, float_t* wb, float_t* res, const bool packed)
{
    extern __shared__ float_t smem_g[];
    float_t* smem = smem_g;
    unsigned x = threadIdx.x, y = blockIdx.x;
    if (packed)
    {
        y = threadIdx.y;
        smem += y*blockDim.x;
    }

    wb += y*(numInputs + 1); 
    // what use cublasSgemm like a filthy pleb?
    if (x >  numInputs) smem[x] = 0; // padding
    else if (x == numInputs) smem[x] = wb[x]; // (?1L)
    else smem[x] = wb[x] * ip[x];  // 2L

    __syncthreads();

    unsigned xSize = CudaUtils::NextPowerOf2(blockDim.x);
#pragma unroll
    for (unsigned int s = xSize/2; s > 0; s >>= 1) { // horizontal reduction
        if (x < s) smem[x] += ( (x + s < numInputs +1) ? smem[x + s] : 0.);
        __syncthreads(); 
    }

    __syncthreads();
    
    if (x==0) res[y] = Activate(act, smem[0]); //1ST;
}

CudaSimpleMatrix::CudaMatrix<float_t> CudaFullyConnectedLayer::ForwardPass(float_t* input){

    Input.Copy(input);
   
    FC_forwardPass<<< EXPAND_KLP(FwdPassKLP) >>> 
        (NumInputs, Act, Input.devData, Weights.devData, Results.devData, FwdSingleBlock);
	
	CUDA_CHECK_SYNCH;

    return Results;
};

__global__ void FC_backwardPass(unsigned numInputs, unsigned numNeurons, ActivationId act, float_t eta,
    float_t* gd, const float_t* ip, const float_t* op, float_t* wb, float_t* pgds)
{
    extern __shared__ float_t smem[];

    unsigned x = threadIdx.x, y = threadIdx.y;
    
    float_t* pgd = smem + y * numInputs;
    wb += y*(numInputs + 1);  

    float_t  g = gd[y], o = op[y], w = wb[x], i = 0; // 4 LD , L2$ would thrash here.
    if(x < numInputs)  i = ip[x]; 
    
    g *= ActivatePrime(act, o); 

    float_t dw = g;
    if (x < numInputs) // not Bias
        pgd[x] = g * w, dw  *= i; 
    
    __syncthreads();
    
    w -= dw*eta;
    wb[x] = w;// 1ST

    unsigned ysize = CudaUtils::NextPowerOf2(blockDim.y);
//#pragma unroll
    for (unsigned int s = ysize / 2; s > 0; s >>= 1) { // vertical reduction
        if (y < s)
            pgd[x] += ((y + s) >= blockDim.y ? 0. : pgd[x + s * numInputs]);
        __syncthreads();
    }
    
    if (y == 0) pgds[x] = pgd[x]; //1ST
}

CudaSimpleMatrix::CudaMatrix<float_t> CudaFullyConnectedLayer::BackwardPass(float_t* backError, float_t eta)
{
    Grads.Copy(backError);

    FC_backwardPass <<<  EXPAND_KLP(BwdPassKLP) >>>
        (NumInputs, NumNeurons, Act, eta, Grads.devData, Input.devData, Results.devData, Weights.devData, PGrads.devData);

    CUDA_CHECK_SYNCH;

    return PGrads;
}
