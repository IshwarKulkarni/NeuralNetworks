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

#include "Cuda_FullyConnectedLayer.cuh"

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

CudaNeuronBlock::CudaNeuronBlock(unsigned numInputs, unsigned numNeurons, ActivationId actId) :
	NumNeurons(numNeurons),
    NumInputs(numInputs),
    Act(actId),
    Weights ({ NumInputs + 1, NumNeurons }),
	Results ({ NumNeurons, 1 }),
	Grads   ({ NumNeurons, 1 }),
	LGrads  ({ NumNeurons, 1 }),
    PGrads  ({ NumInputs, NumNeurons }),
    Input   ({ NumInputs, 1 })
{
	FwdPassKLP = { dim3(), VEC22DIM(Weights.size), Weights.size()*sizeof(double) };
	if( !(FwdSingleBlock = LaunchChecker::Check(FwdPassKLP)) )
		FwdPassKLP = { dim3(NumNeurons) , dim3(NumInputs + 1), (NumInputs + 1)*sizeof(double) };

	BwdPassKLP = { dim3(), dim3(NumInputs + 1, NumNeurons, 1), sizeof(double) * NumInputs * NumNeurons };
	
	if (!LaunchChecker::Check(FwdPassKLP) || !(BwdSingleBlock = LaunchChecker::Check(BwdPassKLP)))
		throw std::runtime_error("Layer too large for GPU");

}

CudaNeuronBlock::~CudaNeuronBlock()
{
	Weights.Clear();
	Results.Clear();
	LGrads.Clear();
	Grads.Clear();
	Input.Clear();
	PGrads.Clear();
}

__global__ void forwardPass( unsigned numInputs, ActivationId act,  
                            const double* ip, double* wb, double* res, const bool packed)
{
    extern __shared__ double smem_g[];
    double* smem = smem_g;
    unsigned x = threadIdx.x, y = blockIdx.x;
    if (packed)
    {
        y = threadIdx.y;
        smem += y*blockDim.x;
    }

    wb += y*(numInputs + 1); 
    
    if (x >  numInputs) smem[x] = 0; // padding
    else if (x == numInputs) smem[x] = wb[x]; // (?1L)
    else smem[x] = wb[x] * ip[x];  // 2L

    __syncthreads();

    unsigned xSize = CudaUtils::NextPowerOf2(blockDim.x);
    for (unsigned int s = xSize/2; s > 0; s >>= 1) { // horizontal reduction
        if (x < s) smem[x] += ( (x + s < numInputs +1) ? smem[x + s] : 0.);
        __syncthreads(); 
    }

    __syncthreads();
    
    if (x==0) res[y] = Activate(act, smem[0]); //1ST;
}

CudaSimpleMatrix::CudaMatrix<double> CudaNeuronBlock::ForwardPass(double* input){

    Input.Copy(input);
   
    forwardPass<<< EXPAND_KLP(FwdPassKLP) >>> 
        (NumInputs, Act, Input.devData, Weights.devData, Results.devData, FwdSingleBlock);
	
	CUDA_CHECK_SYNCH;

    return Results;
};

__global__ void backwardPass(unsigned numInputs, unsigned numNeurons, ActivationId act, double eta,
    double* gd, const double* ip, const double* op, double* wb, double* pgds)
{
    extern __shared__ double smem[];

    unsigned x = threadIdx.x, y = threadIdx.y;
    
    double* pgd = smem + y * numInputs;
    wb += y*(numInputs + 1);  

    double  g = gd[y], o = op[y], w = wb[x], i = ip[x]; // 4 LD , L2$ thrashes here.
    
    g *= ActivatePrime(act, o); 

    double dw = g;
    if (x < numInputs) // not Bias
        pgd[x] = g * w, dw  *= i; 
    
    __syncthreads();
    
    w -= dw*eta;
    wb[x] = w;// 1ST

    unsigned ysize = CudaUtils::NextPowerOf2(blockDim.y);
    for (unsigned int s = ysize / 2; s > 0; s >>= 1) { // vertical reduction
        if (y < s)
            pgd[x] += ((y + s) >= blockDim.y ? 0. : pgd[x + s * numInputs]);
        __syncthreads();
    }
    
    if (y == 0) pgds[x] = pgd[x]; //1ST
}

CudaSimpleMatrix::CudaMatrix<double> CudaNeuronBlock::BackwardPass(double* backError, double eta)
{
    Grads.Copy(backError);
  
    backwardPass <<<  EXPAND_KLP(BwdPassKLP) >>>
        (NumInputs, NumNeurons, Act, eta, Grads.devData, Input.devData, Results.devData, Weights.devData, PGrads.devData);
    
	CUDA_CHECK_SYNCH;

    return PGrads;
}
