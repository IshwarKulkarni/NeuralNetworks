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

#ifndef __CUDA_FULLY_CONNECTED_LAYER_INCLUDED__
#define __CUDA_FULLY_CONNECTED_LAYER_INCLUDED__

#include <string>

#include "utils/CudaSimpleMatrix.cuh"
#include "Activation.hxx"

struct CudaFullyConnectedLayer
{
    unsigned NumNeurons;
    unsigned NumInputs;// Not Bias

    ActivationId Act;
	CudaSimpleMatrix::CudaMatrix<float_t> Weights,  Results, Grads, LGrads, PGrads, Input;
	
    CudaFullyConnectedLayer(unsigned numInputs, unsigned numNeurons, ActivationId actId);

	CudaSimpleMatrix::CudaMatrix<float_t> ForwardPass(float_t* input);

	CudaSimpleMatrix::CudaMatrix<float_t> BackwardPass(float_t* backError, float_t e);
    
	~CudaFullyConnectedLayer();

	CudaUtils::KernelLaunchParams FwdPassKLP, BwdPassKLP;
	bool FwdSingleBlock, BwdSingleBlock;
    
};


struct CudaConvolutionLayer
{
	Vec::Size3 KSz, IpSz, OpSz;
	Vec::Size2 Stride;

	unsigned NumKernels; ActivationId Act;

    CudaConvolutionLayer(Vec::Size3 kSize, unsigned numKernels, Vec::Size3 ipSize, Vec::Size3 opSize, Vec::Size2 Stride, ActivationId act);

	~CudaConvolutionLayer();

	CudaSimpleMatrix::CudaMatrix<float_t> Kernels, Results, Input;

	CudaSimpleMatrix::CudaMatrix<float_t> ForwardPass(float_t* input);

	CudaUtils::KernelLaunchParams FwdPassKLP, BwdPassKLP;
	bool IpInConst;
};
#endif 
