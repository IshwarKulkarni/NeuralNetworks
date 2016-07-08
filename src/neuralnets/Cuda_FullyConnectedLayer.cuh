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

struct CudaNeuronBlock
{
    unsigned NumNeurons;
    unsigned NumInputs;// Not Bias

    ActivationId Act;
	CudaSimpleMatrix::CudaMatrix<double> Weights,  Results, Grads, LGrads, PGrads, Input;
	
    CudaNeuronBlock(unsigned numInputs, unsigned numNeurons, ActivationId actId);

	CudaSimpleMatrix::CudaMatrix<double> ForwardPass(double* input);

	CudaSimpleMatrix::CudaMatrix<double> BackwardPass(double* backError, double e);
    
	~CudaNeuronBlock();

	CudaUtils::KernelLaunchParams FwdPassKLP, BwdPassKLP;
	bool FwdSingleBlock, BwdSingleBlock;
    
};
#endif 
