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

	size_t SharedMemSizeInElems;
    
};
#endif 
