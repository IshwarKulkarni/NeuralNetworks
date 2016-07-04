#ifndef __CUDA_FULLY_CONNECTED_LAYER_INCLUDED__
#define __CUDA_FULLY_CONNECTED_LAYER_INCLUDED__

#include <string>

#include "utils/CudaSimpleMatrix.cuh"
#include "Activation.hxx"

struct CudaNeuronBlock
{
    ActivationId Act;
    CudaSimpleMatrix::CudaMatrix<double> Weights, Results, LGrads;

	CudaNeuronBlock(Vec::Size2 WeightSize, ActivationId actId);

	CudaSimpleMatrix::CudaMatrix<double> Fire(double* input);
    
	~CudaNeuronBlock();
};
#endif 
