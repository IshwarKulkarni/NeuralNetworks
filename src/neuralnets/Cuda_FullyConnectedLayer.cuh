#ifndef __CUDA_FULLY_CONNECTED_LAYER_INCLUDED__
#define __CUDA_FULLY_CONNECTED_LAYER_INCLUDED__

#include <string>

#include "utils/CudaSimpleMatrix.cuh"
#include "Activation.hxx"

struct CudaNeuronBlock
{
    ActivationFunction Act;
    CudaSimpleMatrix::CudaMatrix<double> Weights, Results, LGrads;

    CudaNeuronBlock(SimpleMatrix::Matrix<double>& NeuronWeights, std::string activationName ):
        Act(GetCudaActivationFunction(activationName)),
        Weights(NeuronWeights),
        Results({ NeuronWeights.size.y,1 }),
        LGrads({ NeuronWeights.size.y,1 })
    {
        NeuronWeights.Clear(); // Creator should destroy this. Bad!
    }

    void Fire(double* input);
    
    ~CudaNeuronBlock()
    {
        Weights.Clear();
        Results.Clear();
        LGrads.Clear();
    }
};
#endif 
