#include "Cuda_FullyConnectedLayer.cuh"


__global__ void DotAndActivate(
    double* weightsAndBias, double* inputs,
    size_t numWeights, 
    double* results, double* lgrads,
    ActivationFunction activation)
{
    size_t id = threadIdx.x;
    results[id] = weightsAndBias[numWeights];
    for (size_t i = 0; i < numWeights; i++)
        results[id] += weightsAndBias[i] * inputs[i];

    results[id] = activation(results[id], lgrads[id]);
}

void CudaNeuronBlock::Fire(double* input){

    double* devInput; size_t pitch;
    cudaMallocPitch(&devInput, &pitch, Weights.size.x, 1);

    DotAndActivate<<<1,Weights.size.y>>>( 
        Weights.devData, devInput, 
        Weights.size.x, 
        Results.devData, LGrads.devData,
        Act);

    cudaFree(devInput);

};