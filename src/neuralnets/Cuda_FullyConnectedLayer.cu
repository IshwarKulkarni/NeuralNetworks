#include "Cuda_FullyConnectedLayer.cuh"

#ifdef _DEBUG
#define NN_DEBUG 1
#define NN_PRINT 1
#else
#define NN_PRINT 0
#define NN_DEBUG 0
#endif

using namespace CudaSimpleMatrix;

__global__ void DotAndActivate(
	CudaMatrix<double> wAndB, double* inputs,
	 double* lgrads, ActivationId activation, 
	 double* results)
{
	size_t y = threadIdx.x;
	results[y] = wAndB.at(y, (wAndB.size.x -1));
	//
	for (size_t x = 0; x < (wAndB.size.x - 1); x++)
		results[y] += wAndB.at(y,x) * inputs[x];
	
	results[y] = Activate(activation, results[y], lgrads[y]);
}

CudaNeuronBlock::CudaNeuronBlock(Vec::Size2 WeightsSize, ActivationId actId) :
	Act(actId),
	Weights(WeightsSize),
	Results({ WeightsSize.y, 1 }),
	Grads({ WeightsSize.y, 1 }),
	LGrads({ WeightsSize.y, 1 }),
	PGrads({WeightsSize.x-1, WeightsSize.y}),
	Input({ WeightsSize.x - 1, 1 })

{
	double rs = double(1 / sqrt(Weights.size.x));
	for (size_t i = 0; i < Weights.size(); ++i)
		Weights.devData[i] = NN_DEBUG ? 0.1 : Utils::URand(rs, -rs);

	CUDA_CHECK_SYNCH;
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

CudaSimpleMatrix::CudaMatrix<double> CudaNeuronBlock::ForwardPass(double* input){

	CudaSimpleMatrix::CudaMatrix<double> in({ Weights.size.x-1, 1 }, input);
	
	CUDA_CHECK_SYNCH;
	DotAndActivate <<<1, unsigned( Weights.size.y) >>> (
		Weights, in.devData, LGrads.devData, Act, Results.devData);

    CUDA_CHECK_SYNCH;

    in.Clear();

	return Results;

};