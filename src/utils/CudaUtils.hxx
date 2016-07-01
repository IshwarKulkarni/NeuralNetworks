#ifndef __CUDA_UTILS_INCLUDED__
#define __CUDA_UTILS_INCLUDED__

#include "utils/utils.hxx"
#include <string>
#include <cuda_runtime.h>

#define CUDA_CHECK() do{						\
	cudaDeviceSynchronize();					\
	cudaError_t err = cudaGetLastError();		\
	if (err != cudaSuccess) {					\
		Logging::Log << Logging::Log.flush		\
			<< "\n" << __FILE__ <<":"<< __LINE__\
			<< " >>> "<< cudaGetErrorString(err)\
			<< "\n";							\
		throw std::runtime_error("");			\
		}										\
	} while (false);							\

template<typename T>
inline T* cudaAllocCopy( size_t length = 1, const T* inData = nullptr, cudaMemcpyKind kind = cudaMemcpyHostToDevice)
{
	T* devData = nullptr;
	length = length*sizeof(T);
	cudaMalloc((void**)(&devData), length);
	CUDA_CHECK();

	if (inData) cudaMemcpy(devData, inData, length, kind);
	
	CUDA_CHECK();

	return devData;
}

template<typename T>
inline T* cudaAllocCopy(T* devData, size_t length = 1, const T* inData = nullptr, cudaMemcpyKind kind = cudaMemcpyHostToDevice)
{
	length *= sizeof(T); 
	cudaMalloc((void**)(&devData), length);
	CUDA_CHECK();

	if (inData) cudaMemcpy(devData, inData, length, kind);

	CUDA_CHECK();
	return devData;
}


template<typename T>
T* cudaPitchedAllocCopy(size_t w, size_t h, size_t& pitch, const T* inData = nullptr, cudaMemcpyKind dirn = cudaMemcpyHostToDevice)
{
	w *= sizeof(T);

	T* devData = 0;
	cudaMallocPitch(&devData, &pitch, w, h);
	if (inData)
		cudaMemcpy2D(devData, pitch, inData, pitch, w, h, dirn);

	return devData;
}



#endif