#ifndef __CUDA_SIMPLE_MATRIX_INCLUDED__
#define __CUDA_SIMPLE_MATRIX_INCLUDED__
#include <algorithm>

#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include "SimpleMatrix.hxx"

#include <iostream>

#define CUDA_CHECK(Err) if((Err) != cudaSuccess) { \
    std::cerr << "\nCuda error at " << __FUNCTION__ << " " << __FILE__ << "(" << __LINE__ << "): "  \
              << cudaGetErrorString(cudaGetLastError()); \
    throw std::runtime_error( std::string("Cuda Error ") + cudaGetErrorString(cudaGetLastError()));\
} 

#define CUDA_CHECK_SYNCH CUDA_CHECK(cudaDeviceSynchronize());

#define CUDA_CHECK_FREE(ptr)  CUDA_CHECK(cudaFree(ptr));

namespace CudaSimpleMatrix {

    template<typename T>
    struct CudaMatrix 
    {
        CudaMatrix(Vec::Size2 s, T* inData = nullptr) : size(s)
        {
            CUDA_CHECK(cudaMallocManaged(&devData, size() * sizeof(T), cudaMemAttachGlobal));

            if (inData) 
                CUDA_CHECK(cudaMemcpy(devData, inData, size()* sizeof(T),cudaMemcpyHostToDevice));
        }

        CudaMatrix(const SimpleMatrix::Matrix<T>& mat) : size(mat.size)
        {
            if (!mat.size())  throw std::runtime_error("Cannot construct from empty Matrix");
            
            CUDA_CHECK(cudaMallocManaged(&devData, mat.size() * sizeof(T), cudaMemAttachGlobal));

            CUDA_CHECK(cudaMemcpy(devData, mat.data[0], mat.size()* sizeof(T), cudaMemcpyHostToDevice));
        }

		__device__ __host__
		T& at(size_t y, size_t x) { 
			return devData[size.x*y + size.x]; 
		}

		__device__ __host__
		const T& at(size_t y, size_t x) const { 
			return devData[size.x*y + size.x]; 
		}

        void Clear()
        {
            CUDA_CHECK(devData && cudaFree(devData));
        }

		template<typename Iter>
		void CompareTo(Iter begin, Iter end, const char* msg)
		{
			Logging::Log << msg << " ... ";
			if (!std::equal(begin, end, devData))
			{
				Utils::PrintLinear(Logging::Log, begin, size(),  "\nHost:  \t");
				Utils::PrintLinear(Logging::Log, devData, size(),"\nDevice:\t");
				Logging::Log << Logging::Log.flush;
				throw std::runtime_error("device and host computation disagree");
			}
			Logging::Log << " Match!\n";

		}

        T* devData;
        Vec::Size2 size; // let's say pitch (in element size) is size.z

		void Print(std::ostream& o)
		{
			SimpleMatrix::Out2d(o, &devData, size.x, size.y);
			o.flush();
		}
    };

}

#endif
