#ifndef __CUDA_SIMPLE_MATRIX_INCLUDED__
#define __CUDA_SIMPLE_MATRIX_INCLUDED__
#include <algorithm>

#include "cuda_runtime.h"
#include "SimpleMatrix.hxx"
#include "Vec23.hxx"


namespace CudaSimpleMatrix {

    template<typename T>
    struct CudaMatrix 
    {
        CudaMatrix(Vec::Size2 s, T* inData = nullptr) : size(s)
        {
            const size_t sizeW = sizeof(T)*size.x;

            if (cudaMallocPitch(&devData, &size.z, sizeW, size.y))
                throw std::runtime_error("Cannot do pitched malloc.");

            if (inData && devData)
                if( cudaMemcpy2D(devData, size.z, inData, sizeW, sizeW, size.y, cudaMemcpyHostToDevice) )
                    throw std::runtime_error("Cannot do cudaMemcpy.");
        }

        CudaMatrix(const SimpleMatrix::Matrix<T>& mat) : size(mat.size)
        {
            if (!mat.size())  throw std::runtime_error("Cannot construct from empty Matrix");
            
            const size_t sizeW = sizeof(T)*size.x;

            if (cudaMallocPitch(&devData, &size.z, sizeW, size.y) )
                throw std::runtime_error("Cannot do pitched malloc.");

            if (devData)
                if (cudaMemcpy2D(devData, size.z, mat.data, sizeW, sizeW, size.y, cudaMemcpyHostToDevice))
                    throw std::runtime_error("Cannot do cudaMemcpy.");
        }

        void Clear()
        {
            if (devData && cudaFree(devData))
                throw std::runtime_error("could not free pointer");
        }

        T* CopyOut()
        {
            T* out = new T[size.x*size.y];
            if (cudaMemcpy2D(out, size.x*sizeof(T), devData, size.z, size.x*sizeof(T), size.y, cudaMemcpyDeviceToHost))
                throw std::runtime_error("Cuda memcpy out failed");

            return out;
        }

        std::pair<T*,size_t> CompareDevToHost(T* hostPtr)
        {
            T* devData = CopyOut();
            auto mis = std::mismatch(hostPtr, hostPtr + size.x * size.y, devData);
            if (mis.first == hostPtr + size.x * size.y)
            {
                delete[] devData;
                return make_pair(nullptr, 0);
            }
            return make_pair(devData , mis.first - devData);

        }

        T* devData;
        Vec::Size3 size; // let's say z is size.z
    };
}

#endif
