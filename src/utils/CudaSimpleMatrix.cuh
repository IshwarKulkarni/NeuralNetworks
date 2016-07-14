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

#ifndef __CUDA_SIMPLE_MATRIX_INCLUDED__
#define __CUDA_SIMPLE_MATRIX_INCLUDED__
#include <algorithm>

#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include "SimpleMatrix.hxx"
#include "CudaUtils.cuh"

#include <iostream>

namespace CudaSimpleMatrix {


	template<typename T> struct CuMaxOp  { __device__ __forceinline__ T Apply(const T& t1, const T& t2) const { return (t1 > t2 ? t1 : t2); } };
	template<typename T> struct CuMinOp  { __device__ __forceinline__ T Apply(const T& t1, const T& t2) const { return (t1 > t2 ? t2 : t1); } };
	template<typename T> struct CuBinAdd { __device__ __forceinline__ T Apply(const T& t1, const T& t2) const { return t1 + t2; } };

    template<typename T, typename BinOp>
    __global__
        void reduceHorizontally(T *g_idata, T *g_odata) {  // 1 row of g_data matrix should fit in smem
        extern __shared__ T sdata[];

        unsigned tid = threadIdx.x, i = blockIdx.x*blockDim.x * 2 + threadIdx.x;
        BinOp bop;
        sdata[tid] = bop.Apply(g_idata[i], g_idata[i + blockDim.x]);
        __syncthreads();

        for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s)
                sdata[tid] = bop.Apply(sdata[tid], sdata[tid + s]);
            __syncthreads();
        }

        if (tid == 0) g_odata[blockIdx.x] = sdata[0];
    }

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

		__device__ __host__  T& at(size_t y, size_t x) { return devData[size.x*y + x];  }

		__device__ __host__ const T& at(size_t y, size_t x) const { return devData[size.x*y + x]; }

        void Clear() { CUDA_CHECK(devData && cudaFree(devData));  devData = nullptr; }

        template<typename Iter>
        void CompareTo(Iter begin, Iter end, const char* msg = "", bool throwErr = true, size_t offset = 0)
        {
            auto res = CudaUtils::DevHostCmp(begin, end, devData, false);
            if (!res.first)
            {
                T d = devData[res.second], h = *(begin + res.second);
                Logging::Log
                    << "\n" << msg << " match failed at ("
                    << res.second%size.x << "," << res.second / size.x << ")"
                    << ") Device value: " << d << ". Other value:  " << h
                    << " , D/O - 1 : " << (d/h - 1)
                    << " , |D-O| : " << std::abs(d-h)
                    << " > " << std::numeric_limits<T>::epsilon() << std::hex
                    << " , " << Utils::BitRep<T>(d)() << "-" << Utils::BitRep<T>(h)() 
                    << std::dec
                    << "\nDevice matrix: " << *this << "\nOther Matrix : ";
                SimpleMatrix::Out2d(Logging::Log, &begin, size.x, size.y,"");

                throw std::runtime_error("device and host computation disagree");
            }
            Logging::Log << "\n" << msg << ".. Macth!";
        }

        template <typename U>
        void Copy(const U* in) // std copy for managed. enhance for pure dev ptr later
        {
            for (size_t i = 0; i < size(); ++i) devData[i] = *in++;
        }

        T* devData;
        Vec::Size2 size; 

	private:
		CudaMatrix() {};
    };

    template<typename T>
    std::ostream&  operator << (std::ostream& o,const CudaMatrix<T>& mat) 
    {
        // will have to enhance this when devData is device-only ptr.
        SimpleMatrix::Out2d(o, &(mat.devData), mat.size.x, mat.size.y);
        o.flush();
        return o;
    }
}

#endif
 