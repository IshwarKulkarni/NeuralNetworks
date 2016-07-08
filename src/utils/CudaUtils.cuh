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

#ifndef _CUDA_UTILS_INCLUDED_
#define _CUDA_UTILS_INCLUDED_

#define CUDA_CHECK(Err) if((Err) != cudaSuccess) { \
    std::cerr << "\nCuda error at " << __FUNCTION__ << " " << __FILE__ << "(" << __LINE__ << "): "  \
              << cudaGetErrorString(cudaGetLastError()); \
    throw std::runtime_error( std::string("Cuda Error ") + cudaGetErrorString(cudaGetLastError()));\
} 

#define CUDA_CHECK_ERR  { cudaError_t err = cudaGetLastError();  CUDA_CHECK(err); }

#define CUDA_CHECK_SYNCH  cudaDeviceSynchronize(); CUDA_CHECK_ERR  
	
#define CUDA_CHECK_FREE(ptr)  CUDA_CHECK(cudaFree(ptr));

#define VEC22DIM(s) dim3(unsigned(s.x),unsigned(s.y),1)

#define VEC32DIM(s) dim3(unsigned(s.x),unsigned(s.y),unsigned(s.z))

#define EXPAND_KLP(lp) lp.GridSize, lp.BlockSize, lp.BlockSharedMemSize

#ifndef CUDA_DEVICE_NUM 
#define CUDA_DEVICE_NUM 0
#endif 

namespace CudaUtils
{
    __host__ __device__
    inline unsigned NextPowerOf2(unsigned v)
    {// from https://graphics.stanford.edu/~seander/bithacks.html#RoundUpPowerOf2
        v--;
        v |= v >> 1;
        v |= v >> 2;
        v |= v >> 4;
        v |= v >> 8;
        v |= v >> 16;
        v++;
        return v;
    }
    
    template<typename T, typename Iter> 
    bool DevHostCmp(Iter begin, Iter end, T* devData)
    {
        struct { bool operator()(double a, double b){return abs(a - b) < (10e-12); }} cmp;            
       
        if (!std::equal(begin, end, devData, cmp))
        {
            Logging::Log << setprecision(9);
            auto dist = std::distance(begin, end);
            Utils::PrintLinear(Logging::Log, begin, dist, "\nHost:  \t");
            Utils::PrintLinear(Logging::Log, devData, dist, "\nDevice:\t");
            Logging::Log << Logging::Log.flush;
            return false;
        }
        //Logging::Log << " match\n" << Logging::Log.flush;
        return true;
    }

	struct KernelLaunchParams
	{
		dim3 GridSize, BlockSize;
		size_t BlockSharedMemSize; // per
	};

	inline int size_of(dim3 d) { return d.x*d.y*d.z; }

	struct LaunchChecker
	{
		static bool Check(KernelLaunchParams& klp)
		{
			static LaunchChecker stChecker;
			return
				stChecker.props.sharedMemPerBlock >= klp.BlockSharedMemSize &&

				stChecker.props.maxThreadsPerBlock >= size_of(klp.BlockSize) &&

				stChecker.props.maxGridSize[0] >= int(klp.GridSize.x) &&
				stChecker.props.maxGridSize[1] >= int(klp.GridSize.y) &&
				stChecker.props.maxGridSize[2] >= int(klp.GridSize.z) &&
				stChecker.props.maxThreadsDim[0] >= int(klp.BlockSize.x) &&
				stChecker.props.maxThreadsDim[1] >= int(klp.BlockSize.y) &&
				stChecker.props.maxThreadsDim[2] >= int(klp.BlockSize.z);
		}
	private:
		LaunchChecker()
		{
			cudaGetDeviceProperties(&props, CUDA_DEVICE_NUM);
		}
		cudaDeviceProp props;
	};
}
#endif 
