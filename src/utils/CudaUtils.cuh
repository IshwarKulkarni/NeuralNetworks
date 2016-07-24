/*
Copyright (c) Ishwar R. Kulkarni
All rights reserved.

This file is part of NeuralNetwork Project by
Ishwar Kulkarni , see https://github.com/IshwarKulkarni/NeuralNetworks

If you so desire, you can copy, redistribute and/or modify this source
along with  rest of the project. However any copy/redistribution,
including but not limited to compilation to binaries, must carry
this header in its entirety. A note must be made about the origin
of your copy.NYSE:TZF.CL

NeuralNetwork is being distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
FITNESS FOR A PARTICULAR PURPOSE.

*/

#ifndef _CUDA_UTILS_INCLUDED_
#define _CUDA_UTILS_INCLUDED_

#include "Utils.hxx"
#include "Vec23.hxx"

#define CUDA_CHECK(Err) if((Err) != cudaSuccess) { \
    std::cerr << "\nCuda error at " << __FUNCTION__ << " " << __FILE__ << "(" << __LINE__ << "): "  \
              << cudaGetErrorString(cudaGetLastError()); \
    throw std::runtime_error( std::string("Cuda Error ") );\
} 

#define CUDA_CHECK_ERR  { CUDA_CHECK(cudaPeekAtLastError()); }

#define CUDA_CHECK_SYNCH  {cudaDeviceSynchronize(); CUDA_CHECK_ERR;}
	
#define CUDA_CHECK_FREE(ptr)  CUDA_CHECK(cudaFree(ptr));

#define VEC22DIM(s) dim3(unsigned(s.x),unsigned(s.y),1)

#define VEC32DIM(s) dim3(unsigned(s.x),unsigned(s.y),unsigned(s.z))

#define DIM32SIZE(s) Vec::Size3(s.x,s.y,s.z)

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

    __device__ inline size_t Lin2(const Vec::Size3& s) { return s.x*s.y; }

    template<typename Size3Type>
    __device__ inline size_t LinOffset(const Size3Type& s, const Vec::Size3& v) 
    { return v.z * s.x * s.y + v.y * s.x + v.x; }

    template<typename T, typename Iter> 
    std::pair<bool, size_t> DevHostCmp(Iter begin, Iter end, T* devData, bool print = true)
    {
        auto m = std::mismatch(begin, end, devData, Utils::FloatCompare<T,1>);
        auto dist = std::distance(begin, end);
        
        if (m.first != end || m.second != devData + dist)
        {
            if (print) {
                Logging::Log << setprecision(9);
                Utils::PrintLinear(Logging::Log, begin, dist, "\nHost:  \t");
                Utils::PrintLinear(Logging::Log, devData, dist, "\nDevice:\t");
                Logging::Log << Logging::Log.flush;
            }
            return make_pair(false, std::distance(begin, m.first));
        }
        return make_pair(true, 0);
    }

	struct KernelLaunchParams
	{
		dim3 GridSize, BlockSize;
		size_t BlockSharedMemSize;
	};

    inline std::ostream& operator << (std::ostream& o, const KernelLaunchParams& klp)
    {
        o << "  GridDim : " << DIM32SIZE(klp.GridSize) << "\nBlockSize : " << DIM32SIZE(klp.BlockSize)
            << "\nShared Mem: " << klp.BlockSharedMemSize << "Bytes\n";
        return o;
    }

	inline int size_of(dim3 d) { return d.x*d.y*d.z; }

    struct LaunchChecker
	{
		static bool Check(KernelLaunchParams& klp)
		{
			static LaunchChecker stChecker;
			return
				LaunchChecker::GetProps().sharedMemPerBlock >= klp.BlockSharedMemSize &&
				LaunchChecker::GetProps().maxThreadsPerBlock >= size_of(klp.BlockSize) &&
				LaunchChecker::GetProps().maxGridSize[0] >= int(klp.GridSize.x) &&
				LaunchChecker::GetProps().maxGridSize[1] >= int(klp.GridSize.y) &&
				LaunchChecker::GetProps().maxGridSize[2] >= int(klp.GridSize.z) &&
				LaunchChecker::GetProps().maxThreadsDim[0] >= int(klp.BlockSize.x) &&
				LaunchChecker::GetProps().maxThreadsDim[1] >= int(klp.BlockSize.y) &&
				LaunchChecker::GetProps().maxThreadsDim[2] >= int(klp.BlockSize.z);
		}

		static bool ConstCheck(unsigned constSize)
		{
			static LaunchChecker stChecker;
			return LaunchChecker::GetProps().totalConstMem >= constSize;
		}
	private:
		
		LaunchChecker()
		{
			cudaGetDeviceProperties(&LaunchChecker::GetProps(), CUDA_DEVICE_NUM);
		}

		static cudaDeviceProp& GetProps() {
			static cudaDeviceProp  props; return props;
		}
	};
}
#endif 

