#ifndef __CUDA_UTILS_HXX_INCLUDED__
#define __CUDA_UTILS_HXX_INCLUDED__


#include <istream>
#include <string>
#include <type_traits>
#include <memory>
#include <vector>
#include <map>
#include <set>
#include <type_traits>
#include "Logging.hxx"
#include "Exceptions.hxx"
#include "CudaException.hxx"
#include "Utils.hxx"
#include "imageprocessing/Color.hxx"
#include "imageprocessing/ImageGeneric.hxx"

#include <cmath>

typedef unsigned int uint;
typedef unsigned long ulong;

#define IMAD(a, b, c) ( __mul24((a), (b)) + (c) )
#define Pack(c0,c1,c2,c3) (c0 << 24) | (c1 << 16) | (c2 << 8) | c3)
#define FLT_MAX  3.402823466e+38F        /* copied from limits file, used in numeric_limits<float>::max() */
#define EPSILON  0.0001
#define FLT_CMPR(a,b) (fabs((a)-(b)) < EPSILON)

#define IMXY uint x = IMAD(blockIdx.x,blockDim.x,threadIdx.x), y = IMAD(blockIdx.y,blockDim.y,threadIdx.y);    
#define IMXYCHECK(w,h) IMXY; if( !(x<w&&y<h) ) return;  // if this aint a Macro abuse, nothing is!

#define LaunchConfig(w,h,f)     dim3 gridSize, blockSize; GetGridAndBlockSize2D(w,h,gridSize, blockSize, f);

#define TexCacheSize (36*(1<<10))
#define L1CacheSize  (2*(1<<10))

#define IF_T0_B0 if(threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0 && \
                     blockIdx.x == 0 && blockIdx.y == 0  && blockIdx.z == 0)

//#define ALLOC_DEBUG

#ifdef ALLOC_DEBUG
std::set<void*>  DebugPointerSet;
#define LoggedCudaFree(ptr)    Logging::Log << "ALLOC_DEBUG : " LOG_LOC() << " -> deleting " << #ptr << " = 0x" << ptr << LogEndl ; THROW_IF(ptr && cudaFree(ptr), CUDAFreeException, "Free failed"); DebugPointerSet.erase(ptr);

#define LoggedCudaMallocPitch(devData, pitch, w, h) \
    THROW_IF( cudaMallocPitch(devData, pitch, sizeW, h), CUDA2DAllocException , "cudaMallocPitch failed, width = %d bytes, height = %d",sizeW, h); \
    Logging::Log << "ALLOC_DEBUG : " << LOG_LOC() << " -> Alloc'ed 0x" << *devData << " with width = " << sizeW << " bytes, height = " << h << LogEndl; DebugPointerSet.insert(*devData);

void CHECK_UNDEAD_PTRS() 
{
    if(!DebugPointerSet.empty()) 
    {
        uint i = DebugPointerSet.size();
        Logging::Log << "Following " <<  i << " undeleted pointer" << (i == 1 ? "" : "s" ) <<" found :\n";
        for(auto ptr : DebugPointerSet)
            Logging::Log << i-- << ": 0x" << ptr << LogEndl;
    }
    else
        Logging::Log << "No undeleted pointers.. All Clear!\n";
}

#else 

#define LoggedCudaFree(ptr) THROW_IF(ptr && cudaFree(ptr), CUDAFreeException, "Free failed");

#define LoggedCudaMallocPitch(devData, pitch, w, h) \
    THROW_IF( cudaMallocPitch(devData, pitch, sizeW, h), CUDA2DAllocException , "cudaMallocPitch failed, width = %d bytes, height = %d",sizeW, h);


#endif

#define Debugging 0

__device__ __host__ __forceinline__  
bool IsAlpha(char a) { return (a >='a' && a <='z') || ( a >='A' && a <='Z' ) ;}

__device__ __host__ __forceinline__  
unsigned long long powull(unsigned long long a, unsigned long long b)  // a^b
{
    if( b == 0 )  return 1;
    if( b == 1 )  return a;

    return ( (b % 3 == 0 )  ?  powull(a*a*a,b/3) : 
           ( (b % 2 == 0 )  ?  powull(a*a,  b/2) : a*(powull(a*a,(b-1)/2)) ) );
}

inline std::ostream& operator<<(std::ostream& strm, const dim3& d )
{
    if(d.z!=1)
        strm << d.x << "x" << d.y << "x" << d.z ;
    else
        strm << d.x << "x" << d.y;

    strm << "( " << d.x*d.y*d.z << " )";    

    return strm;
}

inline std::ostream& operator<<(std::ostream& strm, const std::pair<dim3,dim3>& d )
{
    strm    << d.first << " x " << d.second << " : " 
            << d.first.x*d.second.x << "x" 
            << d.first.y*d.second.y << "x" 
            << d.first.z*d.second.z << "( " 
            << d.first.x*d.first.y*d.first.z*d.second.x*d.second.y*d.second.z  
            << " )";    
    return strm;
}

// valid strings:  "1,2,3" or "   -1 2 3" or "(  -1,2,3_"
inline std::istream& operator>>(std::istream& strm, dim3& d ) 
{
    char c,f;
    auto readNextNum = [&] (unsigned& i) { 
        f = strm.peek() ; 
        if(  f == '-' || isdigit(f) || iswspace(f) )
            strm >> i;
        else
            strm >> c >> i ;
    };

    readNextNum(d.x);
    readNextNum(d.y);
    readNextNum(d.z);
    return strm;
}


template<typename T> // product of all those pesky __3 versions structs in cuda
double Product3(const T& t) { return t.x*t.y*t.z; }

template<typename T> // product of all those pesky __4 versions structs in cuda
double Product4(const T& t) { return t.x*t.y*t.z*t.w; }

template<typename T> // Sum of all those pesky __3 versions structs in cuda
double Sum(const T& t) { return t.x + t.y + t.z; }


//__device__ __inline__
//void tex2D(double* val, cudaTextureObject_t cudaTexMatTex, uint imX, uint imY)
//{
//    union i2Tod{int2 i2; double d;}e;
//    tex2D(&(e.i2), cudaTexMatTex, imX, imY);
//    *val = e.d;
//}
//
//__device__ __inline__
//void tex2D(Color* val, cudaTextureObject_t cudaTexMatTex, uint imX, uint imY)
//{
//    tex2D((int*)(val), cudaTexMatTex, imX, imY);
//}

template <typename T>
struct CudaDeleter { inline void operator()(T* ptr) { LoggedCudaFree(ptr); } };

typedef std::unique_ptr<float, CudaDeleter<float> > CudaFloatPointer;

template<typename T>
inline T* cudaCopyOut(const T* inData, const size_t pitch, uint w, uint h)
{
    if( inData )
    {
        T* out = new T[w*h];
        try{ THROW_IF( cudaMemcpy2D(out, w*sizeof(T),  inData,pitch , w*sizeof(T), h, cudaMemcpyDeviceToHost) , 
            CUDACopyException, "cudaMemcpy failed to copy from ptr = 0x%p, and width = %d, height = %d", inData, w, h); }
        catch( ... ) { delete[] out; throw;}
        return out;
    }
    return 0;
}

template<typename T>
void cudaCopyOut(const T* inData, const size_t pitch, uint w, uint h, T** out)
{
    if (!inData) return;
    
    THROW_IF(cudaMemcpy2D(*out, w*sizeof(T), inData, pitch, w*sizeof(T), h, cudaMemcpyDeviceToHost),
            CUDACopyException, "cudaMemcpy failed to copy from ptr = 0x%p, and width = %d, height = %d", inData, w, h);
}

class DelayedCopy{

    typedef std::map<void* , cudaPitchedPtr> PointerMap ;
public:
    static DelayedCopy& Inst()
    {
        static DelayedCopy instance;        
        return instance;
    }

    template<typename T>
    inline bool CudaCopyOut(const T* src, const size_t pitch, uint w, uint h, T* dest)
    {
        THROW_IF(src == NULL || dest == NULL , CUDACopyException,
            "Neither source nor destination of copy can be NULL" ); 
        if( m_HoldOutCopy )
        {
            cudaPitchedPtr pitchedPtr;
            pitchedPtr.ptr = (void*) src;
            pitchedPtr.xsize = w*sizeof(T);
            pitchedPtr.ysize = h;
            pitchedPtr.pitch = pitch;
            m_HoldOutPitchedMap.insert(std::make_pair((void*)dest, pitchedPtr));
            return false;
        }

        THROW_IF(cudaMemcpy2D(  dest,    w*sizeof(T),  
                                src,    pitch , 
                                w*sizeof(T), h, 
                                cudaMemcpyDeviceToHost), CUDACopyException, "CUDA 2d memcpy failed" );
        CUDA_CHECK();
        return true;
    }
    
    // remove without performing update; return if actually OOD
    inline bool RemovePendingUpdate(void* ptr) 
    {
        auto found = m_HoldOutPitchedMap.find(ptr);
        if( found == m_HoldOutPitchedMap.end())
        {
            //Logging::Log << "\nRemoving ptr " << std::hex << "0x" << (unsigned long long) ptr << std::dec <<  "..not found\n";
            return false;
        }

        //Logging::Log << "\nRemoving ptr " << std::hex << "0x" <<  (unsigned long long) ptr << std::dec << " mapped to to 0x" << (unsigned long long) found->second.ptr << std::dec << LogEndl;

        m_HoldOutPitchedMap.erase(found);
        return true;
    }

    // perform one or all updates; return if actually updated
    inline bool PerformPendingUpdates(void* ptr = NULL) 
    {
        if( ptr )
        {
            auto found = m_HoldOutPitchedMap.find(ptr);
            if( found == m_HoldOutPitchedMap.end())
                return false;
        
            PerformOneCopy(found); // proably try catch this
            m_HoldOutPitchedMap.erase(found);
            return false;
        }

        for(PointerMap::iterator it  = m_HoldOutPitchedMap.begin();
                                 it != m_HoldOutPitchedMap.end();
                                 ++it)
            PerformOneCopy(it);

        if( m_HoldOutPitchedMap.size())
        {
            m_HoldOutPitchedMap.clear();
            return true;
        }
        return false;
    }
    
    inline bool HasPendingUpdate(void* ptr = NULL)
    {
        return m_HoldOutPitchedMap.find(ptr) == m_HoldOutPitchedMap.end();
    }
    
    bool HoldOutCopy(){

        bool ret = m_HoldOutCopy;
        m_HoldOutCopy = true;
        return ret;
    }


    bool StopHoldOutCopy(){ // does not update existing OOD pointers;

        bool ret = m_HoldOutCopy;
        m_HoldOutCopy = false;
        return ret;
    }


    bool UTDCopies(){

        bool ret = m_HoldOutCopy;
        m_HoldOutCopy = true;
        return ret;
    }


private:
    template<typename T>
    size_t PerformOneCopy(T iter)
    {
         
        Logging::Log << "Attempting copy from " << std::hex 
            << "0x" << (unsigned long long) iter->second.ptr 
            << " to 0x" << (unsigned long long) iter->first << std::dec ;
        
        char* p  = (char*) iter->first;
        char c = p[iter->second.ysize * iter->second.xsize];
        THROW_IF(cudaMemcpy2D(
                    iter->first,       iter->second.xsize,  
                    iter->second.ptr , iter->second.pitch , 
                    iter->second.xsize,iter->second.ysize,
                    cudaMemcpyDeviceToHost) , CUDACopyException, "CUDA 2d memcpy failed" );
        Logging::Log << " .. Succeeded\n";
        return iter->second.xsize*iter->second.ysize;
    }
    bool m_HoldOutCopy ; // dont turn this ON, bad things will happen
    
    
    PointerMap  m_HoldOutPitchedMap; // encapsulate and hide this and above in a class

    DelayedCopy() {
        m_HoldOutCopy = false;
    }
    DelayedCopy(const DelayedCopy&){}             
    void operator=(const DelayedCopy&){}
};

template<typename T>
T* cudaPitchedAllocDevCopy(const T* inData, uint w, uint h, size_t& pitch, cudaMemcpyKind dirn = cudaMemcpyDeviceToDevice )
{
    const size_t sizeW = sizeof(T)*w;
    
    T* devData  = 0;
    LoggedCudaMallocPitch(&devData,&pitch,sizeW,h);
    if( inData  )
    {
        if( devData )
            try{ THROW_IF ( cudaMemcpy2D(devData,pitch,inData,pitch,sizeW,h,dirn), CUDACopyException ,"CUDA memcpy failed and returned."); }
            catch(...) { LoggedCudaFree(devData); throw; }
    }
    
    return devData;
}

template<typename T>
inline T* cudaCopyOut(const T* inData, uint len)
{
    if( inData )
    {
        auto out = new T[len];
        try {  THROW_IF( cudaMemcpy(out,inData,len*sizeof(T),cudaMemcpyDeviceToHost), 
            CUDACopyException,  "CUDA memcpy from device failed."); }
        catch( ... ) { delete[] out; throw;}
        return out;
    }
    return 0;
}

template<typename T>
inline T* cudaCopyOut(const cudaPitchedPtr& ptr)
{
    return cudaCopyOut((T*)ptr.ptr, ptr.pitch, ptr.xsize, ptr.ysize);
}

template<typename T>
inline T* cudaAllocCopy(const T* inData, uint length=1 ,cudaMemcpyKind kind = cudaMemcpyHostToDevice)
{
    T* devData = 0;
    length = length*sizeof(T);
    THROW_IF( cudaMalloc((void**)(&devData), length), CUDAAllocException, "CUDA malloc pitch failed  and returned." );
    if( inData)
        try { THROW_IF( cudaMemcpy(devData, inData, length , kind) , CUDACopyException,
                "CUDA memcpy failed  and returned." ); }
        catch ( ... ) { LoggedCudaFree(devData); throw; } 

    if(Debugging) 
        Logging::Log << "\nAllocing Device Data=0x" << std::hex << (unsigned long long) devData << std::dec 
        << " Length =" << length << " Copied from 0x" << std::hex << (unsigned long long)inData<< std::dec;

    return devData;
}

//__device__ __constant__ void* DevData;

template<typename T>
T* cudaPitchedAllocCopy(const T* inData, uint w, uint h, size_t& pitch)
{
    const size_t sizeW = sizeof(T)*w;
    T* devData = 0;
    THROW_IF( cudaMallocPitch(&devData, &pitch, sizeW, h), CUDA2DAllocException , "CUDA malloc pitch failed with to copy from ptr = 0x%p, with width = %d, height = %d",inData, w, h);
    //LoggedCudaMallocPitch(&devData, &pitch, sizeW, h);
    if( inData )
    {
        if( devData )
            try{ THROW_IF ( cudaMemcpy2D(devData,pitch,inData,sizeW,sizeW,h,cudaMemcpyHostToDevice), CUDACopyException ,"CUDA memcpy failed and returned."); }
        catch(...) { LoggedCudaFree(devData); throw; }
    }
    Logging::Log << "\nAllocing Device Data=0x" << std::hex << (unsigned)devData << std::dec << " Pitch=" << pitch << ", width=" << w << ", Height="<< h << " Copied from 0x" << std::hex << (unsigned)inData<< std::dec;
    return devData;
}

template<typename T>
void cudaPitchedAllocCopy(cudaPitchedPtr& ptr){  
    // caution, ptr.xsize and ptr.ysize should be filled before calling this function.
    ptr.ptr = cudaPitchedAllocCopy<T>(0, ptr.xsize, ptr.ysize, ptr.pitch);
}

template<typename T>
__host__ __device__ __inline__ T& offsetToPitchedArray(T* ptr, unsigned bytePitch, unsigned x, unsigned y)
{
    return *(T*)((char*)(ptr)+bytePitch * y + x * sizeof(T));
}

template<typename T>
cudaPitchedPtr createPitchedPointer(uint w, uint h)
{
    cudaPitchedPtr ptr;
    ptr.xsize = w, ptr.ysize = h;
    cudaPitchedAllocCopy<T>(ptr);
    return ptr;
}

template<typename T>// takes a vector of pointers and makes an equivalent array on device
T** cudaCopyPtrVector(const std::vector<T*>& vec, unsigned& size) 
{
    unsigned numElem = vec.size();
    vector<T*> devicePtrs(numElem);

    for (size_t i = 0; i < numElem; ++i) //copy the objects
        devicePtrs[i] = cudaAllocCopy(vec[i]);

    return cudaAllocCopy(&(devicePtrs[0]), numElem); // copy the ptr array
}

template<typename T>
void cudaClearPtrArrays(T** arrayPtr, unsigned numPtrs)
{
    T** hostPtrs = cudaCopyOut(arrayPtr, numPtrs);
    for (size_t i = 0; i < numPtrs; ++i)
        cudaFree(hostPtrs[i]);

    cudaFree(arrayPtr);
    delete[] hostPtrs;
}

struct TwoDPrint{

    char RowDelim;

    template<typename O, typename T>
    void Cuda(O& outStream, const T& devPtr, size_t pitch, uint w, uint h, const char* msg = "")
    {
        auto data = cudaCopyOut(devPtr, pitch, w, h);

        outStream << msg << LogEndl;
        for (uint i = 0; i < h; ++i)
        {
            for(uint j = 0; j < w-1 ; ++j)
                outStream <<  data[j + w*i] << RowDelim; 
            outStream << data[w-1 + w*i]  << LogEndl;
        }

        delete[] data;
        outStream.flush();
    }


    template<typename O, typename T>
    void Cuda(O& outStream, const T& devPtr, uint w, uint h, const char* msg = "")
    {
        auto data = cudaCopyOut(devPtr, w*h);

        outStream << msg << LogEndl;
        for (uint i = 0; i < h; ++i)
        {
            for(uint j = 0; j < w-1 ; ++j)
                outStream <<  data[j + w*i] << RowDelim; 
            outStream << data[w-1 + w*i]  << LogEndl;
        }

        delete[] data;
        outStream.flush();
    }

    template<typename O, typename T>
    void Cuda(O& outStream, const cudaPitchedPtr& devPtr, const char* msg = "")
    {
        uint w = devPtr.xsize, h = devPtr.ysize;
        auto data = cudaCopyOut((T*)devPtr.ptr, devPtr.pitch , w,h);

        outStream << msg << LogEndl;
        for (uint i = 0; i < h; ++i)
        {
            for(uint j = 0; j < w-1 ; ++j)
                outStream <<  data[j + w*i] << RowDelim; 
            outStream << data[w-1 + w*i]  << LogEndl;
        }

        delete[] data;
        outStream.flush();
    }


    template<typename O, typename T>
    void Out(O& outStream, const T& data, uint w, uint h, const char* msg = "")
    {
        outStream << msg << LogEndl;
        for (uint i = 0; i < h; ++i)
        {
            for(uint j = 0; j < w-1 ; ++j)
                outStream <<  data[j + w*i] << RowDelim; 
            outStream << data[w-1 + w*i]  << LogEndl;
        }

        outStream.flush();
    }

    template<typename O, typename T>
    void Out(O& outStream, T** data, uint w, uint h, const char* msg = "")
    {
        outStream << msg << LogEndl;
        for (uint i = 0; i < h; ++i)
        {
            for(uint j = 0; j < w-1; ++j)
                outStream << data[i][j] << RowDelim;
            outStream << data[i][w-1] << LogEndl;
        }

        outStream.flush();
    }

    static TwoDPrint& Instance()
    {
        static TwoDPrint instance('\t');
        return instance;
    }

private:
    TwoDPrint(char delim): RowDelim(delim){}
}; static TwoDPrint TwoDPrinter = TwoDPrint::Instance();

#define cudaDeviceMemset(Name, Value) {          \
    void* __devPtr  = 0 ;                     \
    cudaGetSymbolAddress(&__devPtr, "\""#Name"\""); \
    THROW_IF(cudaMemset(__devPtr,Value,sizeof(Name)), CUDAException, "CudaMemset failed"); \
} 

inline void ShowDevices(int device = -1) // send to an implementation file.
{
    int count = 0 ;
    cudaGetDeviceCount(&count);

    for (int i = 0; i < count; i++)
    {
        if (device != -1 && i != device)
            continue;
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
         Logging::Log
            << LogEndl << "Name:               " <<  prop.name 
            << LogEndl << "Compute capability: " <<  prop.major << "." <<  prop.minor 
            << LogEndl << "Clock rate:           " <<  prop.clockRate 
            << LogEndl << "Device copy overlap:" <<  (prop.deviceOverlap ? "Enabled" : "Disabled")
            << LogEndl << "Kernel timeout :    " <<  (prop.kernelExecTimeoutEnabled ? "Enabled" : "Disabled" )
            << LogEndl << " --- Memory Information for device " << i << "---"
            << LogEndl << "Total global mem:    "<<  prop.totalGlobalMem 
            << LogEndl << "Total constant Mem:  "<<  prop.totalConstMem 
            << LogEndl << "Max mem pitch:       "<<  prop.memPitch 
            << LogEndl << "Texture Alignment:   "<<  prop.textureAlignment 
            << LogEndl << " --- MP Information for device " << i << "---"
            << LogEndl << "Multiprocessor count:  "<<  prop.multiProcessorCount 
            << LogEndl << "Shared mem per mp:     "<<  prop.sharedMemPerBlock 
            << LogEndl << "Registers per mp:      "<<  prop.regsPerBlock 
            << LogEndl << "Threads in warp:       "<<  prop.warpSize 
            << LogEndl << "Max threads per block: "<<  prop.maxThreadsPerBlock 
            << LogEndl << "Max thread dimensions: "<<  prop.maxThreadsDim[0] << "x" <<  prop.maxThreadsDim[1] << "x" << prop.maxThreadsDim[2] 
            << LogEndl << "Max grid dimensions:   " << prop.maxGridSize[0]   << "x" <<  prop.maxGridSize[1]   << "x" << prop.maxGridSize[2] 
            << LogEndl ;
    }
}

//gotta move this to a cpp file
inline void GetGridAndBlockSize2D(uint width, uint height, dim3& gridSize, dim3& blockSize, float factor, int device = 0)
{
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    int maxBlocksX = 0, maxBlocksY = 0, maxThreads = 0;

    cudaDeviceGetAttribute(&maxThreads, cudaDevAttrMaxThreadsPerBlock, device);

    maxThreads = (uint)(maxThreads / Clamp(factor, 32, 1));
    blockSize.x = blockSize.y = (uint)(sqrt(maxThreads)); // try for squarest blocks

    cudaDeviceGetAttribute(&maxBlocksX, cudaDevAttrMaxGridDimX, device);
    cudaDeviceGetAttribute(&maxBlocksY, cudaDevAttrMaxGridDimY, device);

    gridSize.z = blockSize.z = 1;
    blockSize.x = iDivUp(width, gridSize.x);
    blockSize.y = iDivUp(height, gridSize.y);
    while (uint(maxThreads) < (blockSize.x *blockSize.y *blockSize.z))
    {
        ++gridSize.x, ++gridSize.y;
        blockSize.x = iDivUp(width, gridSize.x), blockSize.y = iDivUp(height, gridSize.y);
    }
}

inline int SetLastDevice(bool showSetDevice = false)
{
    int count = 0;
    cudaGetDeviceCount(&count);
    cudaSetDevice(count - 1);
    CUDA_CHECK();
    if (showSetDevice)
    {
        Logging::Log << "Set device with following properties:\n";
        ShowDevices(count-1);
    }
    return count - 1;
}


#endif