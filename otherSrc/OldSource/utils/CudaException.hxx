#ifndef __CUDA_EXCEPTION__
#define __CUDA_EXCEPTION__

#include "Logging.hxx"
#include "Exceptions.hxx"
#include <cuda_runtime.h>

static char* __CudaErrorMap[] =  {
"cudaSuccess" ,
"cudaErrorMissingConfiguration" ,
"cudaErrorMemoryAllocation" ,
"cudaErrorInitializationError" ,
"cudaErrorLaunchFailure" ,
"cudaErrorPriorLaunchFailure" ,
"cudaErrorLaunchTimeout" ,
"cudaErrorLaunchOutOfResources" ,
"cudaErrorInvalidDeviceFunction" ,
"cudaErrorInvalidConfiguration" ,
 "cudaErrorInvalidDevice" ,
 "cudaErrorInvalidValue" ,
 "cudaErrorInvalidPitchValue" ,
 "cudaErrorInvalidSymbol" ,
 "cudaErrorMapBufferObjectFailed" ,
 "cudaErrorUnmapBufferObjectFailed" ,
 "cudaErrorInvalidHostPointer" ,
 "cudaErrorInvalidDevicePointer" ,
 "cudaErrorInvalidTexture" ,
 "cudaErrorInvalidTextureBinding" ,
 "cudaErrorInvalidChannelDescriptor" ,
 "cudaErrorInvalidMemcpyDirection" ,
 "cudaErrorAddressOfConstant" ,
 "cudaErrorTextureFetchFailed" ,
 "cudaErrorTextureNotBound" ,
 "cudaErrorSynchronizationError" ,
 "cudaErrorInvalidFilterSetting" ,
 "cudaErrorInvalidNormSetting" ,
 "cudaErrorMixedDeviceExecution" ,
 "cudaErrorCudartUnloading" ,
 "cudaErrorUnknown" ,
 "cudaErrorNotYetImplemented" ,
 "cudaErrorMemoryValueTooLarge" ,
 "cudaErrorInvalidResourceHandle" ,
 "cudaErrorNotReady" ,
 "cudaErrorInsufficientDriver" ,
 "cudaErrorSetOnActiveProcess" ,
 "cudaErrorInvalidSurface" ,
 "cudaErrorNoDevice" ,
 "cudaErrorECCUncorrectable" ,
 "cudaErrorSharedObjectSymbolNotFound" ,
 "cudaErrorSharedObjectInitFailed" ,
 "cudaErrorUnsupportedLimit" ,
 "cudaErrorDuplicateVariableName" ,
 "cudaErrorDuplicateTextureName" ,
 "cudaErrorDuplicateSurfaceName" ,
 "cudaErrorDevicesUnavailable" ,
 "cudaErrorInvalidKernelImage" ,
 "cudaErrorNoKernelImageForDevice" ,
 "cudaErrorIncompatibleDriverContext" ,
 "cudaErrorPeerAccessAlreadyEnabled" ,
 "cudaErrorPeerAccessNotEnabled" ,
 "cudaErrorDeviceAlreadyInUse" ,
 "cudaErrorProfilerDisabled" ,
 "cudaErrorProfilerNotInitialized" ,
 "cudaErrorProfilerAlreadyStarted" ,
 "cudaErrorProfilerAlreadyStopped" ,
 "cudaErrorAssert" ,
 "cudaErrorTooManyPeers" ,
 "cudaErrorHostMemoryAlreadyRegistered" ,
 "cudaErrorHostMemoryNotRegistered" ,
 "cudaErrorOperatingSystem" ,
 "cudaErrorPeerAccessUnsupported" ,
 "cudaErrorLaunchMaxDepthExceeded" ,
 "cudaErrorLaunchFileScopedTex" ,
 "cudaErrorLaunchFileScopedSurf" ,
 "cudaErrorSyncDepthExceeded" ,
 "cudaErrorLaunchPendingCountExceeded" ,
 "cudaErrorNotPermitted" ,
 "cudaErrorNotSupported"} ;

class CUDAException: public ISystemExeption
{
public:
    CUDAException(int err) : ISystemExeption(err)
    {
        Logging::Log << "\n>>> Cuda call returned " ;
        if( err < sizeof(__CudaErrorMap)/sizeof(__CudaErrorMap[0]) )
            Logging::Log << __CudaErrorMap[err] << ", ";
        else                                           
            Logging::Log << "an unknown error " << err << ", "; 
        Logging::Log << cudaGetErrorString((cudaError_t)(err)) << ".\n";
    }
};     


#define CUDA_CHECK()  { cudaDeviceSynchronize(); \
    THROW_IF(cudaGetLastError(), CUDAException, "A previous asynchronous CUDA call failed"); }

#define CUDA_CHECK_MSG(s, ...)  { cudaDeviceSynchronize(); \
    THROW_IF(cudaGetLastError(), CUDAException, s, __VA_ARGS__); }


EXCEPTION_TYPE(CUDAAllocException,             CUDAException )
EXCEPTION_TYPE(CUDA2DAllocException,        CUDAAllocException )
EXCEPTION_TYPE(CUDACopyException,            CUDAException )
EXCEPTION_TYPE(CUDABindException,            CUDAException )
EXCEPTION_TYPE(CUDAIncorrectType,            CUDAException )
EXCEPTION_TYPE(CUDAMemsetException,            CUDAException )
EXCEPTION_TYPE(CUDAFreeException,            CUDAException )
EXCEPTION_TYPE(CUDAHWException,                CUDAException )
EXCEPTION_TYPE(CUDASizeException,            CUDAHWException)

EXCEPTION_TYPE(CUDAGlewSupportException,    CUDAHWException)

EXCEPTION_TYPE(CUDASymCopyExpection,        CUDAException )
EXCEPTION_TYPE(CUDADeviceIncapable,            CUDAException )

#define CU_CHECK_THROW(condition){                    \
    auto __cx = (condition);                        \
    if( __cx ) {                                    \
        CUDAException __CU_ERR(__cx);                \
        std::cerr                                    \
            << ">>> Error in " << __FUNCTION__        \
            << " in " << __FILE__                    \
            << ".\n>>  Error: "                        \
            << " returned on evaluation of "        \
            << #condition << LogEndl                \
            ;std::cerr.flush();                        \
            throw __CU_ERR;                            \
    }                                                \
}        


#endif