#ifndef _MEDIAN_FILTER_INCLUDED_
#define _MEDIAN_FILTER_INCLUDED_

#include "cuda_runtime.h"
#include "CudaMatrix.cuh"
#include "CudaUtils.hxx"
#include "Sort.cuh"
#include "Logging.hxx"


__shared__ int SHARED_MEMORY[];
template<typename T, typename MatrixType>
struct MedianFilterOperator
{
    uint ElemPitch, Width, Height;
    uint FiltWidth, FiltHeight;
    cudaTextureObject_t Tex;

    MedianFilterOperator(MatrixType* mat, uint fw, uint fh):
        ElemPitch(mat->ElemPitch()),
        Width(mat->Width()),
        Height(mat->Height()),
        Tex(mat->BindToTex()),
        FiltWidth(fw*2+1),
        FiltHeight(fh*2+1)
    {
    } 
    
    __forceinline__ __device__
    void Apply(uint pixX, uint pixY, T* result) const 
    {
        // __shared__ pointers should be of size blocDim.x*blockDim.y*FiltWidth*FiltHeight*sizeof(T)
        
        uint threadId = IMAD(threadIdx.y, blockDim.x, threadIdx.x);
        uint filtSize = FiltWidth * FiltHeight;
        T* staging = (T*)(&SHARED_MEMORY) + filtSize*threadId; 

        uint k = 0;
        for ( uint kx = 0; kx < FiltWidth;  ++kx )
        for ( uint ky = 0; ky < FiltHeight; ++ky )
            {
                uint imX = pixX+ kx - FiltWidth/2, imY = pixY + ky - FiltHeight/2;
                uint sy = IMAD(FiltWidth,ky,kx);
                if( imX < Width && imY < Height) 
                    staging[k++] = Get(imX,imY);
            }
        
        QuickSort(staging,k);
        
        T med = staging[k/2];

        result[pixX + pixY * ElemPitch] = med;
    }

    __forceinline__ __device__
    void ApplyNew(uint pixX, uint pixY, T* result) const 
    {
        // __shared__ pointers needs to be of size blocDim.x*blockDim.y*FiltWidth*FiltHeight*sizeof(T)
        
        uint threadId = IMAD(threadIdx.y, blockDim.x, threadIdx.x);
        uint filtSize = FiltWidth * FiltHeight;
        T* staging = (T*)(&SHARED_MEMORY) + filtSize*threadId; 

        uint k = 0;
        for ( uint kx = 0; kx < FiltWidth;  ++kx )
        for ( uint ky = 0; ky < FiltHeight; ++ky )
            {
                uint imX = pixX+ kx - FiltWidth/2, imY = pixY + ky - FiltHeight/2;
                uint sy = IMAD(FiltWidth,ky,kx);
                if( imX < Width && imY < Height) 
                    staging[k++] = Get(imX,imY);
            }
        
        QuickSort(staging,k);
        
        T med = staging[k/2];

        result[pixX + pixY * ElemPitch] = med;

        if(pixX == Width -1 )
            return;

        int movRight = 0;

        uint imX = pixX - FiltWidth/2;
        if( imX < Width )
        for ( uint ky = 0; ky < FiltHeight; ++ky )
        {
            uint imY = pixY + ky - FiltHeight/2;
            if( imY < Height) 
                if(Get(imX,imY) < med)
                    ++movRight;
                else
                    --movRight;
        }
        
        imX = pixX + FiltWidth/2;
        if( imX < Width )
        for ( uint ky = 0; ky < FiltHeight; ++ky )
        {
            uint imY = pixY + ky - FiltHeight/2;
            if( imY < Height) 
                if(Get(imX,imY) < med)
                    --movRight;
                else
                    ++movRight;
        }
        
        result[pixX + pixY * ElemPitch + 1] = staging[(k + movRight)/2];
    }

    ~MedianFilterOperator()
    {
        cudaDestroyTextureObject(Tex);
    }

private:
    __forceinline__ __device__ T Get(uint x, uint y) const
    {
        T val = 0;
        tex2D(&val, Tex, x, y);
        return val;    
    }
};


template<typename OutPutType, typename FillerFun>
__global__ void CreateNew(const FillerFun* devFun, uint width, uint height, OutPutType* out)
{
    IMXYCHECK(width,height);
    if( x*2 < width)
        devFun->ApplyNew(x*2,y,out);
}



template<typename T, bool useNew>
void MedianFilter(ImageGeneric< CuMatrix2D<T> >& cuImg, std::vector<CuMatrix2D<T>*>& newFrames, uint fw = 3, uint fh = 3)
{

    LogVar(useNew);

    dim3 gridSize, blockSize;
    uint sharedSize = 0 ;
    float threadsFactor = 0.87;
    int sharedSizeAllowed = 0 ;
    cudaDeviceGetAttribute(&sharedSizeAllowed, cudaDevAttrMaxSharedMemoryPerBlock, 0);

    fw = Clamp(fw,6,1);
    fh = Clamp(fh,6,1);

    do
    {
        GetGridAndBlockSize2D(cuImg.Width(), cuImg.Height(), gridSize, blockSize, threadsFactor*=1.15f); // exponentially reduce the number of threads per block
        if(useNew) blockSize.x = iDivUp(blockSize.x, 2);
        sharedSize = (2*fw+1)*(2*fh+1)*blockSize.x*blockSize.y*sizeof(T);

    }while(sharedSize > sharedSizeAllowed);
    
    {
        Logging::Timer filter("Median Filter");
        for (uint i = 0; i < cuImg.GetNumChannels(); i++)
        {
            MedianFilterOperator<T, CuMatrix2D<T> > filler(cuImg.GetChannel(i),fw,fh);
            auto devFiller = cudaAllocCopy(&filler);

            cudaPitchedPtr out;
            out.xsize = cuImg.Width(), out.ysize = cuImg.Height();
            
            cudaPitchedAllocCopy<T>(out);
            
            if(useNew)
                CreateNew<<<gridSize, blockSize, sharedSize>>>(devFiller, cuImg.Width(), cuImg.Height(), (T*)out.ptr);
            else
                Create<<<gridSize, blockSize, sharedSize>>>(devFiller, cuImg.Width(), cuImg.Height(), (T*)out.ptr);
            CUDA_CHECK();
            newFrames.push_back( new CuMatrix2D<T>( out ) ) ;
            
            cudaFree(devFiller);
            
        }
    }
        
    LogVar(gridSize);
    LogVar(blockSize);
    LogVar(sharedSize);
    LogVar((2*fw+1)*(2*fh+1));

}


/*


Use thus:
    ImageGeneric<CuMatrix2D<uchar> > cuImg("E:\\img.ppm");
    std::vector<CuMatrix2D<uchar>*>    newFrames;
    MedianFilter<uchar, false> (cuImg, newFrames,i,i);
    ImageGeneric<CuMatrix2D<uchar> > medianImage(newFrames);
    

insertion sort, fw = fh = 3 (7x7 window)
>> gridSize = 61x61x1( 3721 ) >> blockSize = 17x13x1( 221 ) : T = float        Duration was 0.69 seconds.  -> float compare is slow
>> gridSize = 70x70x1( 4900 ) >> blockSize = 15x11x1( 165 ) : T = float        Duration was 0.90 seconds.  -> float compare and memory intensive

>> gridSize = 70x70x1( 4900 ) >> blockSize = 15x11x1( 165 ) : T = int        Duration was 0.78 seconds.   -> sizeof(int) == sizeof(float), but int compare faster, SM vs. MIO
>> gridSize = 70x70x1( 4900 ) >> blockSize = 15x11x1( 165 ) : T = uint        Duration was 0.82 seconds.   -> Signed is better

>> gridSize = 41x41x1( 1681 ) >> blockSize = 25x19x1( 475 ) : T = ushort    Duration was 0.38 seconds.   -> reading with shared memory filled : int > short > char 

>> gridSize = 54x54x1( 2916 ) >> blockSize = 19x15x1( 285 ) : T = uchar        Duration was 0.22 seconds. |
>> gridSize = 70x70x1( 4900 ) >> blockSize = 15x11x1( 165 ) : T = uchar        Duration was 0.21 seconds. | -> num  threads does not matter if all char
>> gridSize = 29x29x1( 841 )  >> blockSize = 36x27x1( 972 ) : T = uchar        Duration was 0.21 seconds. |     

>> gridSize = 32x32x1( 1024 ) >> blockSize = 32x24x1( 768 ) : T = uchar        Duration was 0.26 seconds.   -> anomaly

>> gridSize = 86x86x1( 7396 ) >> blockSize = 12x9x1( 108 )  : T = uchar        Duration was 0.61 seconds.  -> very few threads, slow, memory intensive (vs. compute)
*/


#endif
