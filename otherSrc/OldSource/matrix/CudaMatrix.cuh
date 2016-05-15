#ifndef _CUDAMATRIX_CUH_INCLUDED_
#define _CUDAMATRIX_CUH_INCLUDED_

#include "device_launch_parameters.h"
#include "Matrix.hxx"
#include "utils/CudaUtils.hxx"
#include "utils/CudaException.hxx"
#include "utils/Utils.hxx"
#include "CudaOperators.cuh"

#define iDivUp(a, b) (uint)( (a % b) ? (a / b + 1) : (a / b) )

#define IMAD(a, b, c)    ( __mul24((a), (b)) + (c) )
#define IMXY            uint x = IMAD(blockIdx.x,blockDim.x,threadIdx.x), y = IMAD(blockIdx.y,blockDim.y,threadIdx.y);    
#define IMXYCHECK(w,h)  IMXY; if( !(x<w&&y<h) ) return;  // if this aint a Macro abuse, nothing is!

#pragma warning( disable :  4996 )

/*
Sachhidananda #12
Basava Samiti Extension 6th Cross 
(Near Nanda Kishore Bhavana)
Vidyaranyapura, Bangalore 560093
*/

//#define USE_RECURSION
//http://www.orangeowlsolutions.com/archives/199, /MTd
// compute_35,sm_35
// Set in all files': Properties->Cuda C++-> Host-> Runtime Library->/MTd 
template<typename T> struct CuMaxOp  { __device__ __forceinline__ T Apply(const T& t1, const T& t2) const { return (t1 > t2 ? t1 : t2); } };
template<typename T> struct CuMinOp  { __device__ __forceinline__ T Apply(const T& t1, const T& t2) const { return (t1 > t2 ? t2 : t1); } };
template<typename T> struct CuBinAdd { __device__ __forceinline__ T Apply(const T& t1, const T& t2) const { return t1 + t2 ; } };

template<typename T, typename BOp> __global__ 
void reduceVertALittle(const T* vectorsIn, T* vectorsOut, uint length, uint elemPitch, uint rowsPerBlock)
{
    uint y = blockIdx.x*rowsPerBlock,
        x = threadIdx.x;
    const T* lVectorsIn = vectorsIn + y*elemPitch + x ;

    uint lRowsPerBlock = rowsPerBlock;
    if( blockIdx.x == gridDim.x - 1 && length % rowsPerBlock)
        lRowsPerBlock = length % lRowsPerBlock;

    BOp bop;
    T reduced = lVectorsIn[0];
    for (uint i = 1; i < lRowsPerBlock; i++)
        reduced = bop.Apply(lVectorsIn[i*elemPitch], reduced);
    vectorsOut[blockIdx.x * elemPitch + x] = reduced;
}

template<typename T, typename BOp> __global__ 
void reduceVertALittle(cudaTextureObject_t  tex,T* vectorsOut, uint length, uint elemPitch, uint rowsPerBlock)
{
    uint y = blockIdx.x*rowsPerBlock,
        x = threadIdx.x;
    //const T* lVectorsIn = vectorsIn + y*elemPitch + x ;

    uint lRowsPerBlock = rowsPerBlock;
    if( blockIdx.x == gridDim.x - 1 && length % rowsPerBlock)
        lRowsPerBlock = length % lRowsPerBlock;

    BOp bop;
    T val;
    tex2D(&val, tex,x,y);
    T reduced = val;
    for (uint i = 1; i < lRowsPerBlock; i++)
        //reduced = bop.Apply(lVectorsIn[i*elemPitch], reduced);
        tex2D(&val, tex,x,y +i), reduced = bop.Apply(val, reduced);


    vectorsOut[blockIdx.x * elemPitch + x] = reduced;
}

template<typename T, typename BOp> 
void reduceVertALittleHost(dim3 gridDim, dim3 blockDim, const T* vectorsIn, T* vectorsOut, uint length, uint elemPitch, uint rowsPerBlock, uint pageNum = 0 , uint pageWidth = 0)
{

    dim3 blockIdx, threadIdx;
    for(blockIdx.x  = 0 ; blockIdx.x < gridDim.x; blockIdx.x++)
        for(threadIdx.x  = 0 ; threadIdx.x < blockDim.x; threadIdx.x++)
        {
            uint y = blockIdx.x*rowsPerBlock,
                x = threadIdx.x + pageWidth*pageNum;
            const T* lVectorsIn = vectorsIn + y*elemPitch + x ;

            uint lRowsPerBlock = rowsPerBlock;
            if( blockIdx.x == gridDim.x - 1 && length % rowsPerBlock)
                lRowsPerBlock = length % lRowsPerBlock;

            BOp bop;
            T reduced = lVectorsIn[0];
            for (uint i = 1; i < lRowsPerBlock; i++)
                //reduced = bop.Apply(lVectorsIn[i*elemPitch], reduced);
                    if(lVectorsIn[i*elemPitch] < reduced) 
                        reduced = lVectorsIn[i*elemPitch] ;

            vectorsOut[blockIdx.x * elemPitch + x] = reduced;
        }
#ifdef USE_RECURSION
    if( threadIdx.x == blockDim.x - 1 && blockIdx.x == gridDim.x - 1 && gridDim.x != 1  )
    {
        cudaDeviceSynchronize();
        reduceVertALittle <T, BOp> <<<iDivUp(gridDim.x,rowsPerBlock),blockDim >>> 
            (vectorsOut, vectorsOut+gridDim.x*elemPitch, gridDim.x, elemPitch, rowsPerBlock, pageNum, pageWidth);
    }
#endif
}

template<typename OutPutType, typename FillerFun>
__global__ void Create(const FillerFun* devFun, uint width, uint height, OutPutType* out)
{
    IMXYCHECK(width,height);
    devFun->Apply(x,y,out);
}

template<typename OutPutType, typename Modifier>
__global__ void Modify(Modifier* devFun, OutPutType* data, uint elemPitch, uint width, uint height)
{
    IMXYCHECK(width,height);
    data[x + y*elemPitch]  = devFun->Apply(x,y,data[x + y*elemPitch]);
}

template <typename T>
class CuMatrix2D : public Matrix2D<T>
{
    typedef Matrix2D<T> ParentType;
    
public:
    typedef T ElemType;
        
    CuMatrix2D(const ParentType& mat) : ParentType    (mat)
    {
        m_DevData = cudaPitchedAllocCopy(ParentType::m_Data, m_Width, m_Height, m_Pitch);
    }
    
    CuMatrix2D(std::string file, char colDelim = ',',  char rowDelim = '\n') : ParentType    (file, colDelim, rowDelim )
    {
        m_DevData = cudaPitchedAllocCopy(ParentType::m_Data, m_Width, m_Height, m_Pitch);
    }

    CuMatrix2D(uint w, uint h, const T* data=0) : 
        ParentType    (w,h, data)
    {
        m_DevData = cudaPitchedAllocCopy(ParentType::m_Data, m_Width, m_Height, m_Pitch);
    }

    CuMatrix2D(uint w, uint h, const T& val ) : ParentType    (w,h,val) // should be using device to set values.
    {
        m_DevData = cudaPitchedAllocCopy(ParentType::m_Data, m_Width, m_Height, m_Pitch);
    }

    template<typename TypeCastableToT>
    CuMatrix2D(uint w, uint h, const TypeCastableToT* val ) :  ParentType    (w,h,val) // A host pointer
    {
        m_DevData = cudaPitchedAllocCopy(ParentType::m_Data, m_Width, m_Height, m_Pitch);
    }

    CuMatrix2D(const CuMatrix2D& other) : ParentType()
    {
        Matrix2D<T>::m_Width = other.Width()  ;
        Matrix2D<T>::m_Height = other.Height() ;
        m_Pitch        = other.Pitch()  ;
        m_DevData   = other.CopyDevData();
        Matrix2D<T>::m_Data     = cudaCopyOut(m_DevData, m_Pitch, m_Width, m_Height);
    }

    ~CuMatrix2D()
    {
        if( m_DevData)
        {
            LoggedCudaFree(m_DevData);
            m_DevData = 0;
        }
    }
    
    //////////////////////////////////


    virtual inline T* GetData() const   { Sync(); return m_Data;}

    inline const T* GetDevData() const  { return m_DevData; }
    inline         T* GetDevData()        { return m_DevData; }
    
    inline size_t ElemPitch() const {  return m_Pitch/sizeof(T);}
    inline size_t Pitch() const {  return m_Pitch;}
    
    //////////////////////////////////

    void MatGetGridAndBlockSize2D(dim3& gridSize, dim3& blockSize, float factor , int device = 0)
    {
        GetGridAndBlockSize2D(m_Width, m_Height, gridSize, blockSize, factor, device);
    }
    
    template <typename Bop>  // "Bop has a "T Apply(const T& t1, const T& t2)" method
    T* ReduceVertical(float factor = 64.f) const
    {
        int rowsPerBlock = 64, maxThreads;
        cudaDeviceGetAttribute(&maxThreads, cudaDevAttrMaxThreadsPerBlock, 0);
        if( maxThreads > (int)m_Width) maxThreads = m_Width;
        rowsPerBlock  = int(maxThreads/ ::Clamp(factor,128,1.1));
        if(rowsPerBlock < 2)
            rowsPerBlock = 2;

        
        uint outHeight = 0, outHeightOrig, length = m_Height;
        do { uint k = iDivUp(length,rowsPerBlock); outHeight +=k; length = k; } while (length != 1 );
        size_t pitch = 0 ;
        
        outHeightOrig = outHeight;
        T *out = cudaPitchedAllocCopy((T*)(0),m_Width,outHeight, pitch);
        T* outOrig = out;
//        uint numPages = iDivUp(m_Width,maxThreads);
//        uint pageWidth = m_Width/numPages;
        
        T* in = m_DevData;
        length = m_Height;
        int iter = 0;
        dim3 gridDim, blockDim; 
        
        delete[] cudaCopyOut(in, m_Pitch, m_Width, 1);
        do
        {
            outHeight = iDivUp(length,rowsPerBlock);
            reduceVertALittle<T,Bop> <<<outHeight, maxThreads>>>(in, out, length,(uint)ElemPitch(),rowsPerBlock);
            in = out;
            length = outHeight;
            out += outHeight* ElemPitch();
            ++iter;
        }while(outHeight != 1);
        
        auto ret = cudaCopyOut(outOrig + (outHeightOrig-1)*ElemPitch(), m_Pitch, m_Width, 1);
        
        LoggedCudaFree(outOrig);
        return ret;
    }
    
    virtual void SumElems(T& sum) const{
        CUDA_CHECK();
        unique_ptr<T[]> row(ReduceVertical<CuBinAdd<T> >());
        std::accumulate(row.get(),row.get()+Width(),sum);
    }

    virtual inline T GetMaxElem() const 
    {
        CUDA_CHECK();
        unique_ptr<T[]> row(ReduceVertical<CuMaxOp<T> >());
        return *std::max_element(row.get() , row.get()+m_Width);
    }

    virtual inline T GetMinElem() const 
    {
        CUDA_CHECK();
        unique_ptr<T[]> row(ReduceVertical<CuMinOp<T> >());
        return *std::min_element(row.get() , row.get()+m_Width);
    }

    virtual void Convolve(const Matrix2D<float>* kernel) override
    {
        auto tex = BindToTex();
        ConvolutionOperator<T>  oper((uint)ElemPitch(), Width(), Height(), tex, *kernel);

        ConvolutionOperator<T>* devFun = cudaAllocCopy(&oper);
        dim3 gridSize, blockSize;
        MatGetGridAndBlockSize2D(gridSize, blockSize, 4);
        
        cudaPitchedPtr out;
        out.ptr = cudaPitchedAllocCopy<T>(0,Width(),Height(),out.pitch);
        
        out.xsize = Width();
        out.ysize = Height();

        Create <<< gridSize,blockSize >>> (devFun, m_Width, m_Height,(T*)out.ptr);
        
        LoggedCudaFree(devFun)
        cudaDestroyTextureObject(tex);
        
        LoggedCudaFree(m_DevData);

        m_DevData = (T*)(out.ptr);
    }

    template<typename Fun>
    void UnaryFun(const Fun& fun) // Fun has "__device__ T Apply(uint x, uint y, T Val) const"
    {
        Fun* devFun = cudaAllocCopy(&fun);
        
        dim3 gridSize, blockSize;
        MatGetGridAndBlockSize2D(gridSize, blockSize, 4);
        
        Modify<T,Fun> <<<gridSize,blockSize>>>(devFun, m_DevData, ElemPitch(), Width(), Height());
        cudaFree(devFun);
    }

    virtual CuMatrix2D<T>& Clamp(T& low, T& high) // TODO fix this
    {
        //T lowElem = GetMinElem(), hiElem = GetMaxElem();
        //T range = (hiElem - lowElem);
        //high = lowElem + range*high;
        //low  = lowElem + range*low;

        //ForEachElement([&](T* e)
        //{
        //    if (*e < low) 
        //        *e = lowElem; 
        //    if( *e > high) 
        //        *e = hiElem; 
        //    return true;
        //});
        return *this;
    }

    inline virtual T* CopyDevData() const
    {
        size_t p;
        return cudaPitchedAllocDevCopy(m_DevData, m_Width, m_Height, p);
    }

    inline void Sync() const
    {
        //TODO find a way of setting using isHostOOD before using this.
        THROW_IF( cudaMemcpy2D((void*)m_Data, m_Width*sizeof(T),  m_DevData, m_Pitch, m_Width*sizeof(T), m_Height, cudaMemcpyDeviceToHost) , 
            CUDACopyException, "cudaMemcpy failed to copy from ptr = 0x%Zd, and width = %Zd, height = %Zd", m_Data, m_Width, m_Height) ; 
    }

    template<typename U> const CuMatrix2D<T>& operator+=(const U val) { UnaryFun(CuAlgebraicOperScalar<T,OperAdd>(val)); return *this; }
    template<typename U> const CuMatrix2D<T>& operator-=(const U val) { UnaryFun(CuAlgebraicOperScalar<T,OperSub>(val)); return *this; }
    template<typename U> const CuMatrix2D<T>& operator*=(const U val) { UnaryFun(CuAlgebraicOperScalar<T,OperMul>(val)); return *this; }
    template<typename U> const CuMatrix2D<T>& operator/=(const U val) { UnaryFun(CuAlgebraicOperScalar<T,OperDiv>(val)); return *this; }
    template<typename U> const CuMatrix2D<T>& operator^=(const U val) { UnaryFun(CuAlgebraicOperScalar<T,OperPow>(val)); return *this; }
    //const CuMatrix2D<T>& operator^=(float val) { UnaryFun(CuAlgebraicOperScalar<T,OperPow>(val)); return *this; }

    template<typename U> const CuMatrix2D<T>& operator+=(const CuMatrix2D<U>& mat) { UnaryFun(CuAlgebraicOper1Tex<T, OperAdd>(mat.BindToTex())); return *this; }
    template<typename U> const CuMatrix2D<T>& operator-=(const CuMatrix2D<U>& mat) { UnaryFun(CuAlgebraicOper1Tex<T, OperSub>(mat.BindToTex())); return *this; }
    template<typename U> const CuMatrix2D<T>& operator*=(const CuMatrix2D<U>& mat) { UnaryFun(CuAlgebraicOper1Tex<T, OperMul>(mat.BindToTex())); return *this; }
    template<typename U> const CuMatrix2D<T>& operator/=(const CuMatrix2D<U>& mat) { UnaryFun(CuAlgebraicOper1Tex<T, OperDiv>(mat.BindToTex())); return *this; }

    cudaTextureObject_t BindToTex() const
    {
        // create texture object
        cudaResourceDesc resDesc;
        memset(&resDesc, 0, sizeof(resDesc));
        resDesc.resType = cudaResourceTypePitch2D;
        resDesc.res.pitch2D.devPtr = (void*)GetDevData();
        
        resDesc.res.pitch2D.desc = cudaCreateChannelDesc<T>();
        resDesc.res.pitch2D.width = m_Width;
        resDesc.res.pitch2D.height = m_Height;
        resDesc.res.pitch2D.pitchInBytes = m_Pitch;
        
        cudaTextureDesc texDesc;

        texDesc.addressMode[0] = cudaAddressModeClamp,
        texDesc.addressMode[1] = cudaAddressModeClamp;
        texDesc.addressMode[2] = cudaAddressModeClamp;

        memset(&texDesc, 0, sizeof(texDesc));
        texDesc.readMode = cudaReadModeElementType;

        cudaTextureObject_t tex = 0;
        THROW_IF(cudaCreateTextureObject(&tex, &resDesc, &texDesc, NULL), CUDABindException, "Cuda Texture object creation failed");
        return tex;
    }

    cudaPitchedPtr GetPitchedPtr() const
    {
        return{ (void*)GetDevData(), m_Pitch, m_Width, m_Height};
    }

    virtual void Identify(std::ostream& O) const override
    {
        O    << "\nCuMatrix2D: " << " Device Data=0x"
            << std::hex << (unsigned long long)m_DevData << std::dec
            << " Pitch=" << m_Pitch << ", Elem Pitch="
            << ElemPitch() << ". Parent Type, ";
        
        ParentType::Identify(O);
    }

protected:

    bool m_DevOOD, m_HostOOD;
    CuMatrix2D(){};
    T* m_DevData;
    size_t m_Pitch;
};

#endif
