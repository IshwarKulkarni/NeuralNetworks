#ifndef IMAGE_PROCESSING
#define IMAGE_PROCESSING


#include "Logging.hxx"
#include "ImageGeneric.hxx"
#include "ImageRGB.hxx"
#include "BinaryMatrix.hxx"
#include "Matrix.hxx"

using namespace ImageIO;
typedef std::vector<short int> ShortVector ;
using namespace Logging;


namespace ImageProcessing
{
    template<typename MatrixType>
    void Gamma(ImageGeneric<MatrixType>& img, float gamma)
    {   
        img.Scale(0,1);

        for(uint i = 0 ; i < img.GetNumChannels(); ++i)
            (*img.GetChannel(i))^=gamma;
    }

    template<typename MatrixType>
    void Sharpen(ImageGeneric<MatrixType>& img, uint rad,  float sigma)
    {
        ImageGeneric<MatrixType> forAddition = img;

        img.ApplyFilter(Gaussian2DParameters(rad,rad,sigma));

        img.Scale(0,1000.f);

        struct InvertOper{ 
            typedef typename MatrixType::ElemType T;
#ifdef __CUDA_ARCH__
            __forceinline__ __host__ __device__  
#else
            inline
#endif
            T operator()(uint x, uint y,  T elem) { return elem = 1000.f - elem; }

        } oper;

        img.Scale(0,400);

        for(uint i = 0 ; i < img.GetNumChannels(); ++i)
            img(i).UnaryFun(oper);

        forAddition.Scale(0,100);

        for(uint i = 0 ; i < img.GetNumChannels(); ++i)
            (*img.GetChannel(i)) += (*forAddition.GetFrame(i));

        img.ApplyFilter(SharpenFilterParameters());
    }

    template<typename T>
        struct ContrastOperator 
            { float factor; T operator()(unsigned x, unsigned y, T val)  { return COLORCLAMP((factor * val - 128) + 128);  }  };
    
    
    template<typename MatrixType>
    void AdjustContrast(ImageGeneric<MatrixType> & img, double contrast)
    {
        img.Scale(0,255);

        
        ContrastOperator<typename MatrixType::ElemType> oper;

        contrast *= 255*Clamp(1-contrast,1,-1);
        oper.factor = (259 * (contrast + 255)) / (255 * (259 - contrast));

        for(uint i = 0 ; i < img.GetNumChannels(); ++i)
            img(i).UnaryFun(oper);
    }

    template<typename T>
    ImageGeneric<T> EdgeEnhance(ImageGeneric<T> edgesV)
    {
        Timer EdgeEnhance("EdgeEnhance");

        edgesV.FlattenRGBToGrayScale();

        edgesV.ApplyFilter(Gaussian2DParameters(3,3,2.5));

        auto edgesH = edgesV;
        
        edgesV.ApplyFilter(SobelParameters(FilterParameters::VERTICAL));
        edgesH.ApplyFilter(SobelParameters(FilterParameters::HORIZONTAL));

        edgesV(0) ^= 2 ;
        edgesH(0) ^= 2 ;
        
        edgesV(0) += edgesH(0);

        edgesV(0) ^= 0.5;

        return edgesV;
    }
}// namespace ImageProcessing

#endif

