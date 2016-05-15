#ifndef COLOR_IMAGE_INCLUDED
#define COLOR_IMAGE_INCLUDED

#include "matrix/Matrix.hxx"
#include "ImageGeneric.hxx"
#include "utils/Utils.hxx"
#include "utils/Logging.hxx"
#include "Color.hxx"

#pragma warning(push)
#pragma warning(disable:4996)
#pragma warning(pop)

class ImageRGB : public Matrix2D<Color>
{
    
public:

    typedef Matrix2D<Color> ColorMatrix;

    ImageRGB(const char* file);

    explicit ImageRGB(const ImageGeneric<>& img);

    inline void WriteAsImage(const char* file, ImageIO::ImageFormatType type=ImageIO::PPM)
    {
        ImageIO::WriteImage(file,std::make_pair(Width(),Height()), 3, (byte*)(m_Data), true, type);
    }

    uint* ToIntArray();
    
    ImageRGB(uint width, uint height, const Color& color = White): ColorMatrix(width,height,color){}
    
    ImageRGB(const ImageRGB& img);

private:
};


typedef std::vector<Color> ColorVector;

template<typename Type>
ImageRGB GenerateHeatMap(const Matrix2D<Type>& valueMap, ColorVector& pallet)
{

    unsigned keyWidth = valueMap.Width()/20; 

    if(pallet.size() == 0)
    {
        pallet.push_back(Black);
        pallet.push_back(Blue);
        pallet.push_back(Yellow);
        pallet.push_back(Red);
        pallet.push_back(White);
    }

    auto maxElement = valueMap.GetMaxElem()*1.00001;
    auto minElement = valueMap.GetMinElem()/1.00001;
    auto range      = maxElement - minElement;

    uint w = valueMap.Width()  , h = valueMap.Height();
    ImageRGB heatMap(h,w + keyWidth, White);
    
    uint numIntervals = pallet.size()-1, index;
    float stepWidth = 1.f/numIntervals;
    
    auto interpColor = [&](float relative) {
        index = (uint) ( relative* numIntervals );
        relative -= stepWidth*index;
        relative /= stepWidth;
        return pallet[index]*(1.f-relative) + pallet[index+1]*relative ; 
    };
    uint i,j ; bool  key = false;
    for (i = 0; i < valueMap.Height(); i++){
        try{
            key = false;
            for (j = 0; j < valueMap.Width(); j++)
                heatMap[i*(w+keyWidth) + j] = interpColor(float(valueMap[i*w + j] - minElement)/float(range));
            
            key = true;
            for (j = 0; j < keyWidth; j++)
                heatMap[i*(w+keyWidth) + j + w] = interpColor(float(h-i-1)/h);
        }
        catch ( ... )
        {
            Logging::Log<< "GenHeatMap barfed at " << i << " , " << j  <<  (key? " in key\n" : "\n" );
            throw;
        }
    }
    
    return heatMap;
}
#endif