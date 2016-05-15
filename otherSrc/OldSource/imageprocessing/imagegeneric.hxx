#if defined(_MSC_VER) && _MSC_VER >= 1400  // stop nanny'in' me
#pragma warning(push) 
#pragma warning(disable:4996) 
#endif

#ifndef IMAGEUTILS_INCLUDED
#define IMAGEUTILS_INCLUDED

#define SQR(x) ((x)*(x))

#include "matrix/Matrix.hxx"
#include "imageprocessing/Filter.hxx"
#include "utils/Logging.hxx"
#include "utils/Utils.hxx"

#include <memory>
#include <string.h>
#include <vector>

typedef unsigned char byte;
typedef std::shared_ptr<byte> SharedBytes;

typedef std::pair<uint, uint> ImageSize;
typedef std::pair<uint, uint> UintPair;

typedef float ImageType;

EXCEPTION_TYPE(ImageException,        ISystemExeption);
EXCEPTION_TYPE(WrongNumberOfChannels, ImageException);
EXCEPTION_TYPE(WringFileTypeException, FileIOException);
EXCEPTION_TYPE(InvalidImageType,      FileIOException);
EXCEPTION_TYPE(CorruptFile,           FileIOException);

namespace PPMIO
{
    extern void Write(string filename, uint width, uint height, uint numComponents, const byte* frames);

    extern byte* Read(string filename, uint& width, uint& height, uint& components);
}

namespace ImageIO
{
    extern SharedBytes ReadImage(string filename, bool interleaved, uint& width, uint& height, uint& components);
    
    extern void WriteImage(string filename, const ImageSize& size, const uint numComponents, const byte* unformattedData, bool dataIsInterleaved, ImageFormatType format);
}

struct CropParam
{
    enum CropParamPosition
    {
        CENTER = 1,
        TOP_RIGHT, 
        TOP_LEFT,
        BOTTOM_RIGHT, 
        BOTTOM_LEFT,
        CUSTOM
    };


    CropParam(float clipToVRatio= .9, float clipToHRatio = .9, CropParamPosition pos = CENTER) : 
        Position(CENTER), VRatio (Utils::Clamp(clipToVRatio,1,0)) , HRatio(Utils::Clamp(clipToHRatio, 1,0)){}

    void GetStartsEnds(uint w, uint h, uint& startX, uint& startY, uint& endX, uint& endY)
    {
        switch(Position)
        {

        case  CENTER :
            startX  = uint(0.5*(1 - HRatio)*w);
            startY  = uint(0.5*(1 - VRatio)*h);
            endX    = uint(w*HRatio);
            endY    = uint(h*VRatio);    
            break;
        default:
            THROW_IF(true, InvalidOptionException, "Wrong Crop Position Option provided");
        }
    }

    CropParamPosition Position;
    float VRatio, HRatio;

};

template <typename ImageFrameType = Matrix2D<ImageType> >
class ImageGeneric
{

public:  

    typedef ImageFrameType FrameType;
    typedef std::vector<ImageFrameType*> FrameVector;
    typedef typename ImageFrameType::ElemType Type;

    explicit ImageGeneric(const FrameVector& frames):Channels(frames), Id(GetNextCounter()) {}

    explicit ImageGeneric(std::string filename) : Id(GetNextCounter())
    {
        THROW_IF( filename.length() == 0,  FileIOException, "File name is null");
        uint width, height, numComps;
        auto bytes = ImageIO::ReadImage(filename, false, width, height, numComps);

        uint numPixels = width*height;
        Channels.resize(numComps);

        for(uint i = 0; i < numComps; ++i )
            Channels[i] = new ImageGeneric::FrameType(width,height, bytes.get() + i*numPixels);
    }
        
    template <typename OtherType>
    ImageGeneric(OtherType* data, uint width, uint height, uint numComps): Id(GetNextCounter())
    {
        uint numPixels  = width*height;
        Channels.resize(numComps);

        for( uint i = 0; i < (uint)numComps; ++i )
            Channels[i] = new ImageFrameType(data + i*numPixels,width,height);
    }

    ImageGeneric(ImageFrameType* frame): Id(GetNextCounter()), Channels(FrameVector(1,frame)) {}

    ImageGeneric& ImageGeneric::FlattenToGrayScale()
    {
        for (uint i = 1; i < Channels.size(); i++)
        {
            *Channels[0] += *(Channels[i]);
            delete Channels[i];
        }
    
        Channels.resize(1);
        return *this;
    }

    ImageGeneric& ImageGeneric::FlattenRGBToGrayScale()
    {
        bool isFloatingPoint = std::is_floating_point<typename ImageFrameType::ElemType>::value ;
        THROW_IF(GetNumChannels()!=3 && isFloatingPoint, InvlidImageFormatException, 
        "RGB Flatten can only be applied to 3 channel RGB image of floating points assign *this to a ImageGenric<Matrix2D<float> >" );
        
        *Channels[0] *= 0.2989f;
        *Channels[1] *= 0.5870f;
        *Channels[2] *= 0.1140f;

        for (uint i = 1; i < Channels.size(); i++)
        {
            *Channels[0] += *(Channels[i]);
            delete Channels[i];
        }

        Channels.resize(1);
        return *this;
    }


    Type* ImageGeneric::CloneFrameData(uint i) const
    {
        uint numPixels = GetNumPixels();
        Type* retData = new Type[numPixels];

        auto data = Channels[i]->GetData();

        std::copy(data, data+numPixels, retData);

        return retData;
    }


    ImageGeneric& Crop(CropParam p)
    {
        uint startX, startY, width, height;
        p.GetStartsEnds(Width(), Height(), startX, startY, width, height);

        for (uint i = 0; i < GetNumChannels(); i++)
        {
            auto *f = GetChannel(i);
            Type* data = new Type[width*height];
            uint  fWidth =  f->Width();
            Type* fData = f->GetData();
            for (uint j = 0; j < height; j++)
                copy( fData + j * fWidth, fData + j * fWidth+ width , data + j*width);
            
            SetFrame( new Matrix2D<Type>( width, height, data) , i);
        }
        return *this;
    }

    ImageGeneric& ImageGeneric::operator=(const ImageGeneric& rhs) 
    {
        
        ClearData();

        Channels.resize(rhs.GetNumChannels());

        for (uint i = 0; i < rhs.GetNumChannels(); i++)
            Channels[i] = new FrameType(*rhs.GetFrame(i));

        return *this;
    }

    ImageGeneric(const ImageGeneric& rhs) : Id(GetNextCounter())
    {        
        ClearData();

        Channels.resize(rhs.GetNumChannels());

        for (uint i = 0; i < rhs.GetNumChannels(); i++)
            Channels[i] = new FrameType(rhs(i));
    }

    ImageGeneric& ImageGeneric::Scale(Type low, Type high )
    {
        Type maxElement = std::numeric_limits<Type>().min();
        Type minElement = std::numeric_limits<Type>().max();

        for(uint i = 0; i < Channels.size(); ++i)
        {
            auto maxE = Channels[i]->GetMaxElem();
            auto minE = Channels[i]->GetMinElem();
            maxElement = std::max(maxElement, maxE);
            minElement = std::min(minElement, minE);
        }

        if( maxElement == high && minElement == low)
            return *this;

        for(uint i = 0; i < GetNumChannels(); ++i)
        {
            (*Channels[i]) -= (low - minElement);

            (*Channels[i]) *= (high / maxElement);
        }
        return *this;
    }

    ImageGeneric& ImageGeneric::Clamp(Type low,  Type high)
    {
        for(auto imageFrame : Channels)
            imageFrame->Clamp(low,high);
        return *this;
    }

    const ImageGeneric& Write( std::string filename, 
        const ImageIO::ImageFormatType& format = ImageIO::PPM,
        bool normalizeToUChar = true) const
    {
        auto copyImage = *this;

        auto size = GetImageSize();
        auto numPixels = size.first*size.second;
        
        if(normalizeToUChar)
            copyImage.Scale(0,255);

        byte* out = new byte[numPixels*GetNumChannels()];
        byte* start = out;
        for(FrameType* imageFrame : copyImage.GetChannels())
            out = std::transform(imageFrame->GetData(), 
                imageFrame->GetData()+numPixels, out, 
                [](const Type& f) { return (byte)(f);});
        
        ImageIO::WriteImage(filename, size, GetNumChannels(), start, false, format);
        delete start;
        
        return *this;
    }
    

    inline uint Width() const { return Channels[0]->Width();}

    inline uint Height() const { return Channels[0]->Height();}

    inline uint GetNumPixels() const { return GetNumChannels()*Channels[0]->NumElems(); }

    inline const FrameVector&     GetChannels()const            { return Channels;        } 
    
    inline         ImageFrameType* GetChannel(uint c)            { return Channels[c];    } 
    inline const ImageFrameType* GetChannel(uint c) const   { return Channels[c];    } 
    
    inline         ImageFrameType& operator()(uint c)            { return (*Channels[c]);} // same as above, but return reference
    inline const ImageFrameType& operator()(uint c) const   { return (*Channels[c]);} 
    
    inline ImageSize GetImageSize() const  {  return ( Channels.size() ? UintPair(Channels[0]->Width(), Channels[0]->Height()) : UintPair(0,0)); }

    inline uint GetNumChannels() const  { return (uint)Channels.size(); }

    inline void SetFrame(ImageFrameType* frame, uint i)
    {
        THROW_IF(i>Channels.size(), WrongNumberOfChannels, "You can grow frames only one at a time");
        if( i == Channels.size() )
        {
            Channels.push_back(frame);
        }
        else
        {
            delete Channels[i];
            Channels[i] = frame;
        }
    }

    template<typename FilterParamType>
    ImageGeneric& ApplyFilter(const FilterParamType& param)
    {
        for(ImageFrameType* c: Channels)
            for(uint i = 0; i < param.NumComponents; ++i)
            {
                auto v = param.GetFilterValues(i)    ;
                c->Convolve(v);
                delete v;
            }

        return *this;
    }

    inline void ClearData()
    {
        for(auto c : Channels)
            delete c;

        Channels.clear();
    }
    
    virtual void Identify(std::ostream& O) const
    {
        O    << "Image Generic ID = " << Id << " made up of Matrices: " ;
        for(auto c : Channels)
             O << c->GetId() << ", " ;
        
        O    <<  "Dimension=(" << Width() << "x" << Height() << ")\n" ;
    }
    
    ~ImageGeneric()
    {
        ClearData();
    }
protected:
    
    unsigned Id;
    static unsigned GetNextCounter() { static unsigned id = 0 ; return id++; }
    ImageGeneric(){};

    FrameVector Channels;
};

#endif
