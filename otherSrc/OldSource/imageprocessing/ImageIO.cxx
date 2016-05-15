#include "ImageGeneric.hxx"
#include <fstream>
#include <string>

using namespace std;
using namespace ImageIO;
using namespace StringUtils;

static inline ImageFormatType getIamgeTypeFromName(const string& a)
{
    if(endsWith(a, ".JPEG")) return JPEG;
    if(endsWith(a, ".JPG") ) return JPEG;
    if(endsWith(a, ".JP2") ) return JPEG;
    if(endsWith(a, ".PGM") ) return PPM;
    if(endsWith(a, ".PPM") ) return PPM;

    throw new WringFileTypeException(-1);
}

extern SharedBytes ImageIO::ReadImage(string strIn, bool interleaved, uint& width, uint& height, uint& components)
{
    switch (getIamgeTypeFromName(strIn))
    {
    /*case JPEG:
        {
            JPEGD::jpeg_decoder_file_stream jpregStream;
            THROW_IF(!jpregStream.open(strIn), FileOpenException, "JPEG file open failed");

            auto frames = decompress_jpeg_image_from_stream(&jpregStream, (int*)&width,(int*)&height, (int*)&components, 4);

            THROW_IF(components != 3,WrongNumberOfChannels, "JPEG images are supposed to have 3 channels\n" );

            if(!interleaved)
                return SharedBytes(frames);


            uint pixelsPerComponent = height*width;
            uint numBytes = pixelsPerComponent * components;

            byte* data = new byte[numBytes];

            for( uint i = 0, index = 0 ; i < pixelsPerComponent ; i+=components, ++index)
            {
                data[i]   = frames[index] ;
                data[i+1] = frames[index+pixelsPerComponent];
                data[i+2] = frames[index+2*pixelsPerComponent];
            }

        }*/
    case  PPM:
        {
            auto data = PPMIO::Read(strIn, width, height, components);

            THROW_IF(components != 3,WrongNumberOfChannels, "PPM images are supposed to have 3 channels\n" );

            if( interleaved ) 
                return SharedBytes(data);


            uint pixelsPerComponent = height*width;
            uint numBytes = pixelsPerComponent * components;

            byte* frames = new byte[numBytes];

            for( uint i = 0, index = 0 ; i < numBytes; i+=3, ++index)
            {

                frames[index] = data[i];
                frames[index+pixelsPerComponent] = data[i+1];
                frames[index+2*pixelsPerComponent] = data[i+2];
            }

            delete data;

            return SharedBytes(frames);

        }
    default:
        {
            THROW_IF(true, InvalidImageType, "Type not found");
        }
    }

    return SharedBytes(nullptr);
}

extern void ImageIO::WriteImage(string strIn, const ImageSize& size, const uint numComps, const byte* unformattedData, bool dataIsInterleaved, ImageFormatType format)
{
    switch(format)
    {
    /*case JPEG:
        {
            uint numPixelsPerFrame = size.first*size.second;
            uint numPixels = numPixelsPerFrame*numComps;
            
            byte* formattedData = nullptr;
            
            if(dataIsInterleaved)
            {
                formattedData = new byte[numPixels];
                for( uint i = 0, index = 0; i < numPixels; i+=numComps, ++index)
                    for(uint c = 0; c <numComps; ++c)
                        formattedData[index + numPixelsPerFrame*c]  = unformattedData[i+c];

            }


            jpge::compress_image_to_jpeg_file(strIn, size.first, size.second, numComps,dataIsInterleaved ? formattedData : unformattedData);
            
            if(formattedData)
                delete formattedData;
            
            break;
        }*/
    case PPM:
        {
            uint numPixelsPerFrame = size.first*size.second;
            uint numPixels = numPixelsPerFrame*numComps;
            
            byte* formattedData = nullptr;
            uint count = 0 ;

            if(!dataIsInterleaved)
            {
                formattedData = new byte[numPixels];
                for( uint i = 0, index = 0; i < numPixels; i+=numComps, ++index)
                    for(uint c = 0; c <numComps; ++c)
                        formattedData[i+c]  = unformattedData[index + numPixelsPerFrame*c];
            }

            PPMIO::Write(strIn, size.first, size.second,numComps, dataIsInterleaved ? unformattedData : formattedData);

            if(formattedData)
                delete formattedData;
            
            break;
        }
    default:
        {
            THROW_IF(true, InvalidImageType, "Type not found");
        }
    }
}