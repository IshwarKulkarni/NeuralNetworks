#ifndef __CIFAR_READER_HXX__
#define __CIFAR_READER_HXX__

#include <fstream>
#include <string>
#include "imageprocessing/Color.hxx"
#include "utils/Exceptions.hxx"
#include "utils/utils.hxx"
#include "graphics/geometry/Vec23.hxx"
#include "utils/SimpleMatrix.hpp"
#include "utils/Logging.hxx"

#include <type_traits>

#define CIFARImageLocation "cifar-10-batches-bin\\"

namespace CIFAR
{
    using namespace __IM_Color__;

static const std::string CIFARBinImage[6] = {
    "data_batch_1.bin",
    "data_batch_2.bin",
    "data_batch_3.bin",
    "data_batch_4.bin",
    "data_batch_5.bin",
    "test_batch.bin"
};

static const std::string LableFilename = DATA_LOCATION  CIFARImageLocation "Labels.bin";
static const std::string RGBImagesFilename = DATA_LOCATION  CIFARImageLocation "RGBImages.bin";
static const std::string RGBFloatFilename = DATA_LOCATION  CIFARImageLocation "RGBImagesF.bin";
static const std::string MonochromeFilename = DATA_LOCATION  CIFARImageLocation "Monochrome.bin";
static const std::string MonochromeFloatFilename = DATA_LOCATION  CIFARImageLocation "MonochromeF.bin";

struct ImageReader
{
    static const unsigned ImW = 32, ImH = 32, ImageSizeLin = 1024;
    static const unsigned NumImagesPerOrigFile = 10000;
    static const unsigned NumImages = 60000;
        
    inline ImageReader()
    {
        LableFile.open((LableFilename).c_str(), std::ios::binary),
        RGBImagesFile.open((RGBImagesFilename).c_str(), std::ios::binary),
        RGBImageFloatFile.open((RGBFloatFilename).c_str(), std::ios::binary),
        MonochromeFile.open((MonochromeFilename).c_str(), std::ios::binary),
        MonochromeFloatFile.open((MonochromeFloatFilename).c_str(), std::ios::binary);

        bool filesGood = (LableFile && RGBImagesFile && MonochromeFile && MonochromeFloatFile);
        if(filesGood)
        {
            filesGood &= (Utils::GetStreamSize(LableFile) == NumImages);
            filesGood &= (Utils::GetStreamSize(MonochromeFile) == NumImages*ImageSizeLin);
            filesGood &= (Utils::GetStreamSize(RGBImagesFile) == NumImages*sizeof(Color)*ImageSizeLin);
            filesGood &= (Utils::GetStreamSize(RGBImageFloatFile) == NumImages*sizeof(float)*3*ImageSizeLin);
            filesGood &= (Utils::GetStreamSize(MonochromeFloatFile) == NumImages*sizeof(float)*ImageSizeLin);
        }
        
        if(!filesGood) Convert();
    }

    template<typename T>
    inline typename std::enable_if< 
        std::is_same<typename T, Color>::value || 
        std::is_same<typename T, double>::value ||
        std::is_same<typename T, uchar>::value,
        typename T >::type**
        ImageDataCopy(unsigned N = NumImages, unsigned offset = 0, T** inBuffer = 0)
    {
        THROW_IF(N + offset > NumImages, FileIOException, "Offset and number of images cannot be accomodated");

        if (!inBuffer) inBuffer = Matrix<T>({ ImageSizeLin,N }).data;
        offset *= ImageSizeLin*sizeof(T), N *= ImageSizeLin*sizeof(T);
        
        std::ifstream& in =
            (sizeof(T) == 1 ? MonochromeFile : (sizeof(T) == 3 ? RGBImagesFile : MonochromeFloatFile));
        std::streamsize bytesRead = in.seekg(offset, ios::beg).read((char*)(inBuffer[0]), N).gcount();
        THROW_IF(bytesRead != N, FileIOException, "Not enough bytes read! N = %d, bytesRead = %d", N, uint(bytesRead));
        return inBuffer;
    }

    inline double*** ImageDataRaw(unsigned N = NumImages, unsigned offset = 0)
    {
        THROW_IF(N + offset > NumImages, FileIOException, "Offset and number of images cannot be accomodated");
        Vec::Size3 allimages(ImW, ImH, N * 3); // N*3 = NumFrames
        auto images = SimpleMatrix::ColocAlloc<double>(allimages);
        
        auto bytesPerImage = 3 * sizeof(double)*ImageSizeLin;
        N *= unsigned(bytesPerImage); offset *= unsigned(bytesPerImage);

        auto bytesRead = RGBImageFloatFile.seekg(offset).read((char*)images[0][0], N).gcount();
        THROW_IF(bytesRead != N, FileIOException, "Not enough bytes read! N = %d, bytesRead = %d", N, uint(bytesRead));
        return images;
    }
    
    inline unsigned char* LabelData(unsigned N = NumImages, unsigned offset = 0)
    {
        THROW_IF(offset + N > NumImages, FileIOException, "Too many labels asked for");
        char* buf = new char[N];
        std::streamsize bytesRead = LableFile.seekg(offset, std::ios::beg).read(buf, N).gcount();
        THROW_IF(N != bytesRead, FileIOException, "Not enough labels read! N = %d, bytesRead = %d", N, uint(bytesRead));

        return (unsigned char*)buf;
    }

    private:
            
    static void Convert() 
    {
        Logging::Timer Conversion("CIAR Image conversion");

        const unsigned buffSize = (ImageSizeLin * 3);        
        double            pixel3F[buffSize];
        uchar            bufferIn[buffSize];
        double            pixelF[ImageSizeLin];
        unsigned char   pixelM[ImageSizeLin];
        Color            pixels[ImageSizeLin];
        char            labels[NumImagesPerOrigFile];

        std::ofstream
            oLable(LableFilename, std::ios::binary),
            oRGBImages(RGBImagesFilename, std::ios::binary),
            oRGBFloatImages(RGBFloatFilename, std::ios::binary),
            oMonochrome(MonochromeFilename, std::ios::binary),
            oMonochromeF(MonochromeFloatFilename, std::ios::binary);

        THROW_IF(!(oLable && oRGBImages && oMonochrome && oMonochromeF), FileOpenException,
            "Could not open files to write");

        for (unsigned binNum = 0; binNum < 6; ++binNum)
        {
            std::ifstream in(DATA_LOCATION CIFARImageLocation + CIFARBinImage[binNum], std::ios::binary);

            for (unsigned n = 0; n < NumImagesPerOrigFile && in; ++n)
            {
                in.seekg(buffSize*n + n).get(labels[n]).read((char*)bufferIn, buffSize);
                for (unsigned i = 0; i < ImageSizeLin; ++i)
                {

                    pixels[i] = { bufferIn[i], bufferIn[i + 1024], bufferIn[i + 2048] };

                    float l = float(bufferIn[i] + bufferIn[i + 1024] + bufferIn[i + 2048]) / 3.f;
                    pixelM[i] = unsigned char(l), pixelF[i] = l / 255;
                    pixel3F[i] = float(bufferIn[i]) / 255.f;

                    pixel3F[i] = float(bufferIn[i]) / 255.f;
                    pixel3F[i + 1024] = float(bufferIn[i + 1024]) / 255.f;
                    pixel3F[i + 2048] = float(bufferIn[i + 2048]) / 255.f;
                }

                oRGBImages.write((char*)pixels, sizeof(pixels)).flush();
                oMonochromeF.write((char*)pixelF, sizeof(pixelF)).flush();
                oMonochrome.write((char*)pixelM, sizeof(pixelM)).flush();
                oRGBFloatImages.write((char*)pixel3F, sizeof(pixel3F)).flush();
            }
            oLable.write((char*)labels, NumImagesPerOrigFile);
        }

        bool AllFiledWritten = (oMonochromeF.tellp() == std::streamsize(sizeof(pixelF) * NumImages)) &&
                                (oMonochrome.tellp() == std::streamsize(sizeof(pixelM) * NumImages)) &&
                                 (oRGBImages.tellp() == std::streamsize(sizeof(pixels) * NumImages)) &&
                            (oRGBFloatImages.tellp() == std::streamsize(sizeof(pixel3F) * NumImages));

        THROW_IF(!AllFiledWritten, FileIOException, "Could not write the files correctly");
    }

    std::ifstream LableFile, RGBImagesFile, MonochromeFile, MonochromeFloatFile, RGBImageFloatFile;

};
}
#endif