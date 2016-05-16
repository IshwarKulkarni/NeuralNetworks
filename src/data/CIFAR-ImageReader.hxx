#ifndef __CIFAR_READER_HXX__
#define __CIFAR_READER_HXX__

#include <fstream>
#include <string>
#include <algorithm>
#include "utils/utils.hxx"
#include "utils/Vec23.hxx"
#include "utils/SimpleMatrix.hxx"

namespace CIFAR
{
    static const size_t ImW = 32, ImH = 32;
    static const size_t ImD = 3;
    static const size_t ImageSizeLin = ImW*ImH*ImD;
    static const size_t NumImagesPerOrigFile = 10000;
    static const size_t NumImages = 60000;

    struct CIFARReader
    {

    private:

        const std::vector<std::string> CIFARBinImages = { "data_batch_1.bin", "data_batch_2.bin", "data_batch_3.bin",  
                                                          "data_batch_4.bin", "data_batch_5.bin", "test_batch.bin" };

        const std::string  CIFARImageLocation = "cifar-10-batches-bin\\";

        std::string DataLocation;

    public:

        const std::string ClassNames[10] = {  "airplane", "automobile",   "bird", "cat",  "deer",
                                              "dog",      "frog",         "horse","ship", "truck" };

        CIFARReader(std::string dataLocation = DATA_LOCATION) : DataLocation(dataLocation) {};

        // Returns a Volume(ImW,ImH,ImD*N), where ith frame is (i%ImD)th channel of (Floor(i/ImD))th image
        // Not a fourD image.
        inline std::pair<unsigned char***, char*> ImageDataCopy(size_t N = NumImages)
        {
            if (N > NumImages)
                throw std::invalid_argument("Offset and number of images cannot be accomodated");

            Vec::Size3 allimages(ImW, ImH, N * ImD); 
            auto images = SimpleMatrix::ColocAlloc<unsigned char>(allimages); // [0-1] scaled version of images

            size_t numRead = 0;
            auto labels = new unsigned char[N];

            for (auto& binImgName : CIFARBinImages)
            {
                size_t imFromThisBin = std::min(NumImagesPerOrigFile, N - numRead);

                std::string filename = DataLocation + CIFARImageLocation + binImgName;
                std::ifstream inFile(filename, std::ios::binary);
                if (!inFile) throw std::invalid_argument("File read failed for " + filename);

                for (size_t binIm = 0; binIm < imFromThisBin; ++binIm, ++numRead)
                {
                    inFile.read((char*)(labels + numRead), 1);
                    inFile.read((char*)(images[numRead][0]), ImageSizeLin).gcount(); // +1 for the label
                }

                if (imFromThisBin < NumImagesPerOrigFile) break; // we read  all images except from last file/
            }

            return std::make_pair(images, (char*)(labels));
        }

        void TestCIFARReader(size_t N = 100)
        {
            auto imBuffer = new unsigned char[ImageSizeLin];
            auto imSet = ImageDataCopy();

            for (size_t i = 0; i < N; ++i)
            {
                size_t rand = Utils::URand(NumImages);

                unsigned char** imPtrPtr = imSet.first[rand];
                PPMIO::Write(std::to_string(i) + "_" + ClassNames[imSet.second[rand]],  ImW, ImH, ImD, *imPtrPtr, false);
                
            }

            delete[] imBuffer;
            delete[] imSet.second;
            SimpleMatrix::deleteColocArray(imSet.first);
        }

    };
}
#endif
