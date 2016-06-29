/*
Copyright (c) Ishwar R. Kulkarni
All rights reserved.

This file is part of NeuralNetwork Project by 
Ishwar Kulkarni , see https://github.com/IshwarKulkarni/NeuralNetworks

If you so desire, you can copy, redistribute and/or modify this source 
along with  rest of the project. However any copy/redistribution, 
including but not limited to compilation to binaries, must carry 
this header in its entirety. A note must be made about the origin
of your copy.

NeuralNetwork is being distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
FITNESS FOR A PARTICULAR PURPOSE.

*/

#ifndef __CIFAR_READER_HXX__
#define __CIFAR_READER_HXX__

#include <fstream>
#include <string>
#include <algorithm>
#include "utils/Utils.hxx"
#include "utils/Vec23.hxx"
#include "utils/SimpleMatrix.hxx"

namespace CIFAR
{
    static const size_t ImW = 32, ImH = 32;
    static const size_t ImD = 3;
    static const size_t ImageSizeLin = ImW*ImH*ImD;
    static const size_t NumImagesPerOrigFile = 10000;
    static const size_t NumImages = 60000;

    static const char* ClassNames[10] ={   
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck" };
    static const char* CIFARBinImages[6] = {
        "data_batch_1.bin", "data_batch_2.bin", "data_batch_3.bin",
        "data_batch_4.bin", "data_batch_5.bin", "test_batch.bin" };
    struct CIFARReader
    {

    private:

        const std::string CIFARImageLocation = "cifar-10-batches-bin//", DataLocation;

    public:


        CIFARReader(std::string dataLocation = DATA_LOCATION) : DataLocation(dataLocation) {};

        // Returns a Volume(ImW,ImH,ImD*N), where ith frame is (i%ImD)th channel of (Floor(i/ImD))th image
        // Not a fourD image.
        inline std::pair<unsigned char***, unsigned char*> ImageDataCopy(size_t N = NumImages)
        {
            if (N > NumImages)
                throw std::invalid_argument("Offset and number of images cannot be accommodated");

            auto images = SimpleMatrix::ColocAlloc<char>({ImW, ImH, N * ImD});

            size_t numRead = 0;
            auto labels = new char[N];

            for (auto& binImgName : CIFARBinImages)
            {
                size_t imFromThisBin = std::min(NumImagesPerOrigFile, N - numRead);

                std::string filename = DataLocation + CIFARImageLocation + binImgName;
                std::ifstream inFile(filename, std::ios::binary);
                if (!inFile) throw std::invalid_argument("File read failed for " + filename);

                for (size_t binIm = 0; binIm < imFromThisBin; ++binIm, ++numRead)
                {
                    inFile.read(labels + numRead, 1);
                    inFile.read(images[numRead*ImD][0], ImageSizeLin); // +1 for the label
                }

                if (imFromThisBin < NumImagesPerOrigFile) break; // we read  all images except from last file/
            }

            return std::make_pair((unsigned char***)images, (unsigned char*)labels);
        }

        void TestCIFARReader(size_t N = 100)
        {
            auto imBuffer = new unsigned char[ImageSizeLin];
            auto imSet = ImageDataCopy();

            for (size_t i = 0; i < N; ++i)
            {
                size_t rand = i <=16 ? i :  Utils::URand(NumImages);

                PPMIO::Write(std::to_string(i) + "_" + std::to_string(rand) + "_" + ClassNames[imSet.second[rand]],
                    ImW, ImH, ImD, (unsigned char*)(imSet.first[rand*ImD][0]), false);
                
            }

            delete[] imBuffer;
            delete[] imSet.second;
            SimpleMatrix::deleteColocArray(imSet.first);
        }

    };
}
#endif
