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

#include "MNIST-ImageReader.hxx"
#include "DataSets.hxx"
#include "utils/Utils.hxx"
#include "utils/SimpleMatrix.hxx"
#include <fstream>

#define DATA_SPLIT_FILENAME "DataSplits.txt"

using namespace std;
using namespace Vec;
using namespace SimpleMatrix;
using namespace CIFAR;

double VldnFraction = 0.05f;
double TestFraction = 0.05f;

PatternSet<double*> LoadMnistData(unsigned& InputSize, unsigned& OutputSize)
{
    Logging::Timer timer("MNIST data load");
    
    InputSize = MNISTReader::ImageSizeLin;
    TargetPatternDef targetPattern(10);
    targetPattern.NumTargetClasses = 10;
    targetPattern.TargetType = TargetPatternDef::UseUnaryArray;
    PatternSet<double*> data(MNISTReader::NumImages, VldnFraction, TestFraction, targetPattern);
    OutputSize = targetPattern.TargetVectorSize;

    MNISTReader ImageReader(DATA_LOCATION MNISTTHandWriting);
    unsigned char
        **images = ImageReader.ImageDataCopy(MNISTReader::NumImages, 0, true),
        *labels = ImageReader.LabelData(MNISTReader::NumImages, 0, true);

    auto imagesf = ColocAlloc<double>(Vec::Size2(MNISTReader::ImageSizeLin, MNISTReader::NumImages));
    for (unsigned i = 0; i < MNISTReader::ImageSizeLin*MNISTReader::NumImages; ++i)
        imagesf[0][i] = double(images[0][i]) / 255.f;
    
    for (unsigned i = 0; i < MNISTReader::NumImages; ++i)
        data[i].Input = imagesf[i], 
        data[i].Target = data.GetTarget(labels[i]) ;
#ifndef _DEBUG
    data.ShuffleAll();
#endif

    data.SetDataToDelete(imagesf);
    deleteColocArray(images);

    return data;
}

PatternSet<unsigned char***> LoadMnistData2(Vec::Size3& InputSize, unsigned& OutputSize, Vec::Vec2<double> highlo, unsigned N)
{
    Logging::Timer timer("MNIST2 data load");
    
    InputSize = { MNISTReader::ImW, MNISTReader::ImH, 1 };
    TargetPatternDef targetPattern(10, TargetPatternDef::UseUnaryArray, highlo[0], highlo[1]);

    PatternSet<unsigned char***> data(N, VldnFraction, TestFraction, targetPattern);
    OutputSize = targetPattern.TargetVectorSize;

    MNISTReader ImageReader(DATA_LOCATION MNISTTHandWriting);
    auto imageLables = ImageReader.ImageDataCopy2D(N);

    for (unsigned i = 0; i < N; ++i)
        data[i].Input = imageLables.first + i,
        data[i].Target = data.GetTarget(imageLables.second[i]);

#ifndef _DEBUG
    data.ShuffleAll();
#endif
    delete[] imageLables.second;
    data.SetDataToDelete(imageLables.first);
    return std::move(data);
}

PatternSet<unsigned char***> LoadCifarData10(Vec::Size3& InputSize, unsigned& OutputSize, Vec::Vec2<double> highlo, unsigned N)
{
    Logging::Timer timer("CIFAR10 data load");

    InputSize = {CIFAR::ImW, CIFAR::ImH, CIFAR::ImD};

    TargetPatternDef targetPattern(10, TargetPatternDef::UseUnaryArray, highlo[0], highlo[1]);
    PatternSet<unsigned  char***> data(N, VldnFraction, TestFraction, targetPattern);
    OutputSize = targetPattern.TargetVectorSize;

    CIFARReader imageReader;
    //imageReader.TestCIFARReader();
    auto allImages = imageReader.ImageDataCopy();

    for (unsigned i = 0; i < N; ++i)
        data[i].Input = allImages.first + i*CIFAR::ImD,
        data[i].Target = data.GetTarget(allImages.second[i]);

#ifndef _DEBUG
    data.ShuffleAll();
#endif

    delete[] allImages.second;
    data.SetDataToDelete(allImages.first);
    return data;
}
