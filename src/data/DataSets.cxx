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

Vec2<double> SetVldnTestFractions(double vldn, double test)
{
    if (vldn + test >= 1.0)
        throw std::runtime_error("");

    Vec2<double> prev = { VldnFraction, TestFraction };
    VldnFraction = vldn, TestFraction = test;
    return prev;
}

PatternSet<unsigned char***> LoadMnistData2(Vec::Size3& InputSize, unsigned& OutputSize, Vec::Vec2<double> highlo, unsigned N)
{
    Logging::Timer timer("MNIST2 data load");

    if (N > MNISTReader::NumImages + MNISTReader::NumTestImages)
        N = MNISTReader::NumImages + MNISTReader::NumTestImages;

    InputSize = { MNISTReader::ImW, MNISTReader::ImH, 1 };
    TargetPatternDef targetPattern(10, TargetPatternDef::UseUnaryArray, highlo[0], highlo[1]);

    PatternSet<unsigned char***> data(N, VldnFraction, TestFraction, targetPattern);

    OutputSize = targetPattern.TargetVectorSize;

    size_t trainImages = N > MNISTReader::NumImages ? MNISTReader::NumImages : N,
        testImages = N > MNISTReader::NumImages ? N - MNISTReader::NumImages : 0;

    MNISTReader ImageReader(DATA_LOCATION MNISTTHandWriting);
    auto imageLables = ImageReader.ImageDataCopy2D(trainImages);

    auto testImageLables = testImages > 0 ? ImageReader.ImageDataCopy2D(testImages, false)
        : make_pair(nullptr, nullptr);
    for (unsigned i = 0; i < trainImages; ++i)
        data[i].Input = imageLables.first + i,
        data[i].Target = data.GetTarget(imageLables.second[i]);

    for (unsigned i = MNISTReader::NumImages; i < N; ++i)
        data[i].Input = testImageLables.first + i - MNISTReader::NumImages,
        data[i].Target = data.GetTarget(testImageLables.second[i - MNISTReader::NumImages]);

    if (testImageLables.second) 
        delete[] testImageLables.second;
    
    delete[] imageLables.second;

    data.SetDataToDelete(imageLables.first);
    data.SetDataToDelete(testImageLables.first);
#ifndef _DEBUG
    data.ShuffleTrnVldn();
#endif

    return data;
}

PatternSet<unsigned char***> LoadCifarData10(Vec::Size3& InputSize, unsigned& OutputSize, Vec::Vec2<double> highlo, unsigned N)
{
    Logging::Timer timer("CIFAR10 data load");

    N = std::min(N, CIFAR::NumImages); 

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
