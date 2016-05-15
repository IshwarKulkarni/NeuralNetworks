#ifndef DATASETS_CXX_INCLUDED
#define DATASETS_CXX_INCLUDED

#include "MINST-ImageReader.hxx"
#include "DataSets.hxx"
#include "utils/utils.hxx"
#include "utils/SimpleMatrix.hpp"
#include "utils/commandline.hxx"
#include "imageprocessing/imagegeneric.hxx"
#include "linalgebra_ml/CIFAR-ImageReader.hxx"
#include <fstream>

#define MNISTTHandWriting   "MNIST\\"
#define UCIWineData         "uci-winedata\\wine.csv"
#define UCIHandWriting2     "handwriting\\letter-recognition-2.csv"
#define OpticalDigits        "UCI-Handwriting\\optdigits.all.csv"


#define DATA_SPLIT_FILENAME "DataSplits.txt"

using namespace std;
using namespace Vec;
using namespace CIFAR;
using namespace SimpleMatrix;

double VldnFraction = 0.05f;
double TestFraction = 0.05f;


void ReadDataSplitsFromFile()
{
    std::ifstream dataSplit(DATA_LOCATION DATA_SPLIT_FILENAME);

    if (dataSplit)
    {
        auto& splits = NameValuePairParser(dataSplit, " : ", '\0', "#").
            GetPairs<double>();

        unsigned read = 0;
        for (auto& val : splits)
        {
            if (StringUtils::Contains(val.first, "Test", false))
                TestFraction = val.second / 100, read++;
            else
                if (StringUtils::Contains(val.first, "Validation", false))
                    VldnFraction = val.second / 100, read++;
        }

        if (read == 2)
            Logging::Log << "\nData splits are read from file " DATA_LOCATION DATA_SPLIT_FILENAME " ; ";
    }
    else
        Logging::Log << "\nData splits explicitly set; ";
    
    Logging::Log  << "Fracstion: Test - " << TestFraction * 100 << "% and Validation - " << VldnFraction * 100 << "%\n";
}

PatternSet<double*> LoadMnistData(unsigned& InputSize, unsigned& OutputSize)
{
    Logging::Timer timer("MNIST data load");
    
    ReadDataSplitsFromFile();
    
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

PatternSet<double***> LoadMnistData2(Vec::Size3& InputSize, unsigned& OutputSize, unsigned N)
{
    Logging::Timer timer("MNIST2 data load");
    ReadDataSplitsFromFile();

    InputSize = { MNISTReader::ImW, MNISTReader::ImH, 1 };
    TargetPatternDef targetPattern(10);
    targetPattern.NumTargetClasses = 10;
    targetPattern.FillLow = 0;
    targetPattern.FillHigh = 1;
    targetPattern.TargetType = TargetPatternDef::UseUnaryArray;
    PatternSet<double***> data(N, VldnFraction, TestFraction, targetPattern);
    OutputSize = targetPattern.TargetVectorSize;

    MNISTReader ImageReader(DATA_LOCATION MNISTTHandWriting);
    unsigned char
        ***images2D = ImageReader.ImageDataCopy2D(N, 0, true),
        *labels = ImageReader.LabelData(N, 0, true);

    double*** images2Df = SimpleMatrix::ColocAlloc<double>(Size3(MNISTReader::ImW, MNISTReader::ImH,
                        N));

    for (unsigned i = 0; i < N*MNISTReader::ImageSizeLin; ++i)
        images2Df[0][0][i] = double(images2D[0][0][i]) / 255.f;

    for (unsigned i = 0; i < N; ++i)
        data[i].Input = &(images2Df[i]),
        data[i].Target = data.GetTarget(labels[i]);

    delete[] images2D[0][0];
    delete[] images2D[0];
    delete[] images2D;

    data.SetDataToDelete(images2Df);
    return data;
}
/*
PatternSet<int*> OpticalDigitsdata(unsigned& InputSize, unsigned& OutputSize)
{
    Logging::Timer timer("Optical digits data load");
    ReadDataSplitsFromFile();

    Matrix<int> mat =  ReadCSV<int>(ifstream(DATA_LOCATION OpticalDigits));
    OutputSize = 1;
    InputSize = mat.Width() - OutputSize;

    TargetPatternDef targetPattern(OutputSize);
    targetPattern.TargetType = TargetPatternDef::UseBinaryArray;
    targetPattern.NumTargetClasses = 10;
    targetPattern.FillLow = 0;
    targetPattern.FillHigh = 1;
    PatternSet<int*> data(mat.Height(), VldnFraction, TestFraction, targetPattern);
    RoundVertically(mat,1);

    OutputSize = targetPattern.TargetVectorSize;
    
    for (unsigned i = 0; i < mat.Height(); ++i)
        data[i].Input = mat[i],
        data[i].Target = data.GetTarget(mat[i][InputSize]);

    data.SetDataToDelete(mat);
    data.ShuffleAll();
    return data;
}

PatternSet<int*> UCIHandwriting2(unsigned& InputSize, unsigned& OutputSize)
{
    Logging::Timer timer("Handwriting data load");
    ReadDataSplitsFromFile();

    Matrix<int> mat = ReadCSV<int>(ifstream(DATA_LOCATION UCIHandWriting2));
    OutputSize = 3;
    InputSize = mat.Width() - OutputSize;

    TargetPatternDef targetPattern(OutputSize);
    targetPattern.TargetType = TargetPatternDef::UseNone;
    targetPattern.NumTargetClasses = 3;
    PatternSet<int*> data(mat.Height(), VldnFraction, TestFraction, targetPattern);

    for (unsigned i = 0; i < mat.Height(); ++i)
        data[i].Input = mat[i],
        data[i].Target = mat[i] + InputSize;

    data.SetDataToDelete(mat);
    data.ShuffleAll();
    return data;
}

PatternSet<double *> UCIWine(unsigned& InputSize, unsigned& OutputSize)
{
    Logging::Timer timer("UCIWine data load");
    ReadDataSplitsFromFile();

    Matrix<double> mat = ReadCSV<double>(ifstream(DATA_LOCATION UCIWineData));
    InputSize = mat.Width() - 1;

    TargetPatternDef targetPattern(3);
    targetPattern.TargetType = TargetPatternDef::UseBinaryArray;
    PatternSet<double*> data(mat.Height(), VldnFraction, TestFraction, targetPattern);
    OutputSize = targetPattern.TargetVectorSize;

    for (unsigned i = 0; i < mat.Height(); ++i)
        data[i].Input = mat[i] + 1,
        data[i].Target = data.GetTarget(unsigned(mat[i][0]));

    RoundVertically(mat, 1);

    data.SetDataToDelete(mat);
    data.ShuffleAll();
    return data;
}
*/
PatternSet<Color*> LoadCIFAR10RGB(unsigned& InputSize, unsigned& OutputSize);

 PatternSet<double*> LoadCIFAR10MF(unsigned& InputSize, unsigned& OutputSize)
 {
     Logging::Timer timer("CIFAR data load");
     ReadDataSplitsFromFile();

     TargetPatternDef targetPattern(10);
     targetPattern.FillLow = 0;
     targetPattern.FillHigh = 1;
     targetPattern.TargetType = TargetPatternDef::UseBinaryArray;
     PatternSet<double*> data(CIFAR::ImageReader::NumImages, VldnFraction, TestFraction, targetPattern);

     CIFAR::ImageReader imageReader;
     uchar *labels = imageReader.LabelData();
     double**    imagesF = imageReader.ImageDataCopy<double>();

     InputSize = imageReader.ImH * imageReader.ImW;
     OutputSize = targetPattern.TargetVectorSize;

     for (unsigned i = 0; i < CIFAR::ImageReader::NumImages; ++i)
         data[i] = { imagesF[i], data.GetTarget(labels[i]) };

     data.ShuffleAll();
     data.SetDataToDelete(imagesF);
     return data;
 }

 PatternSet<double***> LoadCIFAR10(Size3& InputSize, Size3& OutputSize, unsigned N)
 {
     Logging::Timer timer("CIFAR data load");
     ReadDataSplitsFromFile();

     TargetPatternDef targetPattern(10);
     targetPattern.FillLow = 0;
     targetPattern.FillHigh = 1;
     targetPattern.TargetType = TargetPatternDef::UseBinaryArray;
     PatternSet<double***> data(N, VldnFraction, TestFraction, targetPattern);

     CIFAR::ImageReader imageReader;
     uchar *labels = imageReader.LabelData(N);
     double***    imagesF = imageReader.ImageDataRaw(N);

     InputSize = { imageReader.ImW, imageReader.ImH, 3 };
     OutputSize = { targetPattern.TargetVectorSize,1, 1 };

     for (unsigned i = 0; i < N; ++i)
         data[i] = { &(imagesF[i*3]), data.GetTarget(labels[i]) };

     data.ShuffleAll();
     data.SetDataToDelete(imagesF);
     return data;
 }

#endif
