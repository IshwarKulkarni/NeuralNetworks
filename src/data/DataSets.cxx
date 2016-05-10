#include "MINST-ImageReader.hxx"
#include "DataSets.hxx"
#include "utils/Utils.hxx"
#include "utils/SimpleMatrix.hxx"
#include <fstream>

#define MNISTTHandWriting   "MNIST/"
#define UCIWineData         "uci-winedata//wine.csv"
#define UCIHandWriting2     "handwriting//letter-recognition-2.csv"
#define OpticalDigits       "UCI-Handwriting//optdigits.all.csv"


#define DATA_SPLIT_FILENAME "DataSplits.txt"

using namespace std;
using namespace Vec;
using namespace SimpleMatrix;

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

PatternSet<double***> LoadMnistData2(Vec::Size3& InputSize, unsigned& OutputSize, unsigned N)
{
    Logging::Timer timer("MNIST2 data load");
    
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

    double*** images2Df = SimpleMatrix::ColocAlloc<double>(Size3(MNISTReader::ImW, MNISTReader::ImH,N));

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
