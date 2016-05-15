#include "linalgebra_ml/CIFAR-ImageReader.hxx"
#include "utils/SimpleMatrix.hpp"
#include "linalgebra_ml/DataSets.hxx"
#include "utils/utils.hxx"

using namespace std;
using namespace Vec;
using namespace Utils;
using namespace CIFAR;
using namespace SimpleMatrix;


void CIFARLoader()
{
    CIFAR::ImageReader imageReader;
    Matrix<float> image({ 32,32 });
    for (unsigned i = 1; i < 5; ++i)
    {
        for (int k = -5; k < 5; k++)
        {
            int n = i * 32 + k;
            Matrix<uint> imageI({ 32,32 });

            imageReader.ImageDataCopy<float>(1, n, &(image[0]));

            struct Converter { uchar operator()(float f) { return uchar(f * 255); } };

            Matrix<uchar> imageC(image, Converter());
            Matrix<Color> imagergb(imageI);

            imageC.WriteAsImage("images" + to_string(n));
            imagergb.WriteAsImage("images" + to_string(n));
        }
    }
    image.Clear();
}


void MNISTReader2Test()
{
    Size3 ipSize, outSize;
    auto& data = LoadMnistData2(ipSize, outSize);
    for (unsigned epoc = 0; epoc < 20; ++epoc)
    {
        unsigned idx = Rand(data.GetDataSize());
        unsigned char buf[28 * 28];

        unsigned i = 0;
        for2d(Size2(28, 28))
            buf[i++] = unsigned char(data[idx].Input[0][y][x] * 255);

        PPMIO::Write(std::to_string(epoc) + to_string(BinToNum(data[idx].Target, outSize.x)), 28, 28, 1, buf);
    }
}

int main()
{
    CIFARLoader();
    MNISTReader2Test();
    
    return 0;
}

