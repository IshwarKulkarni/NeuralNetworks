#include "MNIST-ImageReader.hxx"
#include "utils/Utils.hxx"
#include "utils/SimpleMatrix.hxx"


/*
IMAGE FILE (train-images-idx3-ubyte):

[offset] [type]          [value]          [description]
0000     32 bit integer  0x00000803(2051) magic number
0004     32 bit integer  60000            number of images
0008     32 bit integer  28               number of rows
0012     32 bit integer  28               number of columns
0016     unsigned byte   ??               pixel
0017     unsigned byte   ??               pixel
........
xxxx     unsigned byte   ??               pixel

Pixels are organized row-wise. Pixel values are 0 to 255. 0 means background (white), 255 means foreground (black).

LABEL FILE (t10k-labels-idx1-ubyte):

[offset] [type]          [value]          [description]
0000     32 bit integer  0x00000801(2049) magic number (MSB first)
0004     32 bit integer  10000            number of items
0008     unsigned byte   ??               label
0009     unsigned byte   ??               label
........
xxxx     unsigned byte   ??               label
The labels values are 0 to 9.
*/

static const std::string MNISTFileNames[4] = {
    "train-images.idx3-ubyte",
    "t10k-images.idx3-ubyte",
    "train-labels.idx1-ubyte",
    "t10k-labels.idx1-ubyte"

};

using namespace SimpleMatrix;

MNISTReader::MNISTReader(const char* location) : Location(location)
{
    ImageTrainFstrm.open((location + MNISTFileNames[(unsigned)MNISTReader::ImageTrain]).c_str(), std::ios::binary);
    ImageTestFstrm.open((location + MNISTFileNames[(unsigned)MNISTReader::ImageTest]).c_str(), std::ios::binary);
    LabelTrainFstrm.open((location + MNISTFileNames[(unsigned)MNISTReader::LableTrain]).c_str(), std::ios::binary);
    LabelTestFstrm.open((location + MNISTFileNames[(unsigned)MNISTReader::LableTest]).c_str(), std::ios::binary);

    bool allStreamsGood =
        ImageTrainFstrm.good() &
        ImageTestFstrm.good() &
        LabelTrainFstrm.good() &
        LabelTestFstrm.good();

    if(!allStreamsGood)
        throw std::invalid_argument("All MNIST files could not be opened from given location: " + std::string(location) );
}

unsigned char* MNISTReader::ImageData(unsigned idx, bool TrainImg)
{
    std::ifstream& in = TrainImg ? ImageTrainFstrm : ImageTestFstrm;
    if (!in.good())
        return 0;
    in.seekg(ImageFileOffset + idx * ImageSizeLin);
    auto ret = ImAlloc();
    in.read((char*)ret, ImageSizeLin);
    return ret;
}

unsigned char** MNISTReader::ImageData(unsigned N, unsigned offset, bool TrainImg)
{
    if (N + offset <= MNISTReader::NumImages)
    {
        std::ifstream& in = TrainImg ? ImageTrainFstrm : ImageTestFstrm;
        unsigned char** mem = ImAlloc(N);
        in.seekg(ImageFileOffset + offset* ImageSizeLin);
        in.read((char*)(mem[0]), ImageSizeLin * N);
        return mem;
    }
    return 0;
}

unsigned char** MNISTReader::ImageDataCopy(unsigned N, unsigned offset, bool TrainImg)
{
    if (N + offset <= MNISTReader::NumImages)
    {
        std::ifstream& in = TrainImg ? ImageTrainFstrm : ImageTestFstrm;
        SimpleMatrix::Matrix<unsigned char> mem({ImageSizeLin,N });
        in.seekg(ImageFileOffset + offset* ImageSizeLin);
        in.read((char*)(*mem), ImageSizeLin * N);
        return mem;
    }
    return 0;
}

std::pair<unsigned char***, unsigned char*> MNISTReader::ImageDataCopy2D(unsigned N, bool Train)
{
    if (N > NumImages)throw std::invalid_argument("Requested too many images: " + std::to_string(N));

    std::ifstream& imIn = Train ? ImageTrainFstrm : ImageTestFstrm;
    if (!imIn) throw std::invalid_argument("Failed to open MNIST image file.");
    
    ImageTrainFstrm.seekg(ImageFileOffset);
    char*** images = ColocAlloc<char>(Vec::Size3(ImW, ImH, N));
    ImageTrainFstrm.read(images[0][0], ImageSizeLin * N);
    if (imIn.gcount() != N * ImageSizeLin) std::runtime_error("Did not read enough image data");

    std::ifstream& lblIn = Train ? LabelTrainFstrm : LabelTestFstrm;
    lblIn.seekg(LabelFileOffset);
    char* lables = new char[N];
    lblIn.read((lables), N);

    if (lblIn.gcount() != N * ImageSizeLin) std::runtime_error("Did not read enough label data");
    return std::make_pair((unsigned char***)images, (unsigned char*)lables);
}

unsigned MNISTReader::LabelData(unsigned idx)
{
    std::ifstream& in = LabelTrainFstrm;
    in.seekg(LabelFileOffset + idx);
    unsigned char c;
    in >> c;
    return c;
}

unsigned char* MNISTReader::LabelData(unsigned N, unsigned offset,  bool TrainLabel)
{
    std::ifstream& in = TrainLabel ? LabelTrainFstrm : LabelTestFstrm;
    auto mem = new unsigned char[N];
    Pointers.push_back(mem);
    in.seekg(LabelFileOffset + offset);
    in.read((char*)(mem), N);
    return mem;
}

unsigned char* MNISTReader::ImAlloc()
{
    Pointers.push_back(new unsigned char[ImageSizeLin]);
    return Pointers.front();
}

unsigned char** MNISTReader::ImAlloc(unsigned N )
{
    SimpleMatrix::Matrix<unsigned char> mat({ ImageSizeLin,N });
    Pointers.push_back(mat.data[0]);
    ArrayPointers.push_back(mat.data);
    return mat;
}

MNISTReader::~MNISTReader()
{
    for (auto& p : Pointers)
        delete[] p;

    for (auto& ap : ArrayPointers)
        delete[] ap;

    ImageTrainFstrm.close();
    ImageTestFstrm.close();
    LabelTrainFstrm.close();
    LabelTestFstrm.close();
}
