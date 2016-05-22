#ifndef __MNIST_IMAGE_READER__
#define __MNIST_IMAGE_READER__
#include <iostream>
#include <fstream>
#include <map>
#include <vector>
#include <string>

#define MNISTTHandWriting   "MNIST/"

class MNISTReader
{
public:

    static const unsigned ImageSizeLin = 28 * 28;
    static const unsigned ImW = 28;
    static const unsigned ImH = 28;
    static const unsigned NumImages = 60000;

    
    MNISTReader(const char* location);

    // Get idx'th image, returned alloc'ed memory  and freeing is handled;
    unsigned char* ImageData(unsigned idx, bool TrainImg = true /*T = Training , F = test*/);

    //Retrun N images starting from `offset`, with R[i] pointing to array sized ImageSizeLin with image values
    unsigned char** ImageData(unsigned N, unsigned offset, bool TrainImg = true/*Query Training Image not*/);
    
    // Same as above, but the returned pointer is not freed when object dies.
    unsigned char** ImageDataCopy(unsigned N, unsigned offset, bool TrainImg = true/*Query Training Image not*/);

    // Same as above, but the returned pointer is not freed when object dies.
    std::pair<unsigned char***, unsigned char*> ImageDataCopy2D(unsigned N, bool TrainImg = true/*Query Training Image not*/);

    // Get idx'th label;
    unsigned LabelData(unsigned idx);

    //Retrun N labels starting from offset `offset`.
    unsigned char* LabelData(unsigned N, unsigned offset, bool TrainLabel = true/*Query Training label or not*/);

    unsigned char* ImAlloc();
    unsigned char** ImAlloc(unsigned N);

    ~MNISTReader();

private:

    static const unsigned ImageFileOffset = 16;
    static const unsigned LabelFileOffset = 8;

    enum DataType
    {
        ImageTrain = 0,
        ImageTest,
        LableTrain,
        LableTest
    };

    std::vector<unsigned char*> Pointers;
    std::vector<unsigned char**> ArrayPointers;

    std::string Location;
    std::ifstream
        ImageTrainFstrm,
        ImageTestFstrm,
        LabelTrainFstrm,
        LabelTestFstrm;
};

#endif
