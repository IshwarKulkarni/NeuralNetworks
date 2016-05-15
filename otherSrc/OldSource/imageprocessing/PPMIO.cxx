#include "ImageGeneric.hxx"
#include "utils/StringUtils.hxx"
#include <fstream>
#include <sstream>


using namespace PPMIO;
using namespace std;
using namespace StringUtils;

template<typename Type>
void nextToken(ifstream& file, char * buf, Type& token )
{
    while(!file.eof())
    {
        file >> buf;
        if(buf[0]=='#')
            file.getline(buf, 1024);
        else
            break;
    }

    std::stringstream ss;
    ss << buf;
    ss >> token;
}

extern void PPMIO::Write(string strIn, uint width, uint height, uint numComps, const byte* frames)
{ // expects interleaved data;

    if( numComps == 3) 
        establishEndsWith(strIn,".ppm");
    else  if(numComps == 1)
        establishEndsWith(strIn,".pgm");
    else 
        throw new WrongNumberOfChannels(numComps);

    ofstream file(strIn, fstream::out|ios_base::binary);
    uint numPixelsPerFrame = width*height;
    uint numPixels = numPixelsPerFrame*numComps;

    file << (numComps == 3  ? "P6\n" : "P5\n");
    file << width << " " << height << "\n#Created by Ishwar\n";
    file << 255 << "\n";
    file.write((char*)frames,numPixels);        

    file.flush();
}

extern byte* PPMIO::Read(string strIn, uint& width, uint& height, uint& components)
{
    ifstream fileStream(strIn, fstream::out|ios_base::binary);
    THROW_IF(!fileStream.is_open(), UnopenFileStream, "PPM file not open");
    char buf[1023];

    nextToken(fileStream, buf, buf);
    components = 0;
    if(strcmp(buf,"P6") == 0)
        components = 3;
    else if( strcmp(buf,"P4") == 0)
        components = 1;

    THROW_IF(!components, UnsupportedFileFormat, "Unsupported file or corrupt file");

    nextToken(fileStream, buf, width);
    nextToken(fileStream, buf, height);
    unsigned maxVal;
    nextToken(fileStream, buf, maxVal);
    THROW_IF(maxVal>(2<<16), CorruptFile , "Invalid mx val from the PPM file");

    char whitespace = fileStream.get();

    uint pixelsPerComponent = height*width;
    uint numBytes = pixelsPerComponent * components;

    char* data = new char[numBytes];
    std::fill_n(data, numBytes,(char)0);

    fileStream.read(data,numBytes);

    return(byte*)(data);
}
