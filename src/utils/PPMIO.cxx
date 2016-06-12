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

#include "Utils.hxx"
#include <fstream>
#include <sstream>


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

extern void PPMIO::Write(std::string name, size_t width, size_t height, size_t numComps, const unsigned char* data, bool interleaved)
{ 

    unsigned char* formattedData = nullptr;

   if(numComps == 1)
        establishEndsWith(name,".pgm");
    else if (numComps == 3)
    {
        establishEndsWith(name, ".ppm");
        if (!interleaved)
        {
            size_t numPixelsPerFrame = width*height;
            size_t numPixels = numPixelsPerFrame*numComps;

            formattedData = new unsigned char[numPixels];
            for (size_t i = 0, index = 0; i < numPixels; i += numComps, ++index)
                for (size_t c = 0; c < numComps; ++c)
                    formattedData[i + c] = data[index + numPixelsPerFrame*c];
        }
    }
    else 
        throw std::invalid_argument("Wrong Number of channels to write PPM: " + std::to_string(numComps));

    ofstream file(name, fstream::out|ios_base::binary);
    size_t numPixelsPerFrame = width*height;
    size_t numPixels = numPixelsPerFrame*numComps;

    file << (numComps == 3  ? "P6\n" : "P5\n");
    file << width << " " << height << "\n#Created by Ishwar\n";
    file << 255 << "\n";
    if(interleaved)
        file.write((char*)data,numPixels);        
    else
        file.write((char*)formattedData, numPixels);

    file.flush();
    
    if (formattedData)
        delete[] formattedData;
}

extern unsigned char* PPMIO::Read(string name, size_t& width, size_t& height, size_t& components)
{
    ifstream fileStream(name, fstream::out|ios_base::binary);
    if(!fileStream.is_open())
        throw std::invalid_argument("File could not be opened: " + name);
    char buf[1023];

    nextToken(fileStream, buf, buf);
    components = 0;
    if(strcmp(buf,"P6") == 0)
        components = 3;
    else if( strcmp(buf,"P4") == 0)
        components = 1;

    if(!components)
        throw std::invalid_argument("Unsupported PPM file or corrupt file: " + name);

    nextToken(fileStream, buf, width);
    nextToken(fileStream, buf, height);
    unsigned maxVal;
    nextToken(fileStream, buf, maxVal);
    
    if(maxVal>(2<<16))
        throw std::invalid_argument("PPM file is corrpt: " + name);

    char whitespace = fileStream.get();
    if (iswspace(whitespace))
        throw std::runtime_error("File " + name + " is notin correct PPM format");

    size_t pixelsPerComponent = height*width;
    size_t numBytes = pixelsPerComponent * components;

    char* data = new char[numBytes];
    std::fill_n(data, numBytes,(char)0);

    fileStream.read(data,numBytes);

    return(unsigned char*)(data);
}
