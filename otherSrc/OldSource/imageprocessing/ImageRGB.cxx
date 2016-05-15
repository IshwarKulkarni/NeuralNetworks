#include "ImageRGB.hxx"
#include "ImageGeneric.hxx"

ImageRGB::ImageRGB(const ImageRGB& img)
{
    delete[] m_Data;
    m_Height = img.Width();
    m_Width = img.Height();
    m_Data = new Color[m_Height*m_Width];
    std::copy(img.GetData(), img.GetData()+img.NumElems(), m_Data);
}

ImageRGB::ImageRGB(const ImageGeneric<>& img)
{
    THROW_IF(img.GetNumChannels() != 3, WrongNumberOfChannels, "Only Images with three channels can be assigned to ImageRGB type images\n");

    m_Width = img.GetImageSize().first, m_Height= img.GetImageSize().second;

    ImageGeneric<> inImg = img;
    inImg.Scale(0,255);

    const auto& RChannel = img(0);        
    const auto& GChannel = img(1);
    const auto& BChannel = img(2);

    for (uint i = 0; i < img.GetNumPixels(); i++)
    {
        m_Data[i].R = (uchar)(RChannel)(i);
        m_Data[i].G = (uchar)(GChannel)(i);
        m_Data[i].B = (uchar)(BChannel)(i);
    }
}

ImageRGB::ImageRGB(const char* file) 
{
    uint comps;
    SharedBytes bytes = ImageIO::ReadImage(file, true, m_Width,m_Height, comps);
    THROW_IF(comps != 3, WrongNumberOfChannels, "ImageRGB has wrong number of channels\n");
    THROW_IF(!bytes, WrongNumberOfChannels, ".. ");
    m_Data = new Color[NumElems()];
    for (uint i = 0; i < NumElems(); i++)
        m_Data[i]  = *((Color*)(bytes.get()+ i*3));
}

uint* ImageRGB::ToIntArray()
{
    uint* data = new uint[NumElems()];
    for (uint i = 0; i < NumElems(); i++)
        data[i] = m_Data[i];

    return data;
}
