#ifndef BINARYMATRIX_INCLUDED
#define BINARYMATRIX_INCLUDED

#include "Matrix.hxx"
#include "ImageGeneric.hxx"

class BinaryMatrix: public Matrix2D<bool>
{

public:
    
    template<typename Type>
    BinaryMatrix(const Matrix2D<Type>& mat): Matrix2D<bool>(mat.Width(), mat.Height())
    {
        FillData(mat.GetData(), (mat.GetMaxElem() + mat.GetMinElem())/2);
    }

    template<typename Type>
    BinaryMatrix(const Matrix2D<Type>& mat, Type thresh): Matrix2D<bool>(mat.Width(), mat.Height())
    {
        auto min = mat.GetMinElem();
        auto max = mat.GetMaxElem();
        thresh = min + (max - min)*thresh;
        FillData(mat.GetData(), thresh);
    }

    template<typename Type>
    BinaryMatrix operator=(const Matrix2D<Type>& mat)
    {
        delete[] m_Data;
        m_Width = mat.Width();
        m_Height = mat.Height();
        m_Data = new bool[NumElems()];

        FillData(mat.GetData(), (mat.GetMaxElem() + mat.GetMinElem())/2);

        return *this;
    }

    void WriteAsImage(const char* filename, ImageIO::ImageFormatType format = ImageIO::PPM )
    {
        auto n = NumElems();
        unsigned char*  data = new unsigned char[n];
        for (uint  i= 0 ; i < n; ++i)
            data[i] = m_Data[i]*255;
        try { PPMIO::Write(filename, m_Width, m_Height, 1, data); }
        catch(...) { delete[] data; }
    }

private:
    
    template<typename Type>
    void FillData(const Type* data, Type thresh) 
    {
        for (uint i = 0; i < NumElems(); ++i)
            m_Data[i] = data[i] > thresh;
    }
};



#endif