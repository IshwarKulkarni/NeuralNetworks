
#ifndef MATRIX_INCLUDED
#define MATRIX_INCLUDED


#include "utils/Exceptions.hxx"
#include "utils/Logging.hxx"
#include "utils/Utils.hxx"
#include "utils/StringUtils.hxx"
#include "matrix/MatrixIter.hxx"

#include <numeric>  
#include <algorithm>
#include <type_traits>
#include <iomanip> 
#include <iostream>
#include <ostream>
#include <vector>

template<typename T, typename U> struct BinAdd { inline T operator()(const T& t1, const U& t2)const{ return t1 + t2; } };
template<typename T, typename U> struct BinSub { inline T operator()(const T& t1, const U& t2)const{ return t1 - t2; } };
template<typename T, typename U> struct BinMul { inline T operator()(const T& t1, const U& t2)const{ return t1 * t2; } };
template<typename T, typename U> struct BinDiv { inline T operator()(const T& t1, const U& t2)const{ return t1 / t2; } };
template<typename T, typename U> struct BinPow { inline T operator()(const T& t1, const U& t2)const{ return std::pow(t,S); } };

template<typename T, typename U> struct BinAddS { U S; BinAddS(const U& Val):S(Val){} inline T operator()(uint i, const T& t)const{ return t + S; } };
template<typename T, typename U> struct BinSubS { U S; BinSubS(const U& Val):S(Val){} inline T operator()(uint i, const T& t)const{ return t - S; } };
template<typename T, typename U> struct BinMulS { U S; BinMulS(const U& Val):S(Val){} inline T operator()(uint i, const T& t)const{ return t * S; } };
template<typename T, typename U> struct BinDivS { U S; BinDivS(const U& Val):S(Val){} inline T operator()(uint i, const T& t)const{ return t / S; } };
template<typename T, typename U> struct BinPowS { U S; BinPowS(const U& Val):S(Val){} inline T operator()(uint i, const T& t)const{ return (T)(std::pow(t,S)); } };

typedef float ImageType;
template <typename T>
class  Matrix2D
{      

public:    
    typedef T ElemType;
    
    Matrix2D<T>(std::string file, char colDelim = ',', char rowDelim = '\n')
    {
        if( StringUtils::endsWith(file, ".csv") )
            ReadFromCSV(file, colDelim, rowDelim);
        else if (StringUtils::endsWith(file, ".raw"))
            ReadFromRAW(file, colDelim, rowDelim);
        else
            THROW(NativeFormatException, "Not sure how to read this file");
    }

    Matrix2D<T>(uint w, uint h)
    {
        InitInternal(w,h);
    }

    template <typename TypeCastableToT>
    Matrix2D<T>(const uint w, const uint h, const  TypeCastableToT* data)
    {
        InitInternal(w,h);
        if( data )
            for (uint i = 0; i < w*h; i++) 
                m_Data[i] = data[i];
    }

    Matrix2D<T>(const uint w, const uint h, T* data)
    {
        InitInternal(w,h);
        m_Data = data;
    }
    
    Matrix2D<T>(const uint w, const uint h, const T& val )
    {
        InitInternal(w,h);
        std::fill(m_Data, m_Data+NumElems(),val);
    }

    Matrix2D<T>(const Matrix2D<T>& other)
    {
        InitInternal(other.Width(), other.Height());
        const T* otherData = other.GetData();
        std::copy(otherData, otherData+other.NumElems(), m_Data);
    }

    template <class OtherType>
    Matrix2D<T>( const Matrix2D<OtherType>& other )
    {
        delete[] m_Data;
        InitInternal(other.Width(), other.Height());
        std::transform(other.GetData(), other.GetData()+other.NumElems(), m_Data, [](const OtherType& o) { return (T)(o);});
    }

    template <class OtherType>
    Matrix2D<T>( MatrixIter<Matrix2D<OtherType> >* iter, uint w = 0, uint h = 0)
    {
        InitInternal(max(iter->Width(),w),max(iter->Height(),h));
        for (uint i = 0; i < NumElems() ; i++)
        {
            m_Data[i] = *(*iter);
            ++(*iter);
        }
    }

    virtual ~Matrix2D()
    {
        m_Height = m_Width = 0;
        if (m_Data && DeletData)
        {
            delete[] m_Data;
            m_Data = nullptr;
        }
    }
    
    Matrix2D<T>& operator=( const Matrix2D<T>& other)
    {
        const T* otherData = other.GetData();
        m_Height = other.Height();
        m_Width = other.Width();
        
        delete[] m_Data;
        m_Data = new ImageType[m_Height*m_Width];
        std::copy(otherData, otherData+other.Height()*other.Width(), this->m_Data);
        return *this;
    }
    
    template <class OtherType>
    Matrix2D<T>& operator=( const Matrix2D<OtherType>& other )
    {
        const OtherType* otherData = other.GetData();
        m_Height = other.Height();
        m_Width = other.Width();
        THROW_IF( (other.Height() <= 0  || other.Width() <= 0 || otherData == nullptr), 
            UninitilizedException, "This matrix is not initialized\n");

        delete[] m_Data;
        m_Data = new T[m_Height*m_Width];
        std::copy(otherData, otherData+other.Height()*other.Width(), this->m_Data);
        return *this;
    }
    
    virtual inline T* RetainData() { DeletData = false; return m_Data; }
    ///////////////////////////////////////////////////
    template <typename OtherType, typename BinaryFun>
    Matrix2D<T>& Unary1Mat( const Matrix2D<OtherType>& other, const BinaryFun& binFun)
    {
        const OtherType* otherData = other.GetData();
        THROW_IF( (other.Height() != m_Height  || other.Width() != m_Width), WrongSizeException, "Wrong size in +=\n");

        for (uint i = 0;  i < m_Height*m_Width;  i++)
            m_Data[i] = binFun(m_Data[i] , other.GetData()[i]);

        return *this;
    }

    template<typename Fun>
    void UnaryFun(Fun& fun)
    {
        uint k=(uint)(-1);
        uint i= k ; 
        for (uint y = 0;  y < m_Height;  ++y) // TODO OMP outer loop;
        for (uint x = 0;  x < m_Width;   ++x) 
            m_Data[++i] = fun(x,y,m_Data[++k]);
    }

    template<typename Fun>  // 1 argument version of above
    void UnaryFun1(Fun& fun)
    {
        for (uint i = 0;  i < m_Height*m_Width;  i++)
            m_Data[i] = fun(i,m_Data[i]);
    }
    
    template<typename U> const Matrix2D<T>& operator+=(const U val) { UnaryFun1(BinAddS<T,U>(val)); return *this; }
    template<typename U> const Matrix2D<T>& operator-=(const U val) { UnaryFun1(BinSubS<T,U>(val)); return *this; }
    template<typename U> const Matrix2D<T>& operator*=(const U val) { UnaryFun1(BinMulS<T,U>(val)); return *this; }
    template<typename U> const Matrix2D<T>& operator/=(const U val) { UnaryFun1(BinDivS<T,U>(val)); return *this; }
    template<typename U> const Matrix2D<T>& operator^=(const U val) { UnaryFun1(BinPowS<T,U>(val)); return *this; }

    template<typename U> const Matrix2D<T>& operator+=(const Matrix2D<U>& mat) { Unary1Mat(mat,BinAdd<T,U>()); return *this; }
    template<typename U> const Matrix2D<T>& operator-=(const Matrix2D<U>& mat) { Unary1Mat(mat,BinSub<T,U>()); return *this; }
    template<typename U> const Matrix2D<T>& operator*=(const Matrix2D<U>& mat) { Unary1Mat(mat,BinMul<T,U>()); return *this; }
    template<typename U> const Matrix2D<T>& operator/=(const Matrix2D<U>& mat) { Unary1Mat(mat,BinDiv<T,U>()); return *this; }
    template<typename U> const Matrix2D<T>& operator^=(const Matrix2D<U>& mat) { Unary1Mat(mat,BinPow<T,U>()); return *this; }

    virtual inline const T* operator[](uint rowNum) const
    {
#ifdef BOUND_CHECKING
        THROW_IF(rowNum >= Height(), WrongSizeException, "Matrix has %d rows, queried for row %d", Width(), rowNum);
#endif
        return (const T*)(m_Data + rowNum * Width());
    }

    virtual inline T* operator[](uint rowNum) 
    {
#ifdef BOUND_CHECKING
        THROW_IF(rowNum >= Height(), WrongSizeException, "Matrix has %d rows, queried for row %d", Width(), rowNum);
#endif
        return m_Data + rowNum * Width() ;
    }

    //virtual inline MatrixRowIter<Matrix2D<T> > operator[](uint rowNum)
    //{
    //    THROW_IF(rowNum >= Height(), WrongSizeException, "Matrix has %d rows, queried for row %d", Height() - 1, rowNum);
    //    return MatrixRowIter<Matrix2D<T> >(*this, rowNum);
    //}

    virtual inline MatrixRowIter<Matrix2D<T> > begin()
    {
        return MatrixRowIter<Matrix2D<T> >(*this, 0);
    }
    
    virtual inline MatrixRowIter<Matrix2D<T> > end()
    {
        return MatrixRowIter<Matrix2D<T> >(*this, Height());
    }

    virtual inline T& operator()(int x, int y) const// get the element by repeating border values for out of bounds indices
    {
        x = std::min<uint>(std::max<uint>(x, 0), m_Width  - 1);
        y = std::min<uint>(std::max<uint>(y, 0), m_Height - 1);
        return m_Data[x + y*m_Width];
    }

    virtual inline T& operator()(int x, int y) // get the element by repeating border values for out of bounds indices
    {
        x = std::min<uint>(std::max<uint>(x, 0), m_Width - 1);
        y = std::min<uint>(std::max<uint>(y, 0), m_Height - 1);
        return m_Data[x + y*m_Width];
    }

    virtual inline const T& operator()(uint p) const
    {
        THROW_IF( p >= m_Width*m_Height , WrongSizeException, "Array Index too large");
        return m_Data[p];
    }

    T* CopyData() const 
    {
        T* copyOut = new T[NumElems()];
        std::copy(m_Data  , m_Data + NumElems(),copyOut);
        return copyOut;
    }

    // Returned array, Ret, is a colocated array, delete only only Ret[0] (then Ret)
    T** CopyData2D(uint startX = 0 , uint startY = 0, uint endX = -1, uint endY = -1) const
    {
        Utils::Clamp(startX, m_Width , (uint)0);
        Utils::Clamp(endX,     m_Width , (uint)0);
        Utils::Clamp(startY, m_Height, (uint)0);
        Utils::Clamp(endY,     m_Height, (uint)0);

        THROW_IF(startX >= endX || startY >= endY, WrongSizeException, 
            "Degenerate arguments: [(%d, %d),(%d, %d)] ", startX, startY, endX, endY);
        size_t w = endX - startX, h = endY - startY;
        
        T** copy2d = new T*[h];
        
        copy2d[0] = new T[h*w];
        for (uint i = 0; i < h; ++i)
#if defined(_MSC_VER)
#pragma warning(disable:4996)
#endif
            std::copy( m_Data + i*m_Width + startX, 
                       m_Data + i*m_Width + endX, 
                       copy2d[i] = (copy2d[0] + i*w ) );

        return copy2d;
    }

    template<typename OutType = T>
    OutType* CopyColumn(uint c)
    {
        THROW_IF(c >= m_Width, DimensionException, "Invalid column index\n");

        OutType* column = new OutType[m_Height];
        for (uint i = 0; i < m_Height; ++i)
            column[i] = OutType(m_Data[i*m_Width + c]);

        return column;
    }
    
    virtual inline T& operator()(uint p)
    {
        THROW_IF( p >= m_Width*m_Height , WrongSizeException, "Array Index too large");
        return m_Data[p];
    }

    T Get(uint x, uint y) const // same as operator()(int,int) but no bound checking
    {
        return m_Data[x + m_Width*y];
    }

    ////////////////////////////////////////////////////
    
    template<typename FunctionType >
    inline uint ForEachElement(FunctionType func) // return the first element where func returned false
    {
        int index = -1 ; 
        auto a = m_Data;
        while (++index < (int)(m_Height*m_Width) && func(a++));
        return index;
    }

    virtual void SumElems(T& sum) const{
        std::accumulate(m_Data,m_Data+NumElems(),sum);
    }

    virtual inline T GetMaxElem() const 
    {
        T t = *std::max_element(GetData(), GetData()+m_Height*m_Width);
        return t;
    }

    virtual inline T GetMinElem()  const
    {
        T t = *std::min_element(GetData(), GetData()+m_Height*m_Width);
        return t;
    }
    
    Matrix2D<T>& Clamp(T& low, T& high)
    {
        T lowElem = GetMinElem(), hiElem = GetMaxElem();
        T range = (hiElem - lowElem);
        high = lowElem + range*high;
        low  = lowElem + range*low;

        ForEachElement([&](T* e)
        {
            if (*e < low) 
                *e = lowElem; 
            if( *e > high) 
                *e = hiElem; 
            return true;
        });

        return *this;
    }

    //////////////////////////////////////////////////// 

    virtual void Convolve(const Matrix2D<float>* kernel)
    {
        auto rawKernel = kernel->GetData();
        
        uint kh = kernel->Height();
        uint kw = kernel->Width();

        T* data = new T[m_Height*m_Width];
        
        for ( uint x = 0; x < m_Width; ++x )
        for ( uint y = 0; y < m_Height; ++y ) 
        {
            T sum = T(0);
            for ( uint kx = 0; kx < kw; ++kx )
            for ( uint ky = 0; ky < kh; ++ky )
            {
                int imX = x + kx - kw/2;
                int imY = y + ky - kh/2;
#pragma warning(disable: 4800 4804)
                sum += (T)( (*this)(imX, imY)* (*kernel)(kx,ky) );
#pragma warning(default: 4800 4804)
            }
            data[x + m_Width*y] = sum;
        }
        
        delete[] m_Data;
        m_Data = data;
    }
    
    typedef  MatrixIter<Matrix2D<T> > RowIterator;

    inline void RandPerm()
    {
        srand((uint)time(0));
        for (uint i = 0; i < Height()/2; ++i)
        {
            uint fromRow = Rand(Height());
            uint toRow   = Rand(Height());
            for(uint j = 0; j < Width(); ++j)
                std::swap(m_Data[j + m_Width*toRow],m_Data[j + m_Width*fromRow]);
        }
    }

    virtual void Identify(std::ostream& O) const
    {
        O    << "Matrix2D: Id=" << GetId() << std::hex 
            << " this=0x" << (unsigned long long)this << std::dec 
            <<  ", Dimension=(" << m_Width << "x" << m_Height << ")" 
            << ", m_Data=0x" << std::hex << (unsigned long long)m_Data << std::dec << std::endl;
            
    }

   //virtual void Write(std::string name, char colDelim = ',', char rowDelim = '\n')
   //{
   //    if(!StringUtils::endsWith(name, ".csv"))
   //        name += ".csv";
   //
   //    std::ofstream strm(name.c_str());
   //
   //    for ( uint y = 0; y < m_Height; ++y ) 
   //    {
   //        for (uint x = 0; x < m_Width; ++x)
   //            strm << ( (T)( operator()(x, y) ) ) << ( ((x != m_Width - 1) ? colDelim : rowDelim) );
   //    }
   //
   //    strm.flush();
   //}
    
    void WriteToRAW(std::string file)
    {
        StringUtils::establishEndsWith(file, ".raw");
        std::ofstream strm(file.c_str(), ios::binary);

        Utils::WriteAsBytes(strm, m_Width);
        Utils::WriteAsBytes(strm, m_Height);

        uchar one = 0;

#ifdef WIN32
        one = 1; // crossing OS's usually has endianess problems.
#endif

        Utils::WriteAsBytes(strm, one);
        Utils::WriteAsBytes(strm, uint(sizeof(T)));
        char* raw = reinterpret_cast<char*>(m_Data);
        strm.write(raw, m_Width*m_Height*sizeof(T));
    }
    ////////////////////////////////////////////////////

    virtual inline uint size()  const { return m_Height;}
    virtual inline uint Width()  const { return m_Width;}
    virtual inline uint Height() const { return m_Height;}
    virtual inline uint NumElems() const {return m_Width*m_Height;}
    virtual inline T* GetData() const { return m_Data;}
    virtual inline uint GetId() const { return Id;}
    
protected:

    void ReadFromRAW(std::string file, char colDelim, char rowDelim)
    {
        Logging::Log << __FUNCTION__ << ", Reading Matrix data from " << file << " ... ";
        Logging::Timer timer("ReadRAW", false);
        StringUtils::establishEndsWith(file, ".raw");
        std::ifstream strm(file.c_str(), ios::binary);
        ReadAsBytes(strm, m_Width);
        ReadAsBytes(strm, m_Height);

        uchar writtenInWindows = 0; // windows writes 1;
        ReadAsBytes(strm, writtenInWindows);
        THROW_IF(!writtenInWindows, NativeFormatException, "File Not Written in Windows, cannot read");

        uint elemSize = 0;
        ReadAsBytes(strm, elemSize);
        THROW_IF(sizeof(T) != elemSize, BadAssumptionException, "File was not written to read this format");

        InitInternal(m_Width, m_Height);
        strm.read(reinterpret_cast<char*>(m_Data), m_Width*m_Height*sizeof(T));

        Logging::Log << " Done in " << timer.Stop() << "s.\n";
    }

    void ReadFromCSV(std::string file, char colDelim, char rowDelim)
    {
        StringUtils::establishEndsWith(file, ".csv");
        Logging::Log << __FUNCTION__ << ", Reading Matrix data from " << file << " ... ";

        Logging::Timer timer("ReadCSV", false);


        std::ifstream strm(file.c_str());

        THROW_IF(!strm.good(), FileIOException, "The CSV file could not be opened, wrong path, perhaps?");

        std::string line;
        std::getline(strm, line, rowDelim);
        auto width = std::count(line.begin(), line.end(), colDelim) + 1;

        THROW_IF(width == 0, UnsupportedFileFormat, "File is not organized correctly, number of columns appear to be zero");

        strm.seekg(strm.beg);

        std::vector<T> buf;
        uint height = 0;
        while (strm.good())
        {
            if (strm.eof() || rowDelim == strm.peek())
                break;

            buf.resize(buf.size() + width); //  grow one row at a time, this is slow, but faster than file read, so forget it.
            char c = '\0';

            for (int j = 0; j < width; ++j)
            {
                strm >> buf[(height)*width + j];
                c = strm.get();
                if (strm.eof())
                    break;
                THROW_IF(c != colDelim && j != width - 1, UnsupportedFileFormat, "Delimiter is invalid");
            }
            ++height;
            THROW_IF(c != rowDelim && !strm.eof(), UnsupportedFileFormat, "Delimiter is invalid");
        }
        THROW_IF(width*height == 0, DimensionException, "One of the Width or height for matrix is zero");
        InitInternal((uint)(width), height);
        for (uint i = 0; i < width*height; i++) m_Data[i] = buf[i];

        Logging::Log << " Done in " << timer.Stop() << "s.\n";
    }

    uint GetCounterNext()
    { 
        static uint c = 0;
        return ++c;
    }

    uint m_Height, m_Width;
    uint Id;
    T* m_Data;
    bool DeletData;
    
    Matrix2D<T>(): Id(GetCounterNext()), m_Height(0), m_Width(0), m_Data(nullptr){};

    virtual void InitInternal( uint w, uint h)
    {
        m_Height = h;
        m_Width = w;
        Id  = GetCounterNext();
        DeletData = false;
        m_Data = new T[m_Height*m_Width];
    }

};

template<typename T>
std::ostream& operator<<(std::ostream& strm, const Matrix2D<T>& M)
{
    bool isFloating = std::is_floating_point<T>::value;
    bool isIntegral = std::is_integral<T>::value;

    M.Identify(strm);
    auto data = M.GetData();
    auto w = M.Width();
    strm << LogEndl;
    for ( uint x = 0; x < M.Width(); ++x )
        strm << std::setprecision(4) << x << ":\t";
    strm << "\n"; 
    for ( uint y = 0; y < M.Height(); ++y ) 
    {
        strm << y << "| " ;
        for ( uint x = 0; x < M.Width(); ++x )
            if(isFloating)
                strm << (float) data[x + y*w] << " " ;
            else if (isIntegral)
                strm << (int) data[x + y*w] << " " ;
            else 
                strm << data[x + y*w] << " " ;
        strm << "\n";
    }

    return strm;
}
/*
template<typename TypeOne, typename TypeTwo>
inline auto operator+(const Matrix2D<TypeOne>& one, const Matrix2D<TypeTwo>& two) -> Matrix2D<decltype(one.Get(0,0) + two.Get(0,0))>
{
    THROW_IF( two.Height() != one.Height() || two.Width() != one.Width() , DimensionException, "Dimensions mismatch in assigning");

    Matrix2D<decltype( one.Get(0,0) + two.Get(0,0))> 
        ret(one.Width(),one.Height());
    for (uint i = 0; i < one.NumElems(); i++)
        ret(i) = one(i)+two(i);

    return ret;
}
*/
template<typename TypeOne, typename TypeTwo>
Matrix2D<TypeOne>& operator+(const Matrix2D<TypeOne>& one, const Matrix2D<TypeTwo>& two)
{
    THROW_IF( two.Height() != one.Height() || two.Width() != one.Width() , DimensionException, "Dimensions mismatch in assigning");

    auto ret = one;
    for (uint i = 0; i < one.NumElems(); i++)
        ret(i) += two(i);

    return ret;
}

template<typename TypeOne, typename TypeTwo>
inline auto operator*(const Matrix2D<TypeOne>& one, const Matrix2D<TypeTwo>& two) -> Matrix2D<decltype(one.Get(0,0) * two.Get(0,0))>
{
    THROW_IF( two.Height() != one.Height() || two.Width() != one.Width() , DimensionException, "Dimensions mismatch in assigning");

    Matrix2D<decltype( one.Get(0,0) * two.Get(0,0))> ret(one.Width(),one.Height());
    for (uint i = 0; i < one.Width()*one.Height(); i++)
        ret(i) = one(i)*two(i);

    return ret;
}

#endif

