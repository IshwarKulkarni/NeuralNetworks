#ifndef __SIMPLE_MATRIX__
#define __SIMPLE_MATRIX__


#include <iomanip>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstring>

#include "Vec23.hxx"

/*
 Simple T** type matrix manipulation;
 
 Objective:
    Speed, 
    Co-location
Not-Objective:
    Ease of use, 
    User Responsibility
*/

#define for2d2(s,e) \
    for(unsigned y = s.y; y < unsigned(e.y); ++y) \
            for(unsigned x = s.x; x < unsigned(e.x); ++x) 


#define for2d(pxy) for(unsigned y = 0; y < pxy.y; ++y) for(unsigned x = 0; x < pxy.x; ++x) 

#define for3d2(s,e) \
    for(unsigned z = s.z; z < unsigned(e.z); ++z) \
        for(unsigned y = s.y; y < unsigned(e.y); ++y) \
            for(unsigned x = s.x; x < unsigned(e.x); ++x) 

#define for3d(pxy) \
    for(unsigned z = 0; z < pxy.z; ++z) \
        for(unsigned y = 0; y < pxy.y; ++y) \
             for(unsigned x = 0; x < pxy.x; ++x) 

namespace SimpleMatrix
{
    template<typename T>
    T**  Reshape(T* in, unsigned width, unsigned height) // can convert T to T*, NDim to (N+1)Dim arrays;
    {
        T**  mat = new T*[height];
        for (size_t i = 0; i < height; i++)
            mat[i] = &(in[i*width]);
        return mat;
    }

    template<typename OneD, typename TwoD>
    void Reshape(OneD& oneD, TwoD& twoD, unsigned width, unsigned height)
    {
        for (unsigned i = 0; i < height; i++)
            twoD[i] = &(oneD[i*width]);
    }

    template<typename T>
    T** ColocAlloc(Vec::Size2 size)
    {
        if (!size()) return 0;
        return Reshape((T*)memset(new T[size()], 0, sizeof(T)*size()), size.x, size.y);
    }

    template<typename T>
    T*** ColocAlloc(Vec::Size3 size)
    {
        if (!size()) return 0;
        return Reshape(ColocAlloc<T>(Vec::Size2(size.x, size.y * size.z)), size.y, size.z);
    }

    template<typename T>
    void deleteColocArray(T** arr)
    {
        delete[] arr[0]; delete[] arr;
    }

    template<typename T>
    void deleteColocArray(T*** arr)
    {
        delete[] arr[0][0];  delete[] arr[0]; delete[] arr;
    }

    // returns colocated array, delete Ret[0] and Ret
    template<typename T>
    inline T** NumToBinArray(unsigned N, unsigned& W, T zeros = 0., T ones = 1.)
    {
        W = unsigned(ceil(log2(N)));
        auto ret = ColocAlloc<T>(Vec::Size2(W, N));
        for (unsigned i = 0; i < N; ++i)
            for (unsigned j = 0, a = i; j < W; j++, a >>= 1)
                ret[i][j] = (a & 1) ? ones : zeros;

        return ret;
    }

    // Pointers are not freed;
    template<typename T>
    struct Matrix
    {
        T** data;
        Vec::Size2 size;
        typedef T type;
        inline Matrix(Vec::Size2 s, T** d = 0) : size(s) { data = d  ? d : ColocAlloc<T>(size); }
        
              T& at(Vec::Loc loc)        { return data[loc.y][loc.x]; }
        const T& at(Vec::Loc loc) const    { return data[loc.y][loc.x]; }
        
        inline operator T**() { return data;        }
        inline void Set(T** d){ Clear();  data = d;    }
        inline void Clear()      { if(size() && data) deleteColocArray(data); }
        
        inline unsigned Height()const { return size.y; }
        inline unsigned Width() const { return size.x; }
        
        inline void Fill(const T& v) { std::fill(data[0], data[0] + size(), v); }

        template <typename U, typename Copy>
        inline Matrix(const Matrix<U>& other, Copy& copier)
        {
            if(size != other.size)
                Clear(), data = ColocAlloc<T>(other.size), size = other.size;
            std::transform(other.data[0], other.data[0] + size(), data[0], copier);
        }

        template <typename U>
        inline Matrix(const Matrix<U>& other)
        {
            if (size == other.size)
#ifdef _MSC_VER
#pragma warning(disable:4244)
            std::copy(other.data[0], other.data[0] + size(), data[0]);
#pragma warning(default:4244)
#else
            std::copy(other.data[0], other.data[0] + size(), data[0]);
#endif
        }

        template<typename U>
        T DotAt(Vec::Loc loc, const Matrix<U>& kernel) const
        {
            Vec::Size3 middle = kernel.size / 2; middle.z = 0;

            Vec::Vec2<int> start(loc.x - middle.x, loc.y - middle.y);
            Vec::Vec2<int>   end(start.x + kernel.size.x, start.y + kernel.size.y);
            auto iStart = start;

            start.x = std::max(0, start.x);    start.y = std::max(0, start.y);
            end.x = std::min((int)size.x, end.x); end.y = std::min((int)size.y, end.y);

            T sum = 0;
            for (int y = start.y; y < end.y; ++y)
                for (int x = start.x; x < end.x; ++x)
                    sum += at(y, x) * kernel.at(y - iStart.y, x - iStart.x);

            return sum;
        }

        inline T& at(int y, int x)
        {
#ifdef _DEBUG
            if(x < 0 || y < 0 || x >= int(size.x) || y >= int(size.y) )
                throw std::out_of_range("Index invalid: " + std::to_string(x) + "," + std::to_string(y));
#endif
            return data[y][x];
        }

        inline const T& at(int y, int x) const
        {
#ifdef _DEBUG
            if (x < 0 || y < 0 || x >= int(size.x) || y >= int(size.y))
                throw std::out_of_range("Index invalid: " + std::to_string(x) + "," + std::to_string(y));
#endif
            return data[y][x];
        }

        inline T* begin() { return data[0]; }
        inline T* end() { return data[0] + size(); }

        inline T& operator[](size_t idx) {
#ifdef _DEBUG
            if (idx >  size()) throw std::out_of_range("Index invalid: " + std::to_string(idx) );
#endif
            return data[0][idx];
        }

        inline const T& operator[](size_t idx) const {
#ifdef _DEBUG
            if (idx >  size()) throw std::out_of_range("Index invalid: " + std::to_string(idx));
#endif
            return data[0][idx];
        }
    };

    template<typename T>
    struct Matrix3
    {
        T*** data;
        Vec::Size3 size;
        typedef T type;
        
        inline Matrix3(Vec::Size3 s = Vec::Zeroes3, T*** d = 0) : data(d ? d : ColocAlloc<T>(s)) , size(s) {}

              T& at(Vec::Size3 loc)       { return data[loc.z][loc.y][loc.x]; }
        const T& at(Vec::Size3 loc) const { return data[loc.z][loc.y][loc.x]; }
        
        inline operator T***() { return data; }
        inline void Set(Vec::Size3 s, T*** d = 0)
        {
            Clear();  size = s; 
            if ( (data = d) == 0 ) data = ColocAlloc<T>(size);
        }
        inline void ReSet(T*** d) { Clear();  data = d;  }
        
        inline void Clear() { if (size() && data) deleteColocArray(data); }
        
        inline unsigned Depth()  const { return size.z; }
        inline unsigned Height() const { return size.y; }
        inline unsigned Width()  const { return size.x; }
        
        inline Matrix3<T> Fill(const T& v) { std::fill(data[0][0], data[0][0] + size(), v); return *this; }
        
        template <typename U>
        inline Matrix3(const Matrix<U>& other)
        {
            size.y = other.size.y, size.x = other.size.x;
            data = ColocAlloc<T>(size);
            std::copy(other.data[0], other.data[0] + size(), data[0][0]);
        }
        
        template <typename U, typename Copy>
        inline Matrix3(const Matrix3<U>& other, Copy& copier)
        {
            if (size != other.size)
                Clear(), data = ColocAlloc<T>(other.size), size = other.size;
            std::transform(other.data[0][0], other.data[0][0] + size(), data[0][0], copier);
        }

        template <typename U>
        inline Matrix3(const Matrix3<U>& other)
        {
            if (size != other.size)
                Clear(), data = ColocAlloc<T>(other.size), size = other.size;
            std::copy(other.data[0][0], other.data[0][0] + size(), data[0][0]);
        }
    
        // Dot product of two volumes: `kernel` with a bock of *this centered around `loc`
        template<typename U>
        T DotAt(Vec::Loc loc, const Matrix3<U>& kernel) const
        {
            //Logging::Log << "\nAt: " << loc << "\n";
            Vec::Size3 middle = kernel.size / 2; middle.z = 0;

            Vec::Vec2<int> start( loc.x - middle.x, loc.y - middle.y);
            Vec::Vec2<int>   end(start.x + kernel.size.x, start.y + kernel.size.y);
            auto iStart = start;
            
            start.x = std::max(0, start.x);    start.y = std::max(0, start.y);
            end.x = std::min((int)size.x, end.x); end.y = std::min((int)size.y, end.y);

            T sum = 0;
            for (int z = 0; z < (int)size.z; ++z)
                for2d2(start,end)
                    sum += at(z, y, x) * kernel.at(z, y - iStart.y, x - iStart.x);

            return sum;
        }

        inline T& at(int z, int y, int x)
        {
#ifdef _DEBUG
            if (x < 0 || y < 0 || z < 0|| x >= int(size.x) || y >= int(size.y)|| z >= int(size.z))
                throw std::out_of_range("Index invalid: (" + std::to_string(x) + "," + std::to_string(y)+ ","+ std::to_string(z));
#endif 
            return data[z][y][x];
        }
        
        inline const T& at(int z, int y, int x) const
        {
#ifdef _DEBUG
            if (x < 0 || y < 0 || z < 0 || x >= int(size.x) || y >= int(size.y) || z >= int(size.z) )
                throw std::out_of_range("Index invalid: (" + std::to_string(x) + "," + std::to_string(y) + ","+std::to_string(z));
#endif
            return data[z][y][x];
        }

        inline Matrix<T> operator()(unsigned z)
        {
#ifdef _DEBUG
            if (z >= size.z)
                throw std::out_of_range("Index invalid: " + std::to_string(z) );
#endif 
            return{ size, data[z]};
        }

        inline const Matrix<T> operator()(unsigned z) const
        {
#ifdef _DEBUG
            if (z >= size.z)
                throw std::out_of_range("Index invalid: " + std::to_string(z));
#endif 
            return{ size, data[z] };
        }

        inline T* begin()const { return data[0][0]; }
        inline T* end()  const { return data[0][0] + size(); }

        inline T& operator[](size_t idx) {
#ifdef _DEBUG
            if (idx >  size()) throw std::out_of_range("Index invalid: " );
#endif
            return data[0][0][idx];
        }

        inline const T& operator[](size_t idx) const {
#ifdef _DEBUG
            if (idx >  size()) throw std::out_of_range("Index invalid: ");
#endif
            return data[0][0][idx];
        }
    };

    // returns colocated array, delete Ret[0] and Ret
    template<typename T>
    inline T** NumToUnArray(unsigned N, T zeros = 0, T ones = 1)
    {
        Matrix<T> ret = Vec::Size2(N, N);
        ret.Fill(zeros);
        for (size_t i = 1; i < N; ++i)
            ret.data[i][i] = ones;
        return ret;
    }

    template<typename T, typename O>
    O& operator<<(O& out, const Matrix3<T>& k)
    {
        out << k.size ;
        for (size_t i = 0; i < k.size.z; ++i)
            OutCSV(out, k(i), k.size.x, k.size.y, ("\nFrame[ " + std::to_string(i) + " ]"  + ",\t").c_str());

        return out;
    } 

    template<typename T, typename O>
    inline void OutCSV(O& out, const Matrix<T>& k, unsigned w, unsigned h, const char* msg)
    {
        out << std::setprecision(9);
        for (unsigned i = 0; i < h; ++i)
        {
            out << "\n";
            for (unsigned j = 0; j < w; ++j)
                out << k.data[i][j] << ", ";
            
        }
        out << "\n";
        out.flush();
    }

    template<typename O, typename T>
    inline void Out2d(O& outStream, T& data, unsigned w, unsigned h, const char* msg = "", unsigned fwidth = 7)
    {
        outStream << msg << "\n";
        unsigned digits = 0;
        unsigned w1 = w; while (w1) w1 /= 10, digits++, outStream << ' ';
        outStream << " |";

        for (unsigned j = 0; j < w; ++j)
            outStream << std::setw(fwidth + 1) << std::setfill(' ') << std::left << j << " | ";

        std::string dashes(digits+1, '-'); dashes += "|";
        
        std::string d(fwidth + 2, '-');    d += "|-";
            
        for (unsigned j = 0; j < w; ++j) dashes += d;

        outStream << "\n"<< dashes << "\n";
        for (unsigned i = 0; i < h; ++i)
        {
            outStream << std::setw(digits + 1) << std::setfill(' ') << i <<  "|";
            for (unsigned j = 0; j < w ; ++j)
                outStream    << std::setprecision(fwidth-2) << std::setw(fwidth + 1) << std::setfill('0')
                            << std::showpoint << std::left<< std::fixed
                            << data[i][j] << " | ";

            outStream << "\n";
        }
        outStream << dashes << "\n";

        outStream.flush();
    }

    template <typename T>
    Matrix<T> ReadCSV(std::istream& strm)
    {
        std::string line;
        std::getline(strm, line, '\n');
        unsigned width = unsigned(std::count(line.begin(), line.end(), ',') + 1), height = 0;

        if(!width)
            throw std::invalid_argument("File is not organized correctly, number of columns appear to be zero");

        strm.seekg(std::ios::beg);

        std::vector<T*> buf;
        while (strm.good())
        {
            buf.push_back(new T[width]);
            char c = '\0';
            for (unsigned j = 0; j < width - 1; ++j)
                strm >> buf[height][j] >> c;
            strm >> buf[height++][width - 1];
        }

        Matrix<T> out(Vec::Size2(width, height));
        for (unsigned i = 0; i < buf.size(); ++i)
            copy(buf[i], buf[i]+width, out[i]);

        return out;
    }

    template <typename T>
    void ReadCSV(std::istream& strm, unsigned& width, unsigned& height, std::vector<T>& buf)
    {
        height = 0;

        std::string line;
        std::getline(strm, line, '\n');
        width = (unsigned)(std::count(line.begin(), line.end(), ',') + 1);

        if (!width)
            throw std::invalid_argument("File is not organized correctly, number of columns appear to be zero");

        strm.seekg(std::ios::beg);

        while (strm.good())
        {
            ++height;
            buf.resize(buf.size() + width); //  gr one r at a time, this is slow, but faster than file read, so forget it.
            char c = '\0';
            for (unsigned j = 0; j < width; ++j)
            {
                strm >> buf[(height - 1)*width + j];
                c = strm.get();
                if(strm.good() && c != ',' && j != width - 1)
                    throw std::invalid_argument("Delimiter is invalid");
            }
        }
    }

  
    template<typename T>
    void  ReshapeUnmanaged(T**& mat, T*& in, unsigned width, unsigned height) // can convert T to T*, NDim to (N+1)Dim arrays;
    {
        in = new T[width*height];
        mat = new T*[height];
        for (size_t i = 0; i < height; i++)
            mat[i] = &(in[i*width]);
    } 

    template<typename TInner, typename TOuter>
    TOuter** Emplace(TInner* inner, Vec::Size2 innerSize, Vec::Size2 outerSize, Vec::Loc loc = { 0,0 })
    {
        TOuter** outer = ColocAlloc<TOuter>(outerSize);
        for (unsigned i = loc.y; i < loc.y + inner.y; ++i)
            std::copy(outer[i], outer[i + innerSize.x], inner[i - loc.y]);
    }

    template<typename TInner, typename TOuter>
    void Emplace(Matrix<TInner>& inner, Matrix<TInner>& outer, std::unary_function<TInner, TOuter>& conv, Vec::Loc loc = { 0,0 })
    {
        for (unsigned i = loc.y; i < loc.y + inner.y; ++i)
            std::transform(outer[i], outer[i + inner.size.x], inner[i - loc.y], conv);    }

    template<typename T>
    void RoundVertically(Matrix<T>& in, unsigned startCol = 0, unsigned endCol = -1)
    {
        if (endCol == -1) endCol = in.size.x;

        std::vector<Vec::pair<T> > minmax(endCol - startCol);

        for (unsigned i = startCol; i < endCol; ++i)
            minmax[i-startCol] = { in[0][i], in[0][i] };

        for (unsigned y = 1; y < in.size.y; ++y)
            for (unsigned x = startCol; x < endCol; ++x)
                 minmax[x-startCol].x = MIN(minmax[x-startCol].x, in.at(y,x)),
                minmax[x-startCol].y = MAX(minmax[x-startCol].y, in.at(y,x));


        for (unsigned i = 0; i < minmax.size(); ++i)
            minmax[i].y -= minmax[i].x;


        for (unsigned y = 0; y < in.size.y; ++y)
            for (unsigned x = startCol; x < endCol; ++x)
                in[y][x] -= minmax[x - startCol].x,
                in[y][x] = minmax[x - startCol].y ? in[y][x] /= minmax[x - startCol].y : 0;
    }

    static Matrix<float> SobelV(Vec::Size2(3, 3)), SobelH(Vec::Size2(3, 3));

    static bool MakeFilters()
    {
        float sobelV[] = { -1, -2, -1 , 0, 0, 0 ,  1, 2, 1 };
        float sobelH[] = { -1, -0,  1 ,-2, 0, 2 , -1, 0, 1 };

        for (unsigned i = 0; i < SobelV.size(); ++i)
            SobelV[i] = sobelV[i], SobelH[i] = sobelH[i];
        
        return true;
    }

    static const bool FiltersMade = MakeFilters();
}
#endif
