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

#ifndef __SIMPLE_MATRIX__
#define __SIMPLE_MATRIX__


#include <iomanip>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <string>
#include <fstream>
#include <type_traits>

#include "Vec23.hxx"
#include "Utils.hxx"

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
    for(size_t y = s.y; y < size_t(e.y); ++y) \
            for(size_t x = s.x; x < size_t(e.x); ++x) 


#define for2d(pxy) for(size_t y = 0; y < pxy.y; ++y) for(size_t x = 0; x < pxy.x; ++x) 

#define for3d2(s,e) \
    for(size_t z = s.z; z < size_t(e.z); ++z) \
        for(size_t y = s.y; y < size_t(e.y); ++y) \
            for(size_t x = s.x; x < size_t(e.x); ++x) 

#define for3d(pxy) \
    for(size_t z = 0; z < pxy.z; ++z) \
        for(size_t y = 0; y < pxy.y; ++y) \
             for(size_t x = 0; x < pxy.x; ++x) 

#ifdef _DEBUG
#define IF_DEBUG(a) if((a))
#else 
#define IF_DEBUG(a) if(false)
#endif

#define iDivUp(a, b) (size_t)( (a % b) ? (a / b + 1) : (a / b) )

namespace SimpleMatrix
{
    template<typename T>
    T**  Reshape(T* in, size_t width, size_t height) // can convert T to T*, NDim to (N+1)Dim arrays;
    {
        T**  mat = new T*[height];
        for (size_t i = 0; i < height; i++)
            mat[i] = &(in[i*width]);
        return mat;
    }

    template<typename OneD, typename TwoD>
    void Reshape(OneD& oneD, TwoD& twoD, size_t width, size_t height)
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

    template<typename diffType>
    inline Vec::Size3 IndexTo3dIdx(diffType idx, const Vec::Size3& size)
    {
        return Vec::Size3(idx % size.y * size.z, (idx / size.x) % size.z, idx / (size.x*size.y));
    }

    // Pointers are not freed;
    template<typename T>
    struct Matrix
    {
        T** data;
        Vec::Size2 size;
        typedef T type;
        inline Matrix(Vec::Size2 s, T** d = 0) : size(s) { data = d ? d : ColocAlloc<T>(size); }

        T& at(Vec::Size2 loc) { 
            IF_DEBUG(loc.x >= size.x || loc.y >= size.y )
                throw std::out_of_range("Index invalid: ("
                    + std::to_string(loc.x) + "," + std::to_string(loc.y) + ")"); 
            return data[loc.y][loc.x]; 
        }
        const T& at(Vec::Size2 loc) const { 
            IF_DEBUG(loc.x >= size.x || loc.y >= size.y )
                throw std::out_of_range("Index invalid: (" + std::to_string(loc.x) + "," + std::to_string(loc.y) + ")"); 
            return data[loc.y][loc.x]; 
        }

        inline operator T**() { return data; }
        inline operator bool() { return size(); }
        inline void Set(T** d) { Clear();  data = d; }
        inline void Clear() {
            if (size() && data) {
                deleteColocArray(data);
                data = nullptr;
                size = { 0,0 };
            }
        }

        inline unsigned Height()const { return size.y; }
        inline unsigned Width() const { return size.x; }

        inline void Fill(const T& v) { std::fill(data[0], data[0] + size(), v); }

        template<typename U>
        T DotAt(Vec::Loc loc, const Matrix<U>& kernel) const
        {
            Vec::Size3 middle = kernel.size / 2;

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

        template<typename U>
        T DotCornerAt(Vec::Vec2<size_t> s, const Matrix<U>& kernel) const
        {
            Vec::Vec2<size_t> e = { s.x + kernel.size.x, s.y + kernel.size.y };

            T sum = 0;
            for2d2(s, e) sum += at(y, x) * kernel.at(y - s.y, x - s.x);

            return sum;
        }

        inline const T& at(int y, int x) const
        {
            IF_DEBUG(x < 0 || y < 0 || x >= int(size.x) || y >= int(size.y))
                throw std::out_of_range("Index invalid: " + std::to_string(x) + "," + std::to_string(y));

            return data[y][x];
        }

        inline T& at(int y, int x)
        {
            IF_DEBUG(x < 0 || y < 0 || x >= int(size.x) || y >= int(size.y))
                throw std::out_of_range("Index invalid: " + std::to_string(x) + "," + std::to_string(y));
            return data[y][x];
        }

        inline T* begin() { return data[0]; }
        inline T* end() { return data[0] + size(); }

        inline T& operator[](size_t idx) {

            IF_DEBUG(idx > size()) throw std::out_of_range("Index invalid: " + std::to_string(idx));

            return data[0][idx];
        }

        inline const T& operator[](size_t idx) const {
            IF_DEBUG(idx > size()) throw std::out_of_range("Index invalid: " + std::to_string(idx));

            return data[0][idx];
        }

        inline Matrix<T> Copy()
        {
            if (size())
            {
                Matrix<T> other = size;
                std::copy(data[0], data[0] + size(), other.data[0]);
                return other;
            }
            return *this;
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

    private:

    };

    template<typename T>
    struct Matrix3
    {
        T*** data;
        Vec::Size3 size;
        typedef T type;

        inline Matrix3(Vec::Size3 s = Vec::Zeroes3, T*** d = 0) : data(d ? d : ColocAlloc<T>(s)), size(s) {}

        inline T& at(Vec::Size3 loc) {
            IF_DEBUG(loc.x >= size.x || loc.y >= size.y || loc.z >= size.z)
                throw std::out_of_range("Index invalid: ("
                    + std::to_string(loc.x) + "," + std::to_string(loc.y) + "," + std::to_string(loc.z));

            return data[loc.z][loc.y][loc.x];
        }
        const T& at(Vec::Size3 loc) const {
            IF_DEBUG(loc.x >= size.x || loc.y >= size.y || loc.z >= size.z)
                throw std::out_of_range("Index invalid: ("
                    + std::to_string(loc.x) + "," + std::to_string(loc.y) + "," + std::to_string(loc.z));
            return data[loc.z][loc.y][loc.x];
        }

        inline operator T***() { return data; }
        inline void Set(Vec::Size3 s, T*** d = 0)
        {
            Clear();  size = s;
            if ((data = d) == 0) data = ColocAlloc<T>(size);
        }
        inline void ReSet(T*** d) { Clear();  data = d; }

        inline void Clear() {
            if (size() && data) {
                deleteColocArray(data);
                data = nullptr;
                size = { 0,0,0};
            }
        }

        inline unsigned Depth()  const { return size.z; }
        inline unsigned Height() const { return size.y; }
        inline unsigned Width()  const { return size.x; }

        inline Matrix3<T> Fill(const T& v) { std::fill(data[0][0], data[0][0] + size(), v); return *this; }

        // Dot product of two volumes: `kernel` with a bock of *this centered around `loc`
        template<bool PConnection, typename U> // partial connection, why template? Branching overhead avoidance.
        inline T DotAt(Vec::Loc loc, const Matrix3<U>& kernel, bool* connection = nullptr) const
        {
            //Logging::Log << "\nAt: " << loc << "\n";
            Vec::Size3 middle = kernel.size / 2; middle.z = 0;
#ifdef _MSC_VER
#pragma warning( push  )
#pragma warning( disable : 4267 )
#endif

            Vec::Loc start(loc.x - middle.x, loc.y - middle.y);
            Vec::Loc   end(start.x + kernel.size.x, start.y + kernel.size.y);
            auto iStart = start;
#ifdef _MSC_VER
#pragma warning( pop  )
#pragma warning( disable : 4267 )
#endif
            start.x = std::max(0, start.x);    start.y = std::max(0, start.y);
            end.x = std::min((int)size.x, end.x); end.y = std::min((int)size.y, end.y);

            T sum = 0;

            for (int z = 0; z < (int)size.z; ++z)
            {
                if (PConnection && !connection[z]) continue;
                for2d2(start, end)
                    sum += at(z, y, x) * kernel.at(z, y - iStart.y, x - iStart.x);
            }
            return sum;
        }

        template<bool PConnection, typename U>
        inline T DotCornerAt(Vec::Vec3<size_t> s, const Matrix3<U>& kernel, bool* connection = nullptr) const
        {
            Vec::Vec3<size_t> e = { s.x + kernel.size.x, s.y + kernel.size.y, size.z };

            T sum = 0;
            for3d2(s, e)
                if (!PConnection || connection[z]) 
                sum += at(z, y, x) * kernel.at(z, y - s.y, x - s.x);

            return sum;
        }
        
        inline T& at(int z, int y, int x)
        {

            IF_DEBUG(x < 0 || y < 0 || z < 0 || x >= int(size.x) || y >= int(size.y) || z >= int(size.z))
                throw std::out_of_range("Index invalid: (" + std::to_string(x) + "," + std::to_string(y) + "," + std::to_string(z));

            return data[z][y][x];
        }

        inline const T& at(int z, int y, int x) const
        {

            IF_DEBUG(x < 0 || y < 0 || z < 0 || x >= int(size.x) || y >= int(size.y) || z >= int(size.z))
                throw std::out_of_range("Index invalid: (" + std::to_string(x) + "," + std::to_string(y) + "," + std::to_string(z));

            return data[z][y][x];
        }

        inline Matrix<T> operator()(size_t z)
        {
            IF_DEBUG(z >= size.z) throw std::out_of_range("Index invalid: " + std::to_string(z));

            return{ size, data[z] };
        }

        inline const Matrix<T> operator()(size_t z) const
        {
            IF_DEBUG(z >= size.z) throw std::out_of_range("Index invalid: " + std::to_string(z));

            return{ size, data[z] };
        }

        inline T* begin()const { return data[0][0]; }
        inline T* end()  const { return data[0][0] + size(); }

        inline T& operator[](size_t idx) {
            IF_DEBUG(idx > size()) throw std::out_of_range("Index invalid: ");

            return data[0][0][idx];
        }

        inline const T& operator[](size_t idx) const {
            IF_DEBUG(idx > size()) throw std::out_of_range("Index invalid: ");

            return data[0][0][idx];
        }
    };

    // returns colocated array, delete Ret[0] and Ret
    template<typename T>
    inline T** NumToUnArray(unsigned N, T zeros = 0, T ones = 1)
    {
        Matrix<T> ret = Vec::Size2(N, N);
        ret.Fill(zeros);
        for (size_t i = 0; i < N; ++i)
            ret.data[i][i] = ones;
        return ret;
    }

    template<typename T>
    inline std::ostream& operator<<(std::ostream& out, const Matrix3<T>& k)
    {
        out << k.size;
        for (size_t i = 0; i < k.size.z; ++i) { out << "\nFrame " << i << " : " << k(i); }
        return out;
    }

    template<typename T>
    inline void OutCSV(std::ostream& out, const Matrix<T>& k, const char* msg)
    {
        out << msg << std::left << std::setprecision(9);
        for (size_t i = 0; i < k.size.h; ++i)
        {
            out << "\n";
            for (size_t j = 0; j < k.size.w; ++j)
                out << k.data[i][j] << ",\t";

        }
        out << "\n";
        out.flush();
    }

    template<typename T>
    inline void Out2d(std::ostream& outStream, T data, size_t w, size_t h, const char* msg = "", unsigned fwidth = 7)
    {
        outStream << msg << "\n";

        if (std::is_same<typename std::remove_const<T>::type, bool**>::value)
            fwidth = 2;

        unsigned digits = 0;
        size_t w1 = w; while (w1) w1 /= 10, digits++, outStream << ' ';
        outStream << " |";

        for (unsigned j = 0; j < w; ++j)
            outStream << std::setw(fwidth + 1) << std::setfill(' ') << std::left << j << " | ";

        std::string dashes(digits + 1, '-'); dashes += "|";

        std::string d(fwidth + 2, '-');    d += "|-";

        for (unsigned j = 0; j < w; ++j) dashes += d;

        bool isFloatType = std::is_floating_point<
            typename std::remove_const<
            typename std::remove_reference<decltype(data[0][0])>::type> ::type
        >::value;

        outStream << "\n" << dashes << "\n";
        for (unsigned i = 0; i < h; ++i)
        {
            outStream << std::setw(digits + 1) << std::setfill(' ') << i << "|";
            for (unsigned j = 0; j < w; ++j)
            {
                if (isFloatType)
                    outStream
                    << std::setprecision(fwidth - 2) << std::setw(fwidth + 1) << std::setfill('0')
                    << std::showpoint << std::left;
                else
                    outStream
                    << std::setw(fwidth + 1) << std::setfill(' ') << std::right;

                outStream << data[0][i*w + j] << " | ";
            }

            outStream << "\n";
        }
        outStream << dashes << "\n";

        outStream.flush();
    }

    template<>
    inline void Out2d(std::ostream& outStream, bool**& data, size_t w, size_t h, const char* msg, unsigned fwidth)
    {
        for (size_t i = 0; i < h; ++i)
        {
            outStream << "\n";
            for (size_t j = 0; j < w; ++j)
                outStream << data[i][j] << ",\t";
        }
    }

    template<typename T>
    inline std::ostream& operator<<(std::ostream& out, const Matrix<T>& k) {
        Out2d(out, k.data, k.size.w, k.size.h);
        //OutCSV(out, k, "");
        return out;
    }

    template <typename T>
    inline Matrix<T> ReadCSV(std::istream& strm)
    {
        const size_t BufferSize = 1024;
        char buffer[BufferSize];
        strm.getline(buffer, BufferSize, '\r');
        std::string line(buffer);
        size_t lineDem = line.find_first_of("\n\r");  // in Windows format there will be an extra '\n'
        if(lineDem != std::string::npos)
            line.erase(lineDem, std::string::npos);

        size_t width = std::count(line.begin(), line.end(), ',') + 1;

        if (!width) throw std::invalid_argument("Bad CSV file: number of columns is zero");

        strm.seekg(std::ios::beg);

        std::vector<T> buf; T in;
        while (strm.good())
        {
            char c = strm.peek();
            if (c == '\n' || c == '\r') break; // empty line read, done.
            buf.reserve(buf.size() + width);
            for (size_t j(0); j < width && strm; ++j)
            {
                strm >> in >> c;
                buf.push_back(in);
            }
            if (buf.size() % width) throw std::runtime_error("A full row could not be read");
            if (c != '\n' && c != '\r' && c != ',') strm.putback(c); // we already read a char from next row.
        }

        Matrix<T> out(Vec::Size2(width, buf.size() / width));
        std::copy(buf.begin(), buf.end(), out.data[0]);

        return out;
    }

    template<typename T>
    inline void WriteRawMatrix(std::ostream& out, const Matrix3<T>& mat)
    {
        Utils::WriteRawBytes(out, mat.size);
        out.write(static_cast<char*>(mat.data[0][0]), mat.size() * sizeof(T));
    }
    template <typename T>
    inline void WriteRawMatrix(std::ostream& out, const Matrix<T>& mat)
    {
        Utils::WriteRawBytes(out, mat.size);
        out.write(static_cast<char*>(mat.data[0]), mat.size() * sizeof(T));
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
        for (size_t i = loc.y; i < loc.y + innerSize.y; ++i)
            std::copy(outer[i], outer[i + innerSize.x], inner[i - loc.y]);

        return outer;
    }

    template<typename TInner, typename TOuter>
    void Emplace(Matrix<TInner>& inner, Matrix<TInner>& outer, std::unary_function<TInner, TOuter>& conv, Vec::Loc loc = { 0,0 })
    {
        for (unsigned i = loc.y; i < loc.y + inner.y; ++i)
            std::transform(outer[i], outer[i + inner.size.x], inner[i - loc.y], conv);
    }

    template<typename T>
    void RoundVertically(Matrix<T>& in, unsigned startCol = 0, unsigned endCol = unsigned(-1))
    {
        if (endCol == size_t(-1)) endCol = in.size.x;

        std::vector<Vec::pair<T> > minmax(endCol - startCol);

        for (unsigned i = startCol; i < endCol; ++i)
            minmax[i - startCol] = { in[0][i], in[0][i] };

        for (unsigned y = 1; y < in.size.y; ++y)
            for (unsigned x = startCol; x < endCol; ++x)
                minmax[x - startCol].x = MIN(minmax[x - startCol].x, in.at(y, x)),
                minmax[x - startCol].y = MAX(minmax[x - startCol].y, in.at(y, x));


        for (unsigned i = 0; i < minmax.size(); ++i)
            minmax[i].y -= minmax[i].x;


        for (unsigned y = 0; y < in.size.y; ++y)
            for (unsigned x = startCol; x < endCol; ++x)
                in[y][x] -= minmax[x - startCol].x,
                in[y][x] = minmax[x - startCol].y ? in[y][x] /= minmax[x - startCol].y : 0;
    }
}

#endif
