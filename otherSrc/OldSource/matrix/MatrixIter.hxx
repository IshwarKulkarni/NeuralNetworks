#ifndef MATRIXITER_INCLUDED
#define MATRIXITER_INCLUDED

#include "Matrix.hxx"
#include <sstream>
#include <algorithm>

typedef unsigned int uint;
typedef unsigned short ushort;

template <typename MatrixType, int IncX=1, int IncY=1>
class MatrixIter 
{

protected:
    MatrixType& m_Mat;
    uint sy, sx;
    uint ey, ex;
    uint py, px; // current ptr;
    uint xExtent, yExtent;
    typename const MatrixType::ElemType* EndVal;
public:
    /*
        |<---------------mat.Width()-------------->|
 (0,0)->|_________________________________________ _
        |                                          | ^
        |         (sx,sy)       (px,py)            | |
        |        _   \ _________/_______________   | |
        |        ^    |        /                |  | |
        |        |    |       /                 |  | |
        |        |    |     *                   |  | |
        |        |    |                         |  | mat.Height() 
        |    yExtent  |                         |  | |
        |        |    |                         |  | |
        |        |    |                         |  | |
        |        V    |_________________________|  | |
        |             |<---------xExtent------->|  | |
        |                                          | |
        |__________________________________________| V 
*/ 

template <typename MatrixType,  char IncX=1, char IncY=1> // incX/Y = direction of increasing x/y
    MatrixIter(MatrixType& mat, uint startX, uint startY, uint width , uint height ): 
        m_Mat(mat), 
        sy(max(mat.Height(), startY)),
        sx(max(mat.Width(),  startX)),
        ey(min(mat.Height(), sy + height)),
        ex(min(mat.Width() , sx + width )), 
        px(sx), 
        py(sy), 
        xExtent(ex-sx), 
        yExtent(ey-sy),
        EndVal ( m_Mat.GetData() + mat.Width() * mat.Height())
    {
    }    
    
    inline uint Width() const { return xExtent;}
    inline uint Height() const { return yExtent;}
    inline uint X() const { return px; }
    inline uint Y() const { return py; }
    inline MatrixType& Mat() const { return m_Mat; }
    
    virtual inline MatrixIter<MatrixType>& operator++()
    {
        px += IncX;

        if (IncX > 0 && px == ex) py += IncY, px = sx;
        if (IncX < 0 && px <  sx) py += IncY, px = ex-1;

        return *this;

    }

    virtual inline MatrixIter<MatrixType>& operator--()
    {
        px -= IncX;
        if (IncX > 0 && px <  sx) py -= IncY, px = ex - 1;
        if (IncX < 0 && px == ex) py -= IncY, px = sx;
        return *this;
    }

    virtual inline MatrixIter<MatrixType>& operator+=(int i)
    {
        py += (i * IncY) / xExtent;
        px += (i * IncX)% xExtent;
        if (IncX > 0 && px == ex) py += IncY, px = sx;
        if (IncX < 0 && px <  sx) py += IncY, px = ex - 1;
        return *this;
    }

    virtual inline MatrixIter<MatrixType>& operator-=(int i)
    {
        *this += (-i);
        return *this;
    }

    template <typename Fn, typename Pr>
    void ForEach(Fn f, Pr p)
    {
        while(p(sx,sy))
            f(m_Mat(sx,sy));
    }

    inline typename MatrixType::ElemType& operator*()
    {
        return get(px, py);
    }

    inline typename MatrixType::ElemType& get(uint tpx, uint tpy) const
    {
        THROW_IF(IsOOB(tpx,tpy), DimensionException, "Dereferencing error in %s", this->ToString(tpx, tpy).c_str());
        return m_Mat(tpx, tpy);
    }

    virtual inline bool IsOOB(uint tpx , uint tpy ) const
    {
        return (tpx >= ex || tpy >= ey || tpx < sx || tpy < sy);
    }

    std::string ToString(uint tpx , uint tpy ) const
    {
        std::stringstream ss;
        ss << "Iterator on Matrix: ";
        m_Mat.Identify(ss);
        ss << "\nBounds: \n" 
            << "(start y, start x) = (" << sy << ", " << sx << ")\n"
            << "(end y  , end   x) = (" << ey << ", " << ex << ")\nCurrent: \n" 
            << "(y, x) = (" << tpy << ", " << tpx << ")";

        return ss.str();
    }

    inline bool operator== (const MatrixIter<MatrixType>& other)
    {
        return (
            m_Mat == other.Mat() && (
            px == other.X() &&
            py == other.Y()
            )
            ||
            this->IsOOB(px,py) && other.IsOOB()
            );
    }

    inline bool operator!= (const MatrixIter<MatrixType>& other) const { return !(*this==other); }

    // behave like STL containers:
    virtual inline uint size() const { return xExtent*yExtent; }
    virtual inline typename MatrixType::ElemType* begin()  { return &(m_Mat(px,py)); }
    virtual inline typename MatrixType::ElemType*   end() const { return nullptr; }
    virtual inline typename MatrixType::ElemType&   operator[](uint i) const 
    {
        uint npy = (py + (i * IncY) / xExtent);
        uint npx = (px + (i * IncX) % xExtent);
        return get(npy, npx);

    }
};

template<typename MatrixType>
inline std::ostream& operator << (std::ostream& os, const MatrixIter<MatrixType>& mat)
{
    os << mat->ToString();
    return os;
}

template <typename MatrixType>
class MatrixRowIter : public MatrixIter<MatrixType>
{
public:
    MatrixRowIter(MatrixType& mat, uint rowNum) : 
        MatrixIter<MatrixType>(mat, 0, rowNum,mat.Width(), 1){}
};

/*    void Snake(std::vector<Type>& returnVec)
    {
        returnVec.reserve(m_Width*m_Height);
        for( uint s = 0; s < 2*m_Width-1; ++s )
        {
            uint i = 0, end (s), id(1);
            if(s%2 == s/m_Width)  { std::swap(i, end); id = -1;}
            do { returnVec.push_back( Get(i , s-i)) ; }while(i+=id != end);
        }

        return returnVec;
    }
*/

#endif
