#ifndef _BASIC_GEOMETRY_HXX_
#define _BASIC_GEOMETRY_HXX_

#define ABSOLUTE(a) (a<0?-a:a)

#ifdef __CUDACC__
#define FINLINE __forceinline__  __host__ __device__
#else
#define FINLINE __forceinline
#endif

#define VECFORALL for (unsigned i = 0; i < 3; i++)

template<typename T>
struct Interval
{
    T min, max;
    FINLINE T width() const { return ABSOLUTE(max - min); }
};

template<typename T>
struct Vec // a nice quadruplet of aligned elements.
{
    T x, y, z, w;
    FINLINE T& operator[](unsigned i) // make it look like array
    {
        // screwed if these functions are laid out before members
        // So don't make virtual functions;
        unsigned long long  ptr = (unsigned long long) this;
        ptr += sizeof(T) * i;

        return *(T*)(ptr);
    }

    FINLINE const T& operator[](unsigned i) const
    {
        return *(T*)((unsigned long long)(this) + sizeof(T) * (unsigned long long)(i));
    }

    FINLINE Vec<T>(const T& a, const T& b, const T& c, const T& d) :
        x(a), y(b), z(c), w(d){}

    FINLINE Vec<T>(const T& a, const T& b, const T& c) :
        x(a), y(b), z(c){}

    FINLINE Vec<T>(){};

    template<typename U>
    FINLINE Vec<T>(const Vec<U>&  other) { 
        x = other.x,
            y = other.y,
            z = other.z;
    }
};

template<typename T>using Vec3 = Vec < T > ;
template<typename T>using Vec4 = Vec < T > ;

template<typename T>
FINLINE T magnitude2(const Vec3<T>& v)
{
    return v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
}

template<typename T>
FINLINE T magnitude(const Vec3<T>& v)
{
    return sqrt(magnitude2(v));
}

template<typename T>
FINLINE Vec<T>& normalize(Vec<T>& v)
{
    T m = magnitude(v);
    VECFORALL{ v[i] /= m; }
    v.w = m;
    return v;
}

template<typename T>
FINLINE bool operator==(const Vec<T>& a, const Vec<T>& b)
{
    return !memcmp(&a, &b, sizeof(a) - sizeof(a.x));
}


template<typename T, typename U>
FINLINE bool operator==(const Vec<T>& a, const Vec<U>& b)
{
    return a.x == b.x && a.y == b.y && a.z == b.z;
}

template<typename T, typename U>
FINLINE Vec3<T> operator+(Vec3<T> a, const Vec3<U>& b) { VECFORALL{ a[i] += b[i]; } return a; }

template<typename T, typename U>
FINLINE Vec3<T> operator-(Vec3<T> a, const Vec3<U>& b) { VECFORALL{ a[i] -= b[i]; } return a; }

template<typename T, typename U>
FINLINE void operator+= (Vec3<T>& a, const Vec3<U>& b) { VECFORALL{ a[i] += b[i]; } } 

template<typename T, typename U>
FINLINE void operator-= (Vec3<T>& a, const Vec3<U>& b) { VECFORALL{ a[i] -= b[i]; } }

template<typename T, typename Scalar>
FINLINE Vec3<T> operator * (const Scalar& s, const Vec3<T>& v)
{
    return Vec<T>(v.x * s, v.y * s, v.z * s);
}

template<typename T, typename Scalar>
FINLINE Vec3<T> operator/(const Scalar& s, const Vec3<T>& v)
{
    return Vec<T>(v.x / s, v.y / s, v.z / s);
}

template<typename T, typename Scalar>
FINLINE Vec3<T> operator * (const Vec3<T>& v, const Scalar& s)
{
    return Vec<T>( v.x * s, v.y * s, v.z * s );
}

template<typename T, typename Scalar>
FINLINE Vec3<T> operator/(const Vec3<T>& v, const Scalar& s)
{
    return Vec3<T>( v.x / s, v.y / s, v.z / s );
}

template<typename T, typename Scalar>
FINLINE void operator *= (Vec3<T>& v, const Scalar s) { VECFORALL{ v[i] *= s; } }

template<typename T, typename Scalar>
FINLINE void operator /= (Vec3<T>& v, const Scalar s) { VECFORALL{ v[i] /= s; } }

template<typename T, typename U>
FINLINE Vec3<T> operator ^ (const Vec3<T>& a, const Vec3<U>& b) // cross product
{
    return Vec < T > {
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0]
    };
}


template<typename T, typename U> // Remove the component of A that is parallel to B.
FINLINE Vec3<T> operator/(const Vec3<T> &A, const Vec3<U> &B)
{
    T x(magnitude2(B));
    return (x > 0) ? (A - ((A * B) / x) * B) : A;
}

template<typename T, typename U>
FINLINE T operator * (const Vec3<T>& a, const Vec3<U>& b) // dot product
{
    return (a.x*b.x + a.y*b.y + a.z*b.z);
}

// float instantiations
typedef Interval<float> Intervalf;

#define Epsilon  1e-5f
typedef Vec3<float> Vec3f;
typedef Vec4<float> Vec4f;

static const Vec3f NullVec3f( 0 , 0, 0);
static const Vec3f EpsVec3f(Epsilon, Epsilon, Epsilon); 
static const Vec3f XAxisf(1, 0, 0);
static const Vec3f YAxisf(0, 1, 0);
static const Vec3f ZAxisf(0, 0, 1);

FINLINE bool almostEqual(float a, float b) { return ABSOLUTE(a - b) < Epsilon; }
FINLINE bool scaledAlmostEqual(float a, float b, float c) { return ABSOLUTE(a - b) < Epsilon * c; }

template<>
FINLINE bool operator==<float>(const Vec3<float>& a, const Vec3<float>& b)
{
    return
        almostEqual(a.x, b.x) &&
        almostEqual(a.y, b.y) &&
        almostEqual(a.z, b.z);
}

typedef Vec<Vec3f> Matf3x3;

inline Matf3x3 Transpose(const Matf3x3 &M)
{
    Matf3x3 W;
    VECFORALL for (unsigned j = 0; j < 3; j++) 
            W[i][j] = M[j][i];
    return W;
}

struct Colorf: Vec<float>
{
    FINLINE Colorf(unsigned) : Vec < float > ({ 0.f, 0.f, 0.f }){} // fake constructor
    FINLINE Colorf() : Vec < float >({ 0.f, 0.f, 0.f }){}
    FINLINE Colorf(Vec3f a) : Vec < float >(a){}
    FINLINE Colorf(float a, float b, float c) : Vec < float >({a,b,c}){}
};

static const Colorf Blackf(0, 0, 0);
static const Colorf Redf(255, 0, 0);
static const Colorf Greenf(0, 255, 0);
static const Colorf Bluef(0, 0, 255);
static const Colorf Yellowf(255, 255, 0);
static const Colorf Magentaf(255, 0, 255);
static const Colorf Cyanf(0, 255, 255);
static const Colorf Whitef(255, 255, 255);

static const Colorf Grayf(128, 128, 128);
static const Colorf Orangef(255, 128, 0);
static const Colorf Puplef(128, 0, 255);

inline Matf3x3 Adjugate(const Matf3x3& M)
{
    Matf3x3 A;
    A[0][0] = M[1][1] * M[2][2] - M[1][2] * M[2][1];
    A[0][1] = M[1][2] * M[2][0] - M[1][0] * M[2][2];
    A[0][2] = M[1][0] * M[2][1] - M[1][1] * M[2][0];
    
    A[1][0] = M[0][2] * M[2][1] - M[0][1] * M[2][2];
    A[1][1] = M[0][0] * M[2][2] - M[0][2] * M[2][0];
    A[1][2] = M[0][1] * M[2][0] - M[0][0] * M[2][1];
    
    A[2][0] = M[0][1] * M[1][2] - M[0][2] * M[1][1];
    A[2][1] = M[0][2] * M[1][0] - M[0][0] * M[1][2];
    A[2][2] = M[0][0] * M[1][1] - M[0][1] * M[1][0];
    return A;
}

inline float det(const Matf3x3 &M)
{
    return
          M[0][0] * (M[1][1] * M[2][2] - M[1][2] * M[2][1])
        - M[0][1] * (M[1][0] * M[2][2] - M[1][2] * M[2][0])
        + M[0][2] * (M[1][0] * M[2][1] - M[1][1] * M[2][0]);
}



// Compute the inverse of a matrix using its Adjugate 
// and determinant.
inline Matf3x3 Inverse(const Matf3x3& M)
{
    return Transpose(Adjugate(M)) / det(M);
}
#endif
