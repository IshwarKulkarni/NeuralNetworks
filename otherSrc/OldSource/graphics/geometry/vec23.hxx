#ifndef __VEC2__
#define __VEC2__

namespace Vec
{
    template<typename T>
    struct Vec2
    {
        union
        {
            struct {
                T x, y;
            };
            struct {
                T w, h;
            };
            T  Raw[2];
        };

        inline Vec2(T _x = 0, T _y = 0) : x(_x), y(_y) {}

        /*constexpr*/ size_t size() const { return 2; }
        operator T*() { return Raw; }
        inline unsigned operator()() const { return x*y; }
    };

    // element-wise operator
    typedef Vec2<unsigned> Size2;

    typedef Vec2<int> Loc;
    const Loc Origin = Loc(0, 0);
    
    template<typename T>
    using pair = Vec2< T >;

    template<typename T>
    struct Vec3
    {
        union
        {
            struct {
                T x, y, z;
            };
            struct {
                T R, G, B;
            };
            T  Raw[3];
        };

        Vec3(T x_ = 0, T y_ = 0, const T& z_ =0) : x(x_), y(y_), z(z_) {}
        Vec3(Vec2<T> vec2, T _z = 1) : x(vec2.x), y(vec2.y), z(_z) {}
        
        inline unsigned operator()() const { return x*y*z; }
        /*constexpr*/ size_t size() const { return 3; }
        operator T*() { return Raw; }
        template<typename U> operator Vec2<U>() const { return Vec2<U>(x,y); }

        template<typename U> Vec3<T> operator+=(const Vec3<U>& o) { x += o.x, y += o.y, z += o.z; return *this; }
        template<typename U> Vec3<T> operator*=(const Vec3<U>& o) { x *= o.x, y *= o.y, z *= o.z; return *this; }

    };

    typedef Vec3<unsigned> Size3;
    static const Size3  Zeroes3(0, 0, 0);

    typedef Vec3<int> Loc3;
    
    static const Loc3 Origin3(0, 0,0);

    template<typename OStream, typename T>
    inline OStream& operator<<(OStream& os, const Vec3<T>& t)
    {
        os << "[" << t.x << ", " << t.y << ", " << t.z << "]";
        return os;
    }

    template<typename OStream, typename T>
    inline OStream& operator<<(OStream& os, const Vec2<T>& t)
    {
        os << "[" << t.x << ", " << t.y << "]";
        return os;
    }


    template<typename T, typename IStream>
    inline IStream& operator>>(IStream& is, Vec3<T>& size)
    {
        char c;
        is >> size.x >> c >> size.y >> c >> size.z;
        return is;
    }

    template<typename T, typename IStream>
    inline IStream& operator>>(IStream& is, Vec2<T>& size)
    {
        char c;
        is >> size.x >> c >> size.y;
        return is;
    }

    template<typename T, typename Scalar> inline Vec3<T> operator/ (const Vec3<T>& t, Scalar s) { return Vec3<T>(t.x / s, t.y / s, t.z / s); }
    template<typename T, typename U> inline bool operator==(const Vec3<T>& t, const Vec3<U>& s) { return t.x == s.x &&  t.y == s.y && t.z == s.z; }
    template<typename T, typename U> inline bool operator==(const Vec2<T>& t, const Vec2<U>& s) { return t.x == s.x &&  t.y == s.y; }

    template<typename T, typename U> inline bool operator!=(const Vec3<T>& t, const Vec3<U>& s) { return !(t == s); }
    template<typename T, typename U> inline bool operator!=(const Vec2<T>& t, const Vec2<U>& s) { return !(t == s); }

    template<typename T, typename Scalar> inline Vec2<T> operator/ (const Vec2<T>& t, Scalar s) { return Vec2<T>(t.x / s, t.y / s); }
    template<typename T, typename U> inline Vec2<T> operator*(const Vec2<T>& t, const Vec2<U>& s) { return{ t.x*s.x , t.y*s.y }; }
    template<typename T, typename U> inline Vec2<T> operator/(const Vec2<T>& t, const Vec2<U>& s) { return{ t.x*s.x , t.y*s.y }; }
}
#endif
