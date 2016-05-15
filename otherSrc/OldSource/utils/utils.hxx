#ifndef _UTILS_HXX_INCLUDED_
#define _UTILS_HXX_INCLUDED_

#include <algorithm>
#include <cmath>
#include <istream>
#include <iterator>
#include <limits>
#include <memory>
#include <random>
#include <set>
#include <string>
#include <type_traits>
#include <vector>

#include "Exceptions.hxx"
#include "Logging.hxx"
#include "StringUtils.hxx"

#define ARRAY_LENGTH(arr)    (arr == 0 ? 0 : sizeof(arr)/sizeof(arr[0]) )
#define ARRAYEND(arr) (arr + ARRAY_LENGTH(arr))

#define ESC_KEY ((char)(27))

#define PASS_COUNT() Logging::Log << " >> " << __FILE__ << ":" <<  __LINE__ <<  "in function " << __FUNCTION__ << " pass - " << __COUNTER__  << LogEndl; 

#define MAX(a,b) (((a) > (b)) ? (a) : (b))
#define MIN(a,b) (((a) < (b)) ? (a) : (b))

#define iDivUp(a, b) (uint)( (a % b) ? (a / b + 1) : (a / b) )

#define MAYBE if(Utils::URand(1.0)<0.01)

typedef unsigned char uchar;

typedef unsigned int uint;


namespace ImageIO
{
    enum ImageFormatType
    {
        JPEG,
        PPM,
        ERROR_TYPE
    };
}

namespace Utils {
    
    /*struct Identity {
        template<typename U>
        constexpr auto operator()(U&& v) const noexcept
            -> decltype(std::forward<U>(v))
        {
            return std::forward<U>(v);
        }
    };*/

    template<typename T, typename U>
    double GetRMSE(T _1, T _1end, U _2)
    {
        double sum = 0;
        while (_1 != _1end)
        {
            auto p = (*_1++ - *_2++);
            sum = p*p;
        }
        return sqrt(sum);
    }

    template<typename O, typename T>
    void Out2d(O& outStream, const std::vector< std::vector<T> >& data, unsigned w, unsigned h, const char* msg = "")
    {
        std::string dashes(w, '-');
        outStream << msg << LogEndl << dashes;

        for (unsigned i = 0; i < h; ++i)
        {
            for (unsigned j = 0; j < w - 1; ++j)
                outStream << " | " << data[i][j] << ', ';
            outStream << data[i][w - 1] << " |" << LogEndl;
        }

        outStream << msg << LogEndl << dashes;
        outStream.flush();
    }
    
    inline std::streamsize GetStreamSize(std::istream& in)
    {
        auto cur = in.tellg();
        in.seekg(0, std::ios::end);
        auto size = in.tellg();
        in.seekg(cur);
        return size;
    }

    template <
        typename MapType,
        typename KeySetType, // does not really need to be a set.
        typename ValueSetType // does not really need to be a set.
    >
    size_t FlattenMap(const MapType& map, KeySetType& keys, ValueSetType& values, bool fillKeys = true, bool fillValues = true)
    {
        if (!fillKeys && !fillValues) return 0;
        size_t numInserted = 0;

        for (auto s : map)
        {
            if (fillKeys)     keys.insert(keys.end(), KeySetType::value_type(s.first));
            if (fillValues) values.insert(values.end(), ValueSetType::value_type(s.second));
            ++numInserted;
        }
        return numInserted;
    }

    template <typename T>
    typename std::remove_const<typename T::value_type>::type* FlattenStlContainer(const T& container)
    {
        FlattenMap
            typedef typename std::remove_const<typename T::value_type>::type ElemType;
        ElemType* ret = new ElemType[container.size()];
        std::copy(container.begin(), container.end(), ret);
        return ret;
    }

    template<typename Iter>
    Iter partition(Iter first, Iter last, Iter pivot)
    {
        std::swap(*pivot, *last);
        Iter store = first;
        for (Iter i = first; i < last; ++i)
            if (*i < *pivot)
                std::swap(*(store++), *i);

        std::swap(*last, *store);
        return store;
    }

    template<typename Iter>
    Iter select(Iter first, Iter last, int k)
    {
        if (first >= last)
            return first;

        Iter pivot = first + (first - last) / 2;
        Iter pivotNew = partition(first, last, pivot);
        auto pivotDist = pivotNew - first + 1;
        if (pivotDist == k)
            return first + pivotDist;

        if (k < pivotDist)
            return select(first, pivotNew - 1, k);

        return select(pivotNew + 1, last, k - pivotDist);
    }

    template<typename Type>
    class MaxSet
    {
        uint QSize;
        std::set<Type> Queue;
    public:
        MaxSet(uint s) : QSize(s)
        {
        }

        void Insert(Type& item)
        {
            Queue.insert(item);

            if (Queue.size() > QSize)
                Queue.erase(Queue.begin());
        }

        std::set<Type> Get() const
        {
            return Queue;
        }

        template <typename Container>
        void Get(Container& c) const
        {
            c.insert(c.begin(), Queue.begin(), Queue.end());
        }
    private:
        MaxSet();
    };

    struct ProgressDisplay {

        const char* DisplayLineEmpty = "|.........|.........|.........|.........|.........|.........|.........|.........|.........|.........|";
        unsigned long long Limit, DonePC, DoneVal;

        std::string DisplayLine;
        std::string DisplayPCs;
        char FillChar;

        ProgressDisplay(unsigned long long limit) :
            Limit(limit), DonePC(0), DoneVal(0),
            DisplayPCs("0--------10--------20--------30--------40--------50--------60--------70--------80--------90--------100\n"),
            DisplayLine(DisplayLineEmpty),
            FillChar('+')
        {
            std::cout << DisplayPCs << DisplayLine;
        }

        void Update(unsigned long long done)
        {
            DoneVal = done;
            unsigned pc = unsigned(100 * (double(done) / Limit));
            if (DonePC != 100 && pc != DonePC)
            {
                std::cout << StringUtils::Rubout(DisplayLine);
                DisplayLine = DisplayLineEmpty;

                for (unsigned i = 0; i <= pc; ++i)
                    DisplayLine[i] = FillChar;

                std::string pcVal = std::to_string(100 * (float(done) / Limit));
                pcVal.erase(pcVal.begin() + 4, pcVal.end());
                DisplayLine += " : " + pcVal + "%";

                std::cout << DisplayLine;

                if (pc >= 100)
                    std::cout << " Done!\n";

                DonePC = pc;
            }
        }
    };

    template <typename Type1, typename Type2, typename Type3 >
    Type1 Clamp(Type1& v, Type2 sup = 0, Type3 inf = 0)
    {
        if (v < inf) { v = (Type1)(inf); return (Type1)inf; }
        if (v > sup) { v = (Type1)(sup); return (Type1)sup; }
        return v;
    }
    
    static std::default_random_engine Generator
#ifndef _DEBUG
        (unsigned(std::chrono::system_clock::now().time_since_epoch().count()))
#endif
        ;
    static std::uniform_real_distribution<double> distribution(0.0, 100.0);

    template<typename T>
    inline T Rand(T high = 1.0, T low = 0.0)
    {
        return (T)(low + (double)(rand()) / RAND_MAX * (high - low));
    }

    template<typename T>
    inline T URand(T high = 1.0, T low = 0.0)
    {
        return (T)(low + distribution(Generator)*(high - low) / 100.0);
    }


    inline uint* RandPerm(uint Length, bool notReally = false)
    {
        uint* ret = new uint[Length];
        for (uint i = 0; i < Length; ++i) ret[i] = i;
        if (!notReally) std::random_shuffle(ret, ret + Length);
        return ret;
    }

    template<typename T = double>
    struct NormalRandom
    {
        std::default_random_engine Generator;

        std::normal_distribution<T> Distribution;

        NormalRandom(T mean, T stddev) :
            Generator((unsigned)std::chrono::system_clock::now().time_since_epoch().count()),
            Distribution(mean, stddev)
        {
        }

        NormalRandom(T mean, T stddev, unsigned seed) : // debug: for generating non random sequence
            Generator(seed),
            Distribution(mean, stddev)
        {
        }

        inline T operator()() { return Distribution(Generator); }

        inline void Reset() { Distribution.reset(); }

        NormalRandom& GetNormalDist() { // for using 
            static NormalRandom NR;
            return NR;
        }
    };

    template<typename ArrayType> // convert first sizeof(ull)*8 bits to binary.
    inline unsigned long long BinToNum(const ArrayType& in, unsigned arrayLen)
    {
        unsigned long long p = 1, out = (in[0] == 0 ) ? 0 : 1;
        for (unsigned i = 1, p = 2; i < arrayLen && i < sizeof(p) * 8; ++i, p <<= 1)
            out += (in[i] == 0) ? 0 : p;
        
        return out;
    }

    template<typename T>
    uint WriteAsBytes(std::ofstream& strm, const T& t)
    {
        uint s = sizeof(t);
        strm.write(reinterpret_cast<const char*>(&t), s);
        return s;
    }

    template<typename T>
    uint ReadAsBytes(std::ifstream&  strm, T& t)
    {
        uint s = sizeof(t);
        strm.read(reinterpret_cast<char*>(&t), s);
        return s;
    }

    inline bool RoundedCompare(double d1, double d2)
    {
        return int(d1 + 0.5) == int(d2 + 0.5);
    }

    inline bool SameSign(double d1, double d2)
    {
        return (d1 <= 0.0 && d2 <= 0.0) || (d1 > 0.0 && d2 > 0.0);
    }

    template<typename T>
    struct is_numeric
    {
        static const bool value = std::is_arithmetic<T>::value && !std::is_same<T, bool>::value;
    };

    namespace Stability {

#ifdef _MSC_VER
#pragma optimize( "", off )
#endif

        template<typename Arr>
        inline double KahanSum(const Arr& arr, unsigned N)
        {
            double Sum = 0., Corr = 0., y = 0., t = 0.;
            for (unsigned i = 0; i < N; ++i)
            {
                y = arr[i] - Corr;
                t = Sum + y;
                Corr = (t - Sum) - y;
                Sum = t;
            }

            return Sum;
        }

        template<typename Arr>
        inline double KahanRMS(const Arr arr, unsigned N)
        {
            double Sum = 0., Corr = 0., y = 0., t = 0.;
            unsigned i = 0;
            for (unsigned i = 0; i < N; ++i, )
            {
                y = arr[i] * arr[i] - Corr;
                t = Sum + y;
                Corr = (t - Sum) - y;
                Sum = t;
            }

            return sqrt(Sum);
        }

        template<typename Arr1, typename Arr2>
        inline double KahanL0(const Arr1& arr1, const Arr2& arr2, unsigned N)
        {
            double Sum = 0., Corr = 0., y = 0., t = 0.;
            for (unsigned i = 0; i < N; ++i)
            {
                y = arr1[i] * arr2[i] - Corr;
                t = Sum + y;
                Corr = (t - Sum) - y;
                Sum = t;
            }

            return Sum;
        }
#ifdef _MSC_VER
#pragma optimize( "", on )
#endif
    }

    template<typename O, typename T>
    void PrintLinear(O& outStream, const T& lin, std::string message = "", std::string delim = ", ") // needs .size(), ,begin() and .end();
    {
        outStream << message << " [" << lin.size() << "]: ";
        if (delim == "\n")
            outStream << delim;
        if (lin.size())
            for (const auto& elem : lin)
                outStream << elem << delim;
        else
            outStream << "<Empty>";
    }

    template<typename O, typename Iter>
    void PrintLinear(O& outStream, Iter begin, size_t size, std::string message = "", std::string delim = ", ") // needs .size(), ,begin() and .end();
    {
        outStream << message << " [" << size << "]: ";
        if (delim == "\n")
            outStream << delim;

        if (size)
            for (unsigned i = 0; i < size; ++i)
                outStream << *begin++ << delim;
        else
            outStream << "<Empty>";
    }

    namespace SFINAE
    {

        template<bool Switch, unsigned long long  Value>struct  ULong { typedef double type; };
        template<unsigned long long Value> struct ULong<true, Value> { typedef unsigned long long type; };

        template<bool Switch, unsigned long long  Value>struct  UInt { typedef typename ULong < Value < (1ULL << (sizeof(long long) * 8 - 1)), Value >::type type; };
        template<unsigned long long Value> struct UInt<true, Value> { typedef unsigned int type; };


        template<bool Switch, unsigned long long  Value>struct  UShort { typedef typename UInt < Value < (1ULL << (sizeof(int) * 8)), Value>::type type; };
        template<unsigned long long Value> struct UShort<true, Value> { typedef unsigned short int type; };

        template<bool Switch, unsigned long long  Value>struct UChar { typedef typename UShort < Value < (1ULL << (sizeof(short) * 8)), Value>::type type; };
        template<int Value> struct UChar<true, Value> { typedef unsigned char type; };

        template<unsigned long long Value>
        struct SmallestUnsignedType
        {
            typedef typename UChar < Value < (1ULL << (sizeof(char) * 8)), Value>::type Type;
        };

        /* Test:

        cout << sizeof( SFINAE::SmallestUnsignedType<254>::Type ) << endl;
        cout << sizeof( SFINAE::SmallestUnsignedType<65534>::Type )<< endl;
        cout << sizeof( SFINAE::SmallestUnsignedType<4294967291>::Type )<< endl;
        cout << sizeof( SFINAE::SmallestUnsignedType<18446744073709551614>::Type );

        prints
        1
        2
        4
        8
        */
    }
}

#endif
