
#include <istream>
#include <utils/StringUtils.hxx>
#include <utils/Exceptions.hxx>
#include "vectors.hxx"

template<typename T>
std::istream& operator >> (std::istream& strm, Vec3<T>& v)
{
    char sep; 

    strm >> std::skipws >> sep >> v.x >> sep >> v.y >> sep >> v.z >> sep; 

    THROW_IF(!strm.good(), FileParseException, "File Parse error above", 
        PrintLocation(strm, Logging::Log));

    return strm;
}

template<typename T>
std::istream& operator >> (std::istream& strm, Interval<T>& v)
{
    char sep;
    strm >> std::skipws >> sep >> v.min >> sep >> v.max >> sep;

    THROW_IF(!strm.good(), FileParseException, "File Parse error above",
        PrintLocation(strm, Logging::Log));
    return strm;   
}

template <typename T>
inline std::ostream& operator<<(std::ostream& out, const Vec<T>& v)
{
    out << "(" << v.x << ", " << v.y << ", " << v.z << ")";
    return out;
}

template <typename T>
inline std::ostream& operator<<(std::ostream& out, const Interval<T>& v)
{
    out << "(" << v.min << ", " << v.max << ")";
    return out;
}