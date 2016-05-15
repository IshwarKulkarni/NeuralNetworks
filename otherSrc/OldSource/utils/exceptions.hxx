#ifndef EXCEPTION_INCLUDED
#define EXCEPTION_INCLUDED

#include <exception>
#include <iostream>
#include <type_traits>

#define ENABLE_EXCEPTIONS 1

#pragma warning(disable : 4996)

#define WHAT(s) virtual const char* what() const throw() { return  s ; }
#define EXCEPTION_TYPE(Name, SuperType) \
class Name : public SuperType {            \
    public:                                \
    Name(int err): SuperType(err){}        \
};

#define EXCEPTION_TYPE_MSG(Name, SuperType, ErrorMsg) \
class Name : public SuperType {            \
    public:                                \
    Name(int err): SuperType(err){}        \
};

class ISystemExeption: public std::exception
{
    public: 
        ISystemExeption(int err): std::exception(){} 
        virtual const char *what() const { return "Generic Error Type"; }
}; 

EXCEPTION_TYPE(InvalidCodeException, ISystemExeption)
EXCEPTION_TYPE(InvalidOptionException, InvalidCodeException)
EXCEPTION_TYPE(UninitilizedException, InvalidCodeException)
EXCEPTION_TYPE(LogicalOrMath, InvalidCodeException)

EXCEPTION_TYPE(IOException, ISystemExeption)

EXCEPTION_TYPE(FileIOException, IOException)
EXCEPTION_TYPE(FileOpenException, FileIOException)
EXCEPTION_TYPE(BadFileNameOrType, FileIOException)
EXCEPTION_TYPE(NativeFormatException, FileIOException)
EXCEPTION_TYPE(UnsupportedFileFormat, FileIOException)
EXCEPTION_TYPE(BadFileStrmException, FileIOException)
EXCEPTION_TYPE(BadAssumptionException, FileIOException)
EXCEPTION_TYPE(UnopenFileStream, FileIOException)

EXCEPTION_TYPE(InvalidStringFormat, FileIOException)

EXCEPTION_TYPE(DataException, ISystemExeption)
EXCEPTION_TYPE(InvalidFileFormatException, DataException)
EXCEPTION_TYPE(InvalidArgumentException, DataException)
EXCEPTION_TYPE(InvalidArgumentTypeException, DataException)


EXCEPTION_TYPE(DimensionException, DataException)
EXCEPTION_TYPE(WrongSizeException, DataException)

EXCEPTION_TYPE(StringParseException, DataException)

EXCEPTION_TYPE(InvlidImageFormatException, InvalidFileFormatException)
EXCEPTION_TYPE(FileParseException, InvalidFileFormatException)
EXCEPTION_TYPE(UnexpectedLiteralException, FileParseException)


#if ENABLE_EXCEPTIONS
static char __MessageBuf[1024];
#define THROW_IF( condition, exceptionType, ... ){  \
    auto __cx = (condition);                        \
    if( __cx ) {                                    \
        exceptionType  e(__cx);                     \
        sprintf( __MessageBuf, __VA_ARGS__);        \
        std::cerr                                   \
            << ">>> " << __MessageBuf               \
            << " in " << __FUNCTION__               \
            << " at " << __FILE__                   \
            << ", Line: "  << __LINE__              \
            << ".\n>>  Error: "                     \
            << std::boolalpha << __cx               \
            << " returned on evaluation of "        \
            << #condition << std::endl;             \
        throw e;                                    \
       }                                            \
}


#define WARN_IF( condition, exceptionType, ... ){  \
    auto __cx = (condition);                        \
    if( __cx ) {                                    \
        sprintf( __MessageBuf, __VA_ARGS__);        \
        std::cerr                                   \
            << ">>> " << __MessageBuf               \
            << " in " << __FUNCTION__               \
            << " at " << __FILE__                   \
            << ", Line: "  << __LINE__              \
            << ".\n>>  Warning: "                   \
            << std::boolalpha << __cx               \
            << " returned on evaluation of "        \
            << #condition << std::endl;             \
       }                                            \
}


#define THROW(exceptionType, ... ) {            \
    exceptionType  e(true);                     \
    sprintf( __MessageBuf, __VA_ARGS__);        \
    std::cerr                                   \
        << ">>> " << __MessageBuf               \
        << " in " << __FUNCTION__               \
        << " at " << __FILE__                   \
        << ", Line: "  << __LINE__ << "\n";     \
        std::cerr.flush();                      \
    throw e;                                    \
}

#else 
#define THROW_IF( condition, exceptionType, ... )

#endif

#undef WHATTT
#endif