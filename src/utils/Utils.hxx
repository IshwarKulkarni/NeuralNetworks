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

#ifndef __UTILS_INCLUDED__
#define __UTILS_INCLUDED__

#include <algorithm>
#include <cctype>
#include <chrono>
#include <fstream>
#include <functional>
#include <sstream>
#include <random>
#include <ctime>
#include <cstring>
#include <iostream>
#include <fstream>
#include <set>

#define ARRAY_LENGTH(arr)    (arr == 0 ? 0 : sizeof(arr)/sizeof(arr[0]) )
#define ARRAYEND(arr) (arr + ARRAY_LENGTH(arr))

#define MAX(a,b) (((a) > (b)) ? (a) : (b))
#define MIN(a,b) (((a) < (b)) ? (a) : (b))

#ifdef _MSC_VER 
#define strktok(a,b) strtok_s((a),(b))
#endif

namespace Utils
{
    template<typename T> T Identity(T t) { return t; }
    template<typename T, typename U> T Identity(T t, U u) { return t = u; }

    inline bool RoundedCompare(double d1, double d2)
    {
        return int(d1 + 0.5) == int(d2 + 0.5);
    }

    inline bool SameSign(double d1, double d2)
    {
        return (d1 <= 0.0 && d2 <= 0.0) || (d1 > 0.0 && d2 > 0.0);
    }

    template<typename T>
    inline T iDivUp(T a, T b)
    {
        return (a % b ? ((a / b) + 1) : (a / b));
    }

    static std::default_random_engine Generator
#ifndef _DEBUG
        (unsigned(std::chrono::system_clock::now().time_since_epoch().count()))
#endif
        ;
    static std::uniform_real_distribution<double> distribution(0.0, 100.0);

    template<typename T>
    inline T URand(T high = 1.0, T low = 0.0)
    {
        return (T)(low + distribution(Generator)*(high - low) / 100.0);
    }
    template<typename Node> // Node should support < operator
    class TopN  
    {
        std::set<Node> Set;

    public:
        typedef std::set<Node> BaseMapType;
        const size_t N;
        TopN(size_t n) : N(n) {} 

        void insert(const Node & val)
        {
            Set.insert(val);
            if (Set.size() > N) 
                Set.erase(begin());
        }
        typename BaseMapType::iterator begin() { return Set.begin(); }
        typename BaseMapType::iterator   end() { return Set.end(); }
        void clear()  { return Set.clear(); }
        size_t size() { return Set.size(); }
    private:
        TopN(TopN&);
    };
    inline const char *Rubout(int i)
    {
        if (i < 0) i = -10 * i; // Add a backspace for the negative sign.
        if (i < 10) return "\b";
        if (i < 100) return "\b\b";
        if (i < 1000) return "\b\b\b";
        return "\b\b\b\b";
    }

    inline double TimeSince(std::chrono::high_resolution_clock::time_point& since)
    {
        auto now = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = std::chrono::duration_cast<std::chrono::duration<double>>(now - 
            since);
        return duration.count();
    }

    template<typename T>
    void WriteRawBytes(std::ostream& o, const T& t)
    {
        o.write(static_cast<const char*>(&t), sizeof(t));
    }
	
	template<typename T>
	void PrintLinear(std::ostream& outStream, const T& lin, std::string message = "", std::string delim = ", ") // needs .size(), ,begin() and .end();
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

	template<typename Iter>
	void PrintLinear(std::ostream& outStream, Iter begin, size_t size, std::string message = "", std::string delim = ", ") // needs .size(), ,begin() and .end();
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

}

namespace PPMIO
{
    extern void Write(std::string name, size_t width, size_t height, size_t numComps, const unsigned char* frames, bool interleaved = true);
    extern unsigned char* Read(std::string name, size_t& width, size_t& height, size_t& components);
}

namespace StringUtils
{
    inline bool HasWSpace(const std::string& in)
    {
        for (auto c : in)
            if (iswspace(c))
                return true;
        return false;
    }

    inline std::string& LTrim(std::string &s) {
        s.erase(s.begin(), std::find_if(s.begin(), s.end(), std::not1(std::ptr_fun<int, int>(std::isspace))));
        return s;
    }

    inline std::string& RTrim(std::string& s, std::string trimAfter=" ") {
        auto found = s.find_first_of(trimAfter);
        if (found != std::string::npos)
            s.erase(s.begin() + found, s.end());
        return s;
    }

    inline std::string RTrim(const std::string& ipStr) {
        std::string s(ipStr);
        s.erase(std::find_if(s.rbegin(), s.rend(), std::not1(std::ptr_fun<int, int>(std::isspace))).base(), s.end());
        return s;
    }

    inline std::string StrTrim(const std::string &s) {
        auto r = RTrim(s);
        return LTrim(r);
    }

    inline bool beginsWith(std::string superString, std::string subString)
    {
        for (unsigned i = 0; i < subString.length(); ++i)
            if (superString[i] != subString[i])
                return false;
        return true;
    }

    inline std::vector<std::string> Split(std::string str, std::string delim, bool trim)
    {
        char* tok = strtok(&(str[0]), delim.c_str()); // hell with constness
        std::vector<std::string> tokens(1, tok);

        while ((tok = strtok(0, delim.c_str())) != nullptr) tokens.push_back(tok);

        for (auto& t : tokens) StrTrim(t);
        return tokens;
    }

    template< typename ValType>
    inline ValType StringToType(const std::string& str)
    {
        ValType v;
        std::stringstream(str) >> v;
        return v;
    }

    inline std::string::const_iterator FindEOL(const std::string& s)
    {
        return std::find_if(s.begin(), s.end(),
            [&](char c) { return c == '\n' || c == '\r'; });

    }

    inline std::string ToUpper(const std::string& cstr)
    {
        std::string str(cstr);
        std::transform(str.begin(), str.end(), str.begin(), toupper);
        return str;
    }

    inline std::string ToLower(const std::string& cstr)
    {
        std::string str(cstr);
        std::transform(str.begin(), str.end(), str.begin(), tolower);
        return str;
    }

    inline void ToUpper(std::string& str)
    {
        std::transform(str.begin(), str.end(), str.begin(), toupper);
    }

    inline void ToLower(std::string& str)
    {
        std::transform(str.begin(), str.end(), str.begin(), tolower);
    }


    inline bool endsWith(const std::string& str, const std::string& ends, bool matchcase = false)
    {
        if (str.length() < ends.length()) return false;
        return matchcase ?
            std::string(str.end() - ends.length(), str.end()) == ends :
            ToUpper(std::string(str.end() - ends.length(), str.end())) == ToUpper(std::string(ends));
    }

    inline void establishEndsWith(std::string& str, const std::string& ends, bool matchcase = false)
    {
        if (!endsWith(str, ends, matchcase))
            str += ends;
    }

    inline std::string establishEndsWith(const char* cstr, const std::string& ends, bool matchcase = false)
    {
        if (!cstr || ends.length() == 0)
            return std::string("");
        std::string str(cstr);
        if (!endsWith(str, ends, matchcase))
            str += ends;

        return str;
    }

}

namespace Logging
{
    inline const char* TimeNowStringFull(time_t now = -1, const char* format = "%a, %d-%m-%Y, %H:%M:%S ")
    {
        if (now == -1)
            now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());

        struct tm timeinfo;
#ifdef _WIN32
        localtime_s(&timeinfo, &now);
#else
        timeinfo = *localtime(&now);
#endif
        static const unsigned TimeStampLen = 30;
        static char stamp[TimeStampLen];

        strftime(stamp, TimeStampLen, format, &timeinfo);

        return stamp;
    }

    class Logger
    {
    public:

        struct FlushType { } flush;

        static Logger& Instance() { static Logger inst;  return inst; }

        operator std::ofstream&() { return LogFile; }

        static inline std::string LogFileName() { return "LogFile"; }

        template <typename Type>
        inline Logger& operator<<(const Type& t)
        {
            try { Instance().LogFile << t;  }
            catch (std::exception e) { Instance().LogFile << "Catastrophic error: " << e.what() << std::endl; throw e; }
            return Instance();
        }

        inline Logger& operator<<(const FlushType& t) { LogFile.flush(); return Instance(); }

    private:

        std::ofstream LogFile;

        Logger()
        {
            LogFile.open(LogFileName() + 
#if 0
                TimeNowStringFull(-1, "_%d-%m-%y_%H%M%S") 
#endif
                + ".log", std::ofstream::out);
            LogFile << "Session Logging Started At : " << TimeNowStringFull() << "\nSession Type: " <<
#ifdef _DEBUG
                " Debug \n"
#elif defined NDEBUG
                " Release\n"
#else
                " Unknown, This is likely a bad build"
#endif
                ; std::cerr.rdbuf(LogFile.rdbuf());
        };

        ~Logger()
        {
            LogFile.open(LogFileName(), std::ofstream::out | std::ofstream::app);
            LogFile << "Session Logging Ended At : " << TimeNowStringFull() << std::endl;
            LogFile.flush();
        };

        Logger& operator=(const Logger&);
        Logger(const Logger&);
    };

#define LogPrintf(...)  fprintf(stderr, __VA_ARGS__); fprintf(stdout, __VA_ARGS__)
#define LOG_LOC()     __FILE__ << ":" << __LINE__ << " in " << __FUNCTION__

    static Logger& Log = Logger::Instance();

    class Timer
    {
    public:
        Timer(bool LogOnStart, bool logAtStop = true) :
            Span(0),
            Start(std::chrono::high_resolution_clock::now()),
            LastCheck(std::chrono::high_resolution_clock::now()),
            TimerName(""),
            Stopped(false),            
            LogAtStop(logAtStop)
        {
            if (LogOnStart)
                Log << "A timer is started at " << TimeNowStringFull() << "\n";
        }

        Timer(const char* name = "", bool logAtStop = true) :
            Span(0),
            Start(std::chrono::high_resolution_clock::now()),
            LastCheck(std::chrono::high_resolution_clock::now()),
            TimerName(name),
            Stopped(false),            
            LogAtStop(logAtStop)
        {
        }

        ~Timer()
        {
            if (!Stopped)
            {
                End = std::chrono::high_resolution_clock::now();
                Span = std::chrono::duration_cast<std::chrono::duration<double>>(End - Start);
                if (LogAtStop && TimerName.length())
                    Log << "\n\"" << TimerName << "\"" << " Ended at " << TimeNowStringFull() << ". Duration was " << Span.count() << " seconds.\n";
            }
        }

        double TimeFromLastCheck()
        {
            std::chrono::duration<double> span = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::high_resolution_clock::now() - LastCheck);
            LastCheck = std::chrono::high_resolution_clock::now();
            return span.count();
        }

        double Stop()
        {
            if (Stopped)
                return Span.count();
            End = std::chrono::high_resolution_clock::now();
            Span += std::chrono::duration_cast<std::chrono::duration<double>>(End - Start);
            if (LogAtStop)
                Log << "\"" << TimerName << "\"" << " Stopped at " << TimeNowStringFull() << ". Duration was " << Span.count() << " seconds.\n";
            Stopped = true;
            return Span.count();
        }

        void Restart()
        {
            End = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double>  pause = std::chrono::duration_cast<std::chrono::duration<double>>(End - Start);
            if (LogAtStop)
                Log << "\"" << TimerName << "\"" << " Restarted at " << TimeNowStringFull() << ". Duration from start was: " << pause.count() << " seconds.\n";

            Start = std::chrono::high_resolution_clock::now();
        }

    private:

        std::chrono::duration<double> Span;
        std::chrono::high_resolution_clock::time_point Start, End/*Uninitilized*/, LastCheck;
        std::string TimerName;
        bool Stopped, LogAtStop;

    };
}
#endif