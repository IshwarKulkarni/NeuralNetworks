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

#define ARRAY_LENGTH(arr)    (arr == 0 ? 0 : sizeof(arr)/sizeof(arr[0]) )
#define ARRAYEND(arr) (arr + ARRAY_LENGTH(arr))

#define MAX(a,b) (((a) > (b)) ? (a) : (b))
#define MIN(a,b) (((a) < (b)) ? (a) : (b))

#ifdef _MSC_VER 
#define strktok(a,b) strtok_s((a),(b))
#endif

namespace Utils
{
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
        return ( a % b ? ((a / b) + 1) : (a / b) );
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
}

namespace Logging
{
    inline const char* TimeNowStringFull(time_t now = -1)
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

        strftime(stamp, TimeStampLen, "%a, %d-%m-%Y, %H:%M:%S ", &timeinfo);

        return stamp;
    }

    class Logger
    {
    public:

        static Logger& Instance() { static Logger inst;  return inst; }

        operator std::ofstream&() { return LogFile; }

        static inline const char* LogFileName() { return "LogFile.txt"; }

        template <typename Type>
        inline Logger& operator<<(const Type& t)
        {
            try
            {
                Instance().LogFile << (t);
            }
            catch (std::exception e)
            {
                Instance().LogFile << "Catastrophic error: " << e.what() << std::endl;
                throw e;
            }
            return Instance();
        }

    private:

        std::ofstream LogFile;

        Logger()
        {
            LogFile.open(LogFileName(), std::ofstream::out | std::ofstream::app);
            LogFile << "Session Logging Started At : " << TimeNowStringFull() << " ********" << std::endl;
            std::cerr.rdbuf(LogFile.rdbuf());
        };

        ~Logger()
        {
            LogFile.open(LogFileName(), std::ofstream::out | std::ofstream::app);
            LogFile << "Session Logging Ended At : " << TimeNowStringFull() << " ********" << std::endl;
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
                    Log << "\"" << TimerName << "\"" << " Ended at " << TimeNowStringFull() << ". Duration was " << Span.count() << " seconds.\n";
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