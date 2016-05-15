#ifndef LOGGING_INCLUDED
#define LOGGING_INCLUDED

#include <ctime>
#include <chrono>
#include <string>
#include <stdio.h>
#include <mutex>
#include <iostream> 
#include <fstream>

#include <ctime>
#include <ratio>
#include <chrono>
#include <omp.h>

#ifdef _WIN32
#include <io.h>

#endif

//#define APPEND_TO_LOG

#define LogEndl  "\n" 
#define NOW (std::chrono::system_clock::to_time_t (std::chrono::system_clock::now()) )

inline const char* GetMonthName(unsigned m)
{
    static const char MonthName[][4] = { "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul",
        "Aug", "Sep", "Oct", "Nov", "Dec" };

    if (m < 11)
        return MonthName[11];

    return "";
}

inline const char* GetWeekDay(unsigned m)
{
    static const char WeekDay[][4] = { "Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat" };

    if (m < 6)
        return WeekDay[6];
    return "";
}

inline const char* TimeNowStringFull(std::time_t now = -1)
{

    if (now == -1)
        now = NOW;

    struct tm timeinfo;
#ifdef _WIN32
    localtime_s(&timeinfo, &now);
#elif
    timeinfo = *localtime(&now);
#endif

    static const unsigned TimeStampLen = 30;
    static char stamp[TimeStampLen];

    strftime(stamp, TimeStampLen, "%a, %d-%m-%Y, %H:%M:%S ", &timeinfo);

    return stamp;
}

inline const char* TimeNowString(std::time_t now = -1)
{

    if (now == -1)
        now = NOW;

    struct tm timeinfo;
#ifdef _WIN32
    localtime_s(&timeinfo, &now);
#elif
    timeinfo = *localtime(&now);
#endif

    static const unsigned TimeStampLen = 30;
    static char stamp[TimeStampLen];


    static std::mutex TimeStampAcess;

    std::unique_lock<std::mutex> lock(TimeStampAcess);
    strftime(stamp, TimeStampLen, "%H:%M:%S ", &timeinfo);

    return stamp;
}

namespace Logging
{
    class LogFlush {};
    class Logger
    {
        bool Gate;
    public:

        inline bool Toggle()
        {
            bool old = Gate;
            Gate = !Gate;
            return old;
        }

        template <typename Type>
        inline Logger& operator<<(const Type& t)
        {
            if (!Gate) return Instance();
            auto& now = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double>  timeSinceLastLog = 
                std::chrono::duration_cast<std::chrono::duration<double>>(
                    now - LastLogTime);
            
            try
            {
                if (IntervealLogging && timeSinceLastLog.count() > 1.0)
                {
                    LastLogTime = now;
                    Instance().LogFile << "\n**" << TimeNowString() << LogEndl;
                }

                Instance().LogFile << (t);
                if (TeeStream)
                    (*TeeStream) << t;
                LogFile.flush();
            }
            catch (std::exception e)
            {
                Instance().LogFile << "Catastrophic error: " << e.what() << LogEndl;
                flush();
                LogFile.close();
                throw e;
            }
            return Instance();
        }

        inline Logger& operator<<(const LogFlush&)
        {
            LogFile.flush();
            return Instance();
        }

        operator std::ofstream&()
        {
            return LogFile;
        }

        static Logger& Instance()
        {
            static Logger instance;
            return instance;
        }

        void flush()
        {
            LogFile.flush();
        }

        void Now()
        {
            Instance().LogFile << TimeNowStringFull() << LogEndl;
        }

        void Tee(std::ostream& teeStream)
        {
            TeeStream = &teeStream;
        }

        void StopTee()
        {
            TeeStream = 0;
        }

        bool ResetIntervealLogging(bool l)
        {
            std::swap(l, IntervealLogging);
            return l;
        }

    private:

        static inline const char* LogFileName() { return "LogFile.txt"; }
        std::ofstream LogFile;
        std::ostream*  TeeStream; // would like to make it a vector to add multiple tees.
        std::chrono::high_resolution_clock::time_point LastLogTime;
        bool IntervealLogging;
        Logger()
        {
#ifdef APPEND_TO_LOG
            LogFile.open(LogFileName(), std::ofstream::out | std::ofstream::app);
#else
            LogFile.open(LogFileName(), std::ofstream::out);
#endif
            LogFile << LogEndl << "******** Session Logging Started At : " << TimeNowStringFull() << " ********\n";
            std::cerr.rdbuf(LogFile.rdbuf());
            Gate = true;
            LastLogTime = std::chrono::high_resolution_clock::now();
        };

        ~Logger()
        {
            LogFile << LogEndl << "******** Session Logging Ended At : " << TimeNowStringFull() << " ********\n";
            LogFile.flush();
            LogFile.close();
        }

        Logger(const Logger&);
        void operator=(const Logger&);
    };
    static Logger& Log = Logger::Instance();

#define LogVarP( a ) Logging::Log << "\n>> "  << #a << " in " << __FUNCTION__ << " at " << __FILE__ << ":"  << __LINE__ << " : " << a  << LogEndl;
#define LogVar( a )  Logging::Log << " " << #a << " = " << a  << LogEndl;
#define LogVarH( a ) Logging::Log << #a << " = 0x" << std::hex << a << std::dec<< "\n";
#define LogPrintf(...)  fprintf(stderr, __VA_ARGS__); fprintf(stdout, __VA_ARGS__)
#define LOG_LOC()     __FILE__ << ":" << __LINE__ << " in " << __FUNCTION__

    class Timer
    {
    public:
        Timer(bool LogOnStart, bool logAtStop=true) :
            Span(0), 
            Start(std::chrono::high_resolution_clock::now()),
            TimerName(""),
            Stopped(false), 
            LastCheck(std::chrono::high_resolution_clock::now()),
            LogAtStop(logAtStop)
        {
            if (LogOnStart)
                Log << "A timer is started at " << TimeNowString() << "\n";
        }

        Timer(const char* name = "", bool logAtStop = true) :
            Span(0), 
            Start(std::chrono::high_resolution_clock::now()),
            TimerName(name),
            Stopped(false),
            LastCheck(std::chrono::high_resolution_clock::now()),
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
                    Log <<"\"" << TimerName << "\"" << " Ended at " << TimeNowString() << ". Duration was " << Span.count() << " seconds.\n";
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
            if(LogAtStop)
                Log <<  "\"" << TimerName << "\"" << " Stopped at " << TimeNowString() << ". Duration was " << Span.count() << " seconds.\n";
            Stopped = true;
            return Span.count();
        }

        void Restart()
        {
            End = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double>  pause = std::chrono::duration_cast<std::chrono::duration<double>>(End - Start);
            if (LogAtStop)
                Log <<"\"" << TimerName << "\"" << " Restarted at " << TimeNowString() << ". Duration from start was: " << pause.count() << " seconds.\n";
            
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
