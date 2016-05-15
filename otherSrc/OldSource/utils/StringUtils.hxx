#ifndef _STRING_UTILS_
#define _STRING_UTILS_

#include <algorithm>
#include <cctype>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include "utils/Logging.hxx"
#include "utils/Exceptions.hxx"

#define MaxLineLength 512

namespace StringUtils{

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

    template< typename ValType> 
    inline ValType StringToType(const std::string& str)
    {
        ValType v;
        stringstream(str) >> v;
        return v;
    }

    // From http://stackoverflow.com/a/217605
    inline std::string& LTrim(std::string &s) {
        s.erase(s.begin(), std::find_if(s.begin(), s.end(), std::not1(std::ptr_fun<int, int>(std::isspace))));
        return s;
    }

    inline std::string& RTrim(std::string &s) {
        s.erase(std::find_if(s.rbegin(), s.rend(), std::not1(std::ptr_fun<int, int>(std::isspace))).base(), s.end());
        return s;
    }

    inline std::string& RTrim(std::string &s, std::string& trimAfter) {
        auto found = s.find_first_of(trimAfter);
        if (found != std::string::npos)
            s.erase(s.begin() + found, s.end());
        return s;
    }

    inline std::string& StrTrim(std::string &s) {
        return LTrim(RTrim(s));
    }

    inline std::string LTrim(const std::string& ipStr) {
        std::string s(ipStr);
        s.erase(s.begin(), std::find_if(s.begin(), s.end(), std::not1(std::ptr_fun<int, int>(std::isspace))));
        return s;
    }

    inline std::string RTrim(const std::string& ipStr) {
        std::string s(ipStr);
        s.erase(std::find_if(s.rbegin(), s.rend(), std::not1(std::ptr_fun<int, int>(std::isspace))).base(), s.end());
        return s;
    }

    inline std::string StrTrim(const std::string &s) {
        return LTrim(RTrim(s));
    }

    inline std::vector<std::string> Split(std::string str, std::string delim, bool trim)
    {
        char* tok = strtok(&(str[0]), delim.c_str()); // hell with constness
        std::vector<std::string> tokens(1, tok);

        while (tok = strtok(0, delim.c_str())) tokens.push_back(tok);

        for (auto& t : tokens) StrTrim(t);
        return tokens;
    }

    inline std::string Replace1(std::string inString, std::string orig, std::string replace)
    {
        auto f = inString.find(orig);
        if (f != std::string::npos)
            inString.replace(f, replace.length(), replace.c_str());
        
        return inString;
    }

    inline void Replace1Inplace(std::string& inString, std::string orig, std::string replace)
    {
        auto f = inString.find(orig);
        if (f != std::string::npos)
            inString.replace(f, replace.length(), replace.c_str());
    }


    static const std::string Empty;
    struct CaseInsensitiveMatch{
        const std::string& First;
        inline CaseInsensitiveMatch() : First(Empty) {}
        inline CaseInsensitiveMatch(const std::string& s) : First(s) {}
        inline bool operator()(const std::string& s) { return ToUpper(s) == ToUpper(First); }
        inline bool operator()(const std::string& s1, const std::string s2) const
        {
            return ToUpper(s1) == ToUpper(s2);
        }
        inline bool operator()(const std::pair<const std::string&, const std::string&> p)
        {
            return ToUpper(p.first) == ToUpper(First);
        }
    };

    struct CaseInsensitivePredicate{
        inline bool operator()(const std::string& s1, const std::string s2) const  { return ToUpper(s1) < ToUpper(s2); }
        inline bool operator()(const char&s1, const char& s2) { return toupper(s1) == toupper(s2); }
    };

    inline bool beginsWith(std::string superString, std::string subString)
    {
        for (unsigned i = 0; i < subString.length(); ++i)
            if (superString[i] != subString[i])
                return false;
        return true;
    }

    inline bool beginsWithIgnoreCase(std::string& superString, std::string& substring)
    {
        for (unsigned i = 0; i < substring.length(); ++i)
            if (toupper(superString[i]) != toupper(substring[i]))
                return false;
        return true;
    }

    inline bool HasWSpace(const std::string& in)
    {
        for (auto c : in)
            if (iswspace(c))
                return true;
        return false;
    }

    struct CaseInsensitiveHasher
    {
        inline size_t operator()(const std::string& s){
            const size_t _FNV_prime = 16777619U; // fibbed from the MSVC hashing implem
            size_t _Val = 2166136261U;
            std::string sU = ToUpper(s);
            for (auto& c : sU)
                _Val ^= (size_t)(c), _Val *= _FNV_prime;
            return _Val;
        }
    };

    template<typename T> using StringMap = std::map < std::string, T >;

    template<typename T> using StringUnorderedMap = std::unordered_map<std::string, T>;

    template<typename T> using StringCaseInsensitiveMap = std::map<std::string, T, CaseInsensitivePredicate>;

    template<typename T> using StringCaseInsensitiveUnorderedMap = std::unordered_map<std::string, T, CaseInsensitiveHasher, CaseInsensitiveMatch>;

    struct Months
    {
        static const std::vector<std::string>& Month3L()
        {
            static const std::vector<std::string> months = {
                "Jan",
                "Feb",
                "Mar",
                "Apr",
                "May",
                "Jun",
                "Jul",
                "Aug",
                "Sep",
                "Oct",
                "Nov",
                "Dec"
            };
            return  months;
        }

        static const std::vector<std::string>& Month()
        {
            static const std::vector<std::string> months = {
                "January",
                "February",
                "March",
                "April",
                "May",
                "June",
                "July",
                "August",
                "September",
                "October",
                "November",
                "December"
            };
            return  months;
        }
        static unsigned char DaysInMonth(unsigned Month, bool leap = false)
        {
            unsigned char days[12] = { 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31 };
            if (Month == 1){ if (leap) return 29; }
            if (Month < 12) return days[Month];
            return -1;
        }

        static inline unsigned MonthNameToIdx(const char* month) // behaviour undefined on non-month strings.
        {
            switch (*month){
            case 'f': case 'F': return 1;
            case 's': case 'S': return 8;
            case 'o': case 'O': return 9;
            case 'n': case 'N': return 10;
            case 'd': case 'D': return 11;

            case 'm': case 'M':
                switch (month[2])
                {
                case 'r': case 'R': return 2;
                case 'y': case 'Y': return 4;
                default: return unsigned(-1);
                }

            case 'a': case 'A':
                switch (month[1])
                {
                case 'p': case 'P': return 3;
                case 'u': case 'U': return 7;
                default: return unsigned(-1);
                }

            case 'J': case 'j':
                if (month[0] != 'a' && month[0] != 'A')
                    switch (month[2])
                {
                    case 'n': case 'N': return 5;
                    case 'l': case 'L': return 6;
                    default: return unsigned(-1);
                } // end n,l

                else  return 0;
            default: return unsigned(-1);
            }
        }

        static bool IsMonthString3(std::string mon)
        {

            static const std::unordered_set<std::string> months = {
                "January",
                "February",
                "March",
                "April",
                "May",
                "June",
                "July",
                "August",
                "September",
                "October",
                "November",
                "December"
            };
            return  months.find(mon) == months.end();;
        }

        static bool IsMonthString(std::string mon)
        {
            static const std::unordered_set<std::string> months = {
                "Jan",
                "Feb",
                "Mar",
                "Apr",
                "May",
                "Jun",
                "Jul",
                "Aug",
                "Sep",
                "Oct",
                "Dec"
            };
            return  months.find(mon) == months.end();
        }
    };

    inline unsigned long long PrintLocation(std::istream& inStrm, std::ostream& outStrm = std::cout)
    {
        std::ios  state(NULL);
        state.copyfmt(inStrm);
        inStrm.clear();
        auto loc = inStrm.tellg();
        inStrm.seekg(0, std::ios::beg);

        static char line[MaxLineLength];
        unsigned lineNum = 0, charsRead = 0, totCharsRead = 0;
        do
        {
            inStrm.getline(line, MaxLineLength);
            charsRead = (unsigned)inStrm.gcount();
            THROW_IF(charsRead >= MaxLineLength, FileParseException, "Line too long for parsing.");

            if (inStrm.tellg() > loc)
                break;

            totCharsRead += charsRead;
            ++lineNum;
        } while (true);


        int col = (int)loc - totCharsRead - 3;
        if (col < 0)
            col = 0;

        outStrm
            << "Line " << lineNum << ", Col " << col << " : "
            << "\n\t" << StrTrim(line)
            << "\n\t" << std::string(col, ' ') << "^^^\n";

        inStrm.copyfmt(state);
        inStrm.seekg(loc, std::ios_base::beg);

        unsigned long long ret = lineNum; // because I can, that's why
        ret <<= 32;
        ret |= col;

        return ret;
    }

    template<typename T>
    inline std::string Concatinate(std::string str, const T& t)
    {
        std::ostringstream ss;
        ss << str << t;
        std::string out(ss.str());
        return out;
    }

    template<typename T>
    inline std::string ToString(const T& t)
    {
        std::ostringstream ss;
        ss << t;
        std::string out(ss.str());
        return out;
    }

    template<typename T>
    inline std::string Rubout(const T& in)
    {
        size_t numChars = ToString(in).length();
        return std::string(numChars, '\b') + std::string(numChars, ' ') + std::string(numChars, '\b');
    }

    template<>
    inline std::string Rubout<char>(const char& in)
    {
        if(in == '\n')
            return "\033";
        return "\b";
    }

    inline void StripAfter(std::string& strIn, std::string separator)
    {
        strIn.erase(strIn.find(separator), std::string::npos);
    }

    inline std::string StripAfter(const char* in, std::string separator)
    {
        std::string strIn(in);
        strIn.erase(strIn.find(separator), std::string::npos);
        return strIn;
    }

    inline bool Contains(std::string strIn, std::string substring, bool matchCase = true)
    {
        if (matchCase)
            return strIn.find(substring) != std::string::npos;

        ToUpper(substring), ToUpper(strIn);
        return strIn.find(substring) != std::string::npos;
    }

    inline void StripPrefix(std::string& strIn, std::string& prefix)
    {

    }

    // Something like the ideal type_info::name
    template<typename T>
    inline const char*  TypeToName() { return "Unknown Type"; }

    template<>
    inline const char*  TypeToName<double>() { return "Double"; }

    template<>
    inline const char*  TypeToName<float>() { return "Float"; }

    template<>
    inline const char*  TypeToName<char>() { return "Char"; }

    template<>
    inline const char*  TypeToName<short>() { return "Short"; }

    template<>
    inline const char*  TypeToName<int>() { return "Int"; }

} // end StringUtils
#endif

