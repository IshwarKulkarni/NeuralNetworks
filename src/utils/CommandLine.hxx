#ifndef _COMMANDLINE_INCLUDED_
#define _COMMANDLINE_INCLUDED_

#include <fstream>
#include <vector>
#include <sstream>
#include <iostream>
#include <map>
#include <tuple>

#include "Utils.hxx"

// This file started out as a function to parse command line. Now it's a way of retrieving name value pairs form different sources.
// It's not even its final form!


// Parser to get Name-Value pairs from a stream/string(s)
// Can be used to parse command line args with 1st constructor
// Name value pairs can be read from a file by 2nd constructor ( set ignoreUptoNthDelim value to 1 if there's any prefix data).
class NameValuePairParser
{

private:
    std::string Separator;  // when there are multiple arguments passed as name value pairs, each arg is preceded by a \Separator
    char ArgDelim;          // \ArgDelim separates the name from value e.g. in "-name=value" (defaults of the first constructor) ArgDelim is '=' and Separator is "-"
    std::string CommentDelim;        // if \CommentDelim is the first char, that kills the string.
    const std::string LastLineToRead; // when processing a file, stop after reading line that begins with this.

    std::vector<std::string> Prefixes; // arguments that do not follow " <ArgDelim>name<Separator>value " format, example the prog name, i.e. argv[0] in arguments to main()

    std::vector<std::pair<std::string, std::string> > Pairs; // untyped name value pairs.

public:
    
    // when \ArgDelim is '=' , \Separator is "-", ignoreUptoNthDelim = 2 the arguments  that are correctly parsed looks like:
    
    // ProgramName.exe Prefix1 -Prefix2 -Name1=Value1 -Name2=Value2


    NameValuePairParser(
        unsigned argc, char** argv,     //  main() args
        std::string separator = "=", char argDelim = '-', std::string comDelim = "", std::string lastLine = ""):  // these 2 default params parse args like "-name=value"
        Separator(separator),
        ArgDelim(argDelim),
        CommentDelim(comDelim),
        LastLineToRead(lastLine)
    {    
        if(StringUtils::HasWSpace(separator))
            throw std::invalid_argument(separator + " is not a valid separator for the NameValue Pair, cannot contain spaces\n");

        Prefixes.push_back(argv[0]); // program name, argv[0]

        for (unsigned i = 1; i < argc; ++i)
           ProcessArg(argv[i]);
    }

    NameValuePairParser(const std::string& argString, const std::string& separator ,  char argDelim = '\0',  std::string comDelim = "", std::string lastLine = ""):
        Separator(separator),
        ArgDelim(argDelim),
        CommentDelim(comDelim),
        LastLineToRead(lastLine)
        {
            std::istringstream iss(argString);
            Init(iss);
        }

    NameValuePairParser(std::istream& arg, std::string separator, char argDelim = '\0', std::string comDelim = "", std::string lastLine = "") :
        Separator(separator),
        ArgDelim(argDelim),
        CommentDelim(comDelim),
        LastLineToRead(lastLine)
        {
            Init(arg);
        }

    void ProcessArg(std::string arg)
    {
        StringUtils::LTrim(arg);
        StringUtils::RTrim(arg, CommentDelim);
        if (!arg.length()) return;


        if(ArgDelim && arg[0] != ArgDelim)
            throw std::invalid_argument("Bare word argument found at: " + arg);
        
        size_t splitpoint = arg.find(Separator);
        if (splitpoint == std::string::npos)
            arg.erase(arg.begin(), arg.begin() + 1),
            Prefixes.push_back(arg);
        else
        {
            Pairs.push_back(
                make_pair(
                    PreprocessName(std::string(arg, false, splitpoint)), 
                    StringUtils::StrTrim(std::string(arg.begin() + splitpoint + Separator.length(), arg.end()))
                    )
                );
        }
    }

    //const StringPairVector& GetPairs() const { return Pairs;}

    template<typename ValType>
    std::vector<std::pair<std::string, ValType> >  GetPairs()
    {
        std::vector<std::pair<std::string, ValType> > castPairs(Pairs.size());

        for (unsigned i = 0; i < Pairs.size(); ++i)
            castPairs[i] = std::make_pair(Pairs[i].first, 
                StringUtils::StringToType<ValType>(Pairs[i].second));

        return castPairs;
    }

    //template<> StringPairVector GetPairs<std::string>() { return Pairs; }

    template<typename ValType> 
    inline bool Get(const std::string& name, ValType& value) const  // returns true if found and parsed, if !caseSensitive search is linear
    {
        const std::string& ppName = PreprocessName(name);

        std::stringstream ss;

        for (auto& pair : Pairs)
            if (pair.first == ppName && ss << StringUtils::StrTrim(pair.second))
                break;

        ss >> value;
        return false;
    }

    template<typename ValType> // trivially constructible  ValType
    ValType Get(const std::string& name, bool caseSensitive = true) const
    {
        ValType v; 
        Get(name, v, caseSensitive);
        return v;
    }

    size_t GetNumPrefixes() const { return Prefixes.size(); }

    const inline std::string& GetArg(unsigned idx) const { return Prefixes[idx]; } // return ws separated prefix
    
    inline void Die() {  Pairs.clear();  Prefixes.clear(); }

private:

    inline std::string PreprocessName(const std::string& name) const
    {
        if (name[0] == ArgDelim)
            return StringUtils::StrTrim(std::string(name.begin() + 1, name.end()));
        return StringUtils::StrTrim(name);
    }

    void Init(std::istream& strm)
    {
        std::string str;
        const unsigned bufsize = 512;
        char buf[bufsize];

        while (strm.getline(buf, bufsize))
        {
            
            if (LastLineToRead.length() && StringUtils::StrTrim(buf).find(LastLineToRead, 0) == 0)
                break;

            if(strm.gcount() > bufsize)
                throw std::overflow_error("Extracted line was too long, Ignoring line\n");

            ProcessArg(std::string(buf));
        }

            
    }
};

#endif
