
#include "utils/StringUtils.hxx"
#include "utils/Utils.hxx"
#include "utils/logging.hxx"

using namespace std;
using namespace Logging;
using namespace StringUtils;

struct segment
{
public:
    segment(string& lastTime, string& lastText) :
        mText(lastText)
    {
        stringstream ss(lastTime);
        tm tmPt = { 0 };
        tmPt.tm_year = (2015 - 1900);
        string mon;
        char c;
        ss >> tmPt.tm_mday >> mon >> tmPt.tm_hour >> c >> tmPt.tm_min;
        tmPt.tm_mon = Months::MonthNameToIdx(mon.c_str());
        mTime = mktime(&tmPt);
    }

    ~segment()
    {
    }
    string mText;
    time_t mTime;
};

bool hasName(const string& s, string& lastName)
{
    static const string Names[] = { "Anand Sundar Ram","Varun Prabhu", "NIRANJAN G", "saranath arun" };
    
    for (size_t i = 0; i < ARRAY_LENGTH(Names); i++)
    {
        if(!StringUtils::beginsWith(s,Names[i] ))
            continue;;
        lastName = Names[i];
        return true;
    }
    
    return false;
}

bool nameEnd(const string& s, string& lastTime)
{
    static const string NameEnds[] = {
        "Anand ###",
        "NIRANJAN ###",
        "Varun Prabhu ###",
        "saranath ###"
    };

    for (size_t i = 0; i < ARRAY_LENGTH(NameEnds); i++)
    {
        if (!StringUtils::beginsWith(s, NameEnds[i]))
            continue;;
        
        auto b = s.find("###");
        lastTime = std::string( s.begin() + b+ 3, s.end());
        return true;
    }

    return false;

}

bool timeOnlyLine(string& s, string& lastTime)
{
    unsigned date=-1, hh = 25, mm =-1;
    static char mon[4];
    sscanf(s.c_str(), "%d %s, %d:%d\n",&date,&mon,&hh,&mm );
    if (Months::IsMonthString(mon))
            if (date <= Months::DaysInMonth(Months::MonthNameToIdx(mon)))
                return ( (lastTime = s).length() || true);

    return false;

}

int main()
{

    std::ifstream file("C:\\Users\\Ishwar\\Desktop\\filthy five.txt");

    string s;
    string lastText, lastName, lastTime;
    char buf[1024];
    std::multimap<std::string, segment> byName;
    while (file)
    {
        file.getline(buf, 1024);
        string s(buf);

        if (timeOnlyLine(s, lastTime))
            continue;
        if (hasName(s, lastName))
        {
            file.getline(buf, 1024);
            string l(buf);
            string segString;
            while (!nameEnd(l, lastTime))
            {
                segString += l;
                file.getline(buf, 1024);
                lastText = l;
                l = buf;
            }
            byName.insert(make_pair(lastName, segment(lastTime, lastText)));

        }
        else
            byName.insert(make_pair("me", segment(lastTime, string(buf))));

    }

    map<string, unsigned> byCount; // #comments
    map<string, unsigned> byLength; // lengths
    map<string, unsigned> byLinkCt; // # comments with links
    map<string, string>   byLink; // # comments with links

    unsigned cut;
    for (auto& b : byName)
    {
        byCount[b.first] ++;
        byLength[b.first] += b.second.mLastText.length();

        cut = b.second.mLastText.find("www");
        if (cut != string::npos)
        {
            byLinkCt[b.first]++;
            stringstream ss(b.second.mLastText.substr(cut + 3));
            string s;
            ss >> s;
            byLink[b.first] += (s + "\n");
        }
    }

    Log << "\nComment counts starting Jun, 6th\n";
    for (auto& b : byCount)
        Log << b.first << " : " << b.second << LogEndl;

    
    vector<float> averages;
    auto& c = byCount.begin();
    for (auto& b : byLength)
        averages.push_back(float(b.second) / (c->second)), c++;

    Log << "\nTotal Comment Length:\n";
    auto& a = averages.begin();

    for (auto& b : byLength)
        Log << b.first << " : " << b.second << " average: " << *(a++) << LogEndl;
    
    Log << "\nCounts of comments with links:\n";
    for (auto& b : byLinkCt)
        Log << b.first << " : " << b.second << LogEndl;
    
    for (auto& b : byLink)
        Log << b.first << " : " << b.second << LogEndl;

    return 0;
}