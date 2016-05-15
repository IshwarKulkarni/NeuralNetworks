#include <iostream>
#include <sstream>
#include <string>
#include <list>

using namespace std;


struct XMLTree
{
    static unsigned const short BuffLength = 1024;
    static char Buff[BuffLength];
    
    std::list<XMLTree*>  Children;
    string Name, Data;

    
    inline bool Read(istream& in)
    {
        char c1 = 0, c2 = 0;
        in >> skipws >> c1;
        
        bool read = false;
        if (in.peek() == '?')
        {
            in.getline(Buff, BuffLength, '>');
            return;
        }
        
        in >> skipws >> Name >> skipws;
        in.getline(Buff, BuffLength, '>');
        Data = Buff;
        read = true;
    }
};

int main()
{


}