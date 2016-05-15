#include "graphics/RayTracer.hxx"
#include "utils/StringUtils.hxx"
#include "utils/Exceptions.hxx"
#include "utils/Logging.hxx"
#include "utils/Readable.hxx"

using namespace std;
using namespace StringUtils;
using namespace Logging;

StringCaseInsensitiveMap<Readable*> *ReadableObjectsMap = 0;

extern bool RegisterReadbleObject(Readable* readable)
{
    if (ReadableObjectsMap == 0)
        ReadableObjectsMap = new StringCaseInsensitiveMap<Readable*>();

    ReadableObjectsMap->insert(make_pair(readable->GetPrefix(), readable));
    return true;
}


istream& operator >>(istream& istrm, const CommentHogger& c)
{
    istrm >> skipws;
    static char comment[MaxCommentLineLength];
    char sep;
    do
    {
        memset(comment, 0, MaxCommentLineLength);
        istrm >> skipws >> comment[0];

        if (comment[0] == c.CommentPrefix)
            istrm.getline(comment, MaxCommentLineLength);
        else
            istrm.putback(comment[0]);
        sep = istrm.peek();
    } while ( sep == c.CommentPrefix || iswspace(sep));

    return istrm;
}

