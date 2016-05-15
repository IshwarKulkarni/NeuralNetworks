#ifndef _READABLE_HXX_
#define _READABLE_HXX_

#include <string>
#include <istream>
#include <istream>
#include "utils/StringUtils.hxx"

#define MaxCommentLineLength 256
#define StatementDelimiter ';'

#define SComment(s)  PCommentHogger >> s

struct CommentHogger {
    char CommentPrefix;
};

std::istream& operator >>(std::istream& istrm, const CommentHogger& c);

struct Readable
{
    virtual std::string GetPrefix() const = 0;
    virtual Readable* MakeFromStream(std::istream& strm) = 0;
};

extern bool RegisterReadbleObject(Readable* readable);

#define Register(class_name)  \
    static bool dummy_variable_##class_name = RegisterReadbleObject( new class_name );


inline void CheckStatementEnd(std::istream& strm)
{
    char endStatement;

    CommentHogger PCommentHogger = { '#' };
    strm >> PCommentHogger >> endStatement;
    THROW_IF(endStatement != StatementDelimiter, FileParseException,
        "Line not ended with \';\' in the above location", StringUtils::PrintLocation(strm, Logging::Log));
}


#endif