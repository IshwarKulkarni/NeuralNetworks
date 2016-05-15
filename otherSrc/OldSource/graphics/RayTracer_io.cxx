#include "graphics/Raytracer.hxx"
#include <string>
#include "utils/StringUtils.hxx"
#include "graphics/geometry/vectors_io.hxx"
#include "utils/Logging.hxx"

using namespace std;
using namespace StringUtils;
using namespace Logging;

static CommentHogger PCommentHogger = { '#' };

extern StringCaseInsensitiveMap<Readable*>* ReadableObjectsMap;

void RayTracer::BuildAllElse()
{
    MaterialContainerInst().Build();
    ObjectContainerInst().Build();
    RasterizerInst().Build();
    CameraInst().Build();
    ShaderInst().Build();

    cout << "Following Readable Objects are registered: \n";
    for (auto& i : *ReadableObjectsMap)
        cout << i.first << LogEndl;

    cout
        << "\n---------------------------------"
        << "\nTotal solid objects     : " << ObjectContainerInst().ObjectVec->size()
        << "\nTotal materials defined : " << MaterialContainerInst().MaterialVec->size()
        << "\nDrawing Axes            : " << boolalpha << ShaderInst().MarkAxes
        << "\n---------------------------------\n\n"
        << "Percentage done:  0";

}

void RayTracer::Build(istream& sceneFileStream)
{
    Timer BuildTimer("Build Time"); 

    THROW_IF(!sceneFileStream.good(), FileIOException, "Scene file not readable\n");

    while (sceneFileStream)
    {
        string prefix;
        sceneFileStream >> std::skipws >> SComment(prefix) >> PCommentHogger;
        if (prefix.length())
        {
            auto found = ReadableObjectsMap->find(prefix);

            THROW_IF(found == ReadableObjectsMap->end(), FileParseException, "Unknown identifier %s", prefix.c_str());

            found->second->MakeFromStream(sceneFileStream);
        }
    }
    BuildAllElse();
}

bool Material::ReadMaterial(istream& strm, Material* mat)
{
    char sep = 0;
    string temp;
    bool somethingRead = false;
    while (sep != StatementDelimiter && strm.peek() != StatementDelimiter)
    {
        strm >> SComment(temp) >> PCommentHogger;
        if (temp[0] == StatementDelimiter)
            break;

        if (CaseInsensitiveMatch()(temp, "diffuse:"))
        {
            strm >> SComment(mat->diffuse) >> SComment(sep);
            somethingRead = true;
        }
        else if (CaseInsensitiveMatch()(temp, "specular:"))
        {
            strm >> SComment(mat->specular) >> SComment(sep);
            somethingRead = true;
        }
        else if (CaseInsensitiveMatch()(temp, "emission:"))
        {
            strm >> SComment(mat->emission) >> SComment(sep);
            somethingRead = true;
        }
        else if (CaseInsensitiveMatch()(temp, "ambient:"))
        {
            strm >> SComment(mat->ambient) >> SComment(sep);
            somethingRead = true;
        }
        else if (CaseInsensitiveMatch()(temp, "reflectivity:"))
        {
            strm >> SComment(mat->reflectivity) >> SComment(sep);
            somethingRead = true;
        }
        else if (CaseInsensitiveMatch()(temp, "translucency:"))
        {
            strm >> SComment(mat->translucency) >> SComment(sep);
            somethingRead = true;
        }
        else if (CaseInsensitiveMatch()(temp, "Phong_exp:"))
        {
            strm >> SComment(mat->Phong_exp) >> SComment(sep);
            somethingRead = true;
        }
    }

    if (!(mat->emission == Blackf))
        mat->type = Material::MaterialType_Light;

    if (!(mat->translucency == Blackf))
        mat->type = Material::MaterialType_Refractive;

    if (sep != StatementDelimiter)
        CheckStatementEnd(strm);

    return somethingRead;
}

Material* Material::MakeFromStream(istream& strm)
{
    Material* mat = new Material;

    string temp; char sep;
    strm >> PCommentHogger >> sep; // this sep is "

    getline(strm, temp, '\"');
    temp = "\"" + temp + "\""; // this is double quoted name

    bool MaterailReadSuccessfully = ReadMaterial(strm, mat);
    THROW_IF(!MaterailReadSuccessfully, FileParseException,
        "Material parsing failed", PrintLocation(strm, Log));

    MaterialContainerInst().AddMaterial(temp, mat);

    return mat;
}

Camera* Camera::MakeFromStream(istream& strm)
{
    Camera& camera = CameraInst();
    char sep;
    if (strm){
        strm >> std::skipws
            >> SComment(camera.eye)     >> SComment(sep)
            >> SComment(camera.lookat)  >> SComment(sep)
            >> SComment(camera.up)      >> SComment(sep)
            >> SComment(camera.x_win)   >> SComment(sep)
            >> SComment(camera.y_win)   >> SComment(sep)
            >> SComment(camera.x_res)   >> SComment(sep)
            >> SComment(camera.y_res)   >> SComment(sep)
            >> SComment(camera.vpdist)  >> SComment(sep);
    }
    THROW_IF(sep != StatementDelimiter, FileParseException, "No end of statement \';\' at the end of Camera description");
    return &camera;
}

Shader* Shader::MakeFromStream(std::istream& strm)
{
    Shader& shader = ShaderInst();

    char sep = 0;
    string temp;
    bool somethingRead = false;
    while (sep != StatementDelimiter && strm.peek() != StatementDelimiter)
    {
        strm >> SComment(temp) >> PCommentHogger;
        if (CaseInsensitiveMatch()(temp, "AxesColor:"))
        {
            strm    
                >> SComment(shader.AxesColorX) >> SComment(sep)
                >> SComment(shader.AxesColorY) >> SComment(sep)
                >> SComment(shader.AxesColorZ) >> SComment(sep);
            shader.MarkAxes = true;
        }
        else if (CaseInsensitiveMatch()(temp, "VoidColor:"))
            strm >> SComment(shader.VoidColor) >> SComment(sep);
        else
            THROW(UnexpectedLiteralException, "Unknown attribute \"%s\" for Shader\n", temp.c_str());
    }

    if (sep != StatementDelimiter)
        CheckStatementEnd(strm);

    return &shader;
}
