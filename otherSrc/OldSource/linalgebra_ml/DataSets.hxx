#ifndef DATASETS_HXX_INCLUDED
#define DATASETS_HXX_INCLUDED

//#include "utils/Utils.hxx"
#include <string>
#include <vector>
#include <type_traits>

#include "graphics/geometry/Vec23.hxx"
#include "imageprocessing/Color.hxx"
#include "utils/exceptions.hxx"
#include "utils/SimpleMatrix.hpp"
#include "linalgebra_ml/CIFAR-ImageReader.hxx"
#include "linalgebra_ml/MINST-ImageReader.hxx"


using namespace __IM_Color__;
struct TargetPatternDef
{
    enum TargetOutputType {
        UseBinaryArray,
        UseUnaryArray,
        UseNone
    };

    double      FillHigh, FillLow;
    unsigned    NumTargetClasses;
    unsigned    TargetVectorSize;
    TargetOutputType TargetType;

    TargetPatternDef(unsigned numTargetClasses = 0, TargetOutputType type = UseBinaryArray) :
        FillHigh(1),
        FillLow(-1),
        TargetVectorSize(0),
        NumTargetClasses(numTargetClasses), 
        TargetType(type){}
};

template<typename TI> // These two types should ahve [] operator
class PatternSet
{
    struct Pattern    
    {
        TI Input;
        double* Target;
    };
    
    /* 
    |------------------------|--------------|-----------|
    |{    Training Set      }|{ Validation }|{ TestSet  } */
    std::vector<Pattern>  Patterns;
    TargetPatternDef TgtPtrnDef;

public:

    inline PatternSet(unsigned dataSize, double validationFraction, double testFraction,
        TargetPatternDef& tgtPtrn) :
        DataSize(dataSize),
        TestSetsize(unsigned(dataSize * testFraction)),
        VldnSize(unsigned(dataSize * validationFraction)),
        TrainSize(dataSize - TestSetsize - VldnSize),
        Patterns(dataSize),
        Targets(0)
    {
        THROW_IF(testFraction + validationFraction > 1.0, LogicalOrMath,
            "Invalid test(%f) and validation(%f) fractions\n", testFraction, validationFraction);

        if (tgtPtrn.TargetType == TargetPatternDef::UseBinaryArray)
            Targets = NumToBinArray(tgtPtrn.NumTargetClasses, tgtPtrn.TargetVectorSize, tgtPtrn.FillLow, tgtPtrn.FillHigh);
        else if (tgtPtrn.TargetType == TargetPatternDef::UseUnaryArray)
            Targets = NumToUnArray(tgtPtrn.NumTargetClasses, tgtPtrn.FillLow, tgtPtrn.FillHigh),
            tgtPtrn.TargetVectorSize = tgtPtrn.NumTargetClasses;
        
        TgtPtrnDef = tgtPtrn;
        DataToDelete = 0;
    }

    inline void ResetHighLow(double low, double high)
    {
        for (unsigned i = 0; i < TgtPtrnDef.NumTargetClasses; i++)
            for (unsigned j = 0; j < TgtPtrnDef.TargetVectorSize; ++j)
                Targets[i][j] = (Targets[i][j] == TgtPtrnDef.FillHigh ? high : low);
    }

    inline void Redistribute(unsigned dataSize, double testFraction, double validationFraction)
    {
        THROW_IF(testFraction + validationFraction > 1.0, LogicalOrMath,
            "Invalid test(%f) and validation(%f) fractions\n", testFraction, validationFraction);

        TestSetsize = unsigned(dataSize * testFraction),
            VldnSize = unsigned(dataSize * validationFraction),
            TrainSize = dataSize - TestSetsize - VldnSize;
    }

    typedef typename std::vector<Pattern>::iterator Iter;

    inline Iter TrainBegin() { return Patterns.begin();    }
    inline Iter TrainEnd()   { return Patterns.begin() + TrainSize; }
    inline Iter VldnBegin()  { return Patterns.begin() + TrainSize; }
    inline Iter VldnEnd()    { return Patterns.begin() + TrainSize + VldnSize; }
    inline Iter TestBegin()  { return Patterns.begin() + TrainSize + VldnSize; }
     inline Iter TestEnd()   { return Patterns.end();        }

    inline unsigned GetDataSize()    { return  DataSize;   }
    inline unsigned GetTestSetsize() { return  TestSetsize;}
    inline unsigned GetVldnSize()    { return  VldnSize;   }
    inline unsigned GetTrainSize()   { return  TrainSize;  }

    inline void ShuffleAll()        { std::random_shuffle(TrainBegin(), TestEnd()); }
    inline void ShuffleTrain()      { std::random_shuffle(TrainBegin(), TrainEnd());}
    inline void ShuffleTrnVldn()    { std::random_shuffle(TrainBegin(), VldnEnd()); }
    inline void ShuffleTest()       { std::random_shuffle(TestBegin(), TestEnd());  }

    inline Pattern& operator[](size_t idx)        { return Patterns[idx]; }
    inline std::vector<Pattern>& GetPatterns()    { return Patterns;      }

    inline double* GetTarget(unsigned u) { return Targets[u]; }

    inline void Summarize(std::ostream& out, bool printAllDistributions = true)
    {
        out << "\nData set summary:"
            << "\n# Training Patterns   : " << TrainSize
            << "\n# Testing Patterns    : " << DataSize - TrainSize - VldnSize
            << "\n# Validation Patterns : " << VldnSize
            << "\n# Target classes      : " << TgtPtrnDef.NumTargetClasses;
            
        
        if (TgtPtrnDef.TargetType != TargetPatternDef::UseNone)
        {
            out << "\nTarget Pattern type   : " <<
                (TgtPtrnDef.TargetType == TargetPatternDef::UseBinaryArray ? "Binary" : "Unary")
                << ", with length " << TgtPtrnDef.TargetVectorSize
                << " {" << int(TgtPtrnDef.FillHigh) << ", " << int(TgtPtrnDef.FillLow) << "}\n";

            if (printAllDistributions)  PrintAllDistributions(out, TgtPtrnDef.TargetType);
        }
        out << "\n=============================================\n";
    }

    inline void Clear() 
    {
        if (Targets) SimpleMatrix::deleteColocArray(Targets); 
        if (DataToDelete2) SimpleMatrix::deleteColocArray(DataToDelete2); //order is important when 3d arrays are deleted.
        if (DataToDelete)  delete[] DataToDelete2;
    }

    void PrintAllDistributions(std::ostream& out, TargetPatternDef::TargetOutputType type)
    {
        auto& te = TrainEnd();
        te--;
        PrintDistribution(out, TrainBegin(), te, type, "Training Set");

        if (TestSetsize)
            PrintDistribution(out, TestBegin(), TestEnd(), type, "Test Set");
        if (VldnSize)
            PrintDistribution(out, VldnBegin(), VldnEnd(), type, "Validation Set");
    }

    void PrintDistribution(std::ostream& out, Iter start, Iter end,
        TargetPatternDef::TargetOutputType type, std::string msg = "")
    {
        unsigned numSamples = 0;
        int patternSize = TgtPtrnDef.TargetVectorSize;
        if (type == TargetPatternDef::UseNone)
        {
            out << "Assuming Binary output type:\n";
            type = TargetPatternDef::UseBinaryArray;
        }

        vector<unsigned> dist(TgtPtrnDef.NumTargetClasses);

        auto BinaryToInt = [&](double* Tgt) {
            unsigned a = 0;
            for (int i = 0, b = 1; i < patternSize; ++i, b *= 2)
                if (Tgt[i] == TgtPtrnDef.FillHigh)
                    a += b;
            return a;
        };

        auto UnaryToInt = [&](double* Tgt) {
            for (int i = 0; i < patternSize; ++i)
                if (Tgt[i] == TgtPtrnDef.FillHigh) return i;
            return 0;
        };

        if (type == TargetPatternDef::UseBinaryArray)
            for (auto d = start; d != end; ++d, ++numSamples)
                ++dist[BinaryToInt(d->Target)];
        else if (type == TargetPatternDef::UseUnaryArray)
            for (auto d = start; d != end; ++d, ++numSamples)
                ++dist[UnaryToInt(d->Target)];

        out << "\n" << msg << " Class Distribution [" << numSamples << "] :\n";
        for (unsigned i = 0; i < dist.size(); ++i)
            out << "\tClass " << i << " : " << 100 * (float(dist[i]) / numSamples) << "%"
                <<  setprecision(4) << setw(6) << showpoint << left << fixed
                << "\t|" << std::string(uint(200.f * dist[i] / numSamples),'*') << "|\n";

        out << "********\n";
    }

    void SetDataToDelete(TI ptr) { DataToDelete = ptr; }
    void SetDataToDelete(TI* ptr) { DataToDelete2 = ptr; }
    void SetDataToDelete(TI** ptr) { DataToDelete3 = ptr; }

private:
    unsigned DataSize, TestSetsize, VldnSize, TrainSize;
    double**     Targets;
    TI      DataToDelete, *DataToDelete2, **DataToDelete3;

};

extern void  ReadDataSplitsFromFile();

PatternSet<double*> LoadMnistData(unsigned& InputSize, unsigned& OutputSize);
/*
PatternSet<int*> OpticalDigitsdata(unsigned& InputSize, unsigned& OutputSize);

PatternSet<int*> UCIHandwriting2(unsigned& InputSize, unsigned& OutputSize);

PatternSet<double*> UCIWine(unsigned& InputSize, unsigned& OutputSize);
*/
PatternSet<double***> LoadMnistData2(Vec::Size3& InputSize, unsigned& OutputSize, unsigned N = MNISTReader::NumImages);

PatternSet<Color*> LoadCIFAR10RGB(unsigned& InputSize, unsigned& OutputSize);

PatternSet<double*> LoadCIFAR10MF(unsigned& InputSize, unsigned& OutputSize);

PatternSet<double***> LoadCIFAR10(Vec::Size3& InputSize, Vec::Size3& OutputSize, unsigned N = CIFAR::ImageReader::NumImages);

#endif

