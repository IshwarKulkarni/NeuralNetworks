/*
Copyright (c) Ishwar R. Kulkarni
All rights reserved.

This file is part of NeuralNetwork Project by 
Ishwar Kulkarni , see https://github.com/IshwarKulkarni/NeuralNetworks

If you so desire, you can copy, redistribute and/or modify this source 
along with  rest of the project. However any copy/redistribution, 
including but not limited to compilation to binaries, must carry 
this header in its entirety. A note must be made about the origin
of your copy.

NeuralNetwork is being distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
FITNESS FOR A PARTICULAR PURPOSE.

*/

#ifndef DATASETS_HXX_INCLUDED
#define DATASETS_HXX_INCLUDED

#include <string>
#include <vector>
#include <type_traits>

#include "utils/Vec23.hxx"
#include "utils/SimpleMatrix.hxx"
#include "data/MNIST-ImageReader.hxx"
#include "data/CIFAR-ImageReader.hxx"

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

    TargetPatternDef(unsigned numTargetClasses = 0, TargetOutputType type = UseBinaryArray, 
        double fillLo = 0, double fillHi = 1) :
        FillHigh(fillHi),
        FillLow(fillLo),
        NumTargetClasses(numTargetClasses),
        TargetVectorSize(0),        
        TargetType(type){}
};

struct Char255ToDoubleVolume {
    inline void operator()(unsigned char*** in, SimpleMatrix::Matrix3<double>& out) {
        auto inLin = in[0][0];
        for (auto& o : out) o = double(*inLin++) / 255;
    }
};

template<typename TI, typename ConverterType = Char255ToDoubleVolume > // First type should have [] operator, second 
class PatternSet
{
    struct Pattern
    {
        ConverterType Converter;
        TI Input;
        void GetInput(SimpleMatrix::Matrix3<double>& V) { Converter(Input, V); }
        double* Target;
    };
    
    /* 
    |---------------------------|--------------|-----------|
    |{      Training Set       }|{ Validation }|{ TestSet }|
    */
public:

    inline PatternSet(unsigned dataSize, double validationFraction, double testFraction,
        TargetPatternDef& tgtPtrn) :
        DataSize(dataSize),
        TestSetsize(unsigned(dataSize * testFraction)),
        VldnSize(unsigned(dataSize * validationFraction)),
        TrainSize(dataSize - TestSetsize - VldnSize),
        Targets(0),
        DataToDelete(nullptr),
        Patterns(dataSize)       
    {
        if(testFraction + validationFraction >= 1.0)
            throw std::logic_error ("Invalid test and validation fractions: " 
                + std::to_string(testFraction) + ", " + std::to_string(validationFraction));

        if (tgtPtrn.TargetType == TargetPatternDef::UseBinaryArray)
            Targets = SimpleMatrix::NumToBinArray(tgtPtrn.NumTargetClasses, tgtPtrn.TargetVectorSize, tgtPtrn.FillLow, tgtPtrn.FillHigh);
        else if (tgtPtrn.TargetType == TargetPatternDef::UseUnaryArray)
            Targets = SimpleMatrix::NumToUnArray(tgtPtrn.NumTargetClasses, tgtPtrn.FillLow, tgtPtrn.FillHigh),
            tgtPtrn.TargetVectorSize = tgtPtrn.NumTargetClasses;
        
        TgtPtrnDef = tgtPtrn;
    }

    inline void ResetHighLow(double low, double high)
    {
        for (unsigned i = 0; i < TgtPtrnDef.NumTargetClasses; i++)
            for (unsigned j = 0; j < TgtPtrnDef.TargetVectorSize; ++j)
                Targets[i][j] = (Targets[i][j] == TgtPtrnDef.FillHigh ? high : low);
    }

    inline void Redistribute(unsigned dataSize, double testFraction, double validationFraction)
    {
        if (testFraction + validationFraction > 1.0)
            throw std::logic_error("Invalid test and validation fractions: "
                + std::to_string(testFraction) + ", " + std::to_string(validationFraction));

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
        out.flush();
    }

    inline void Clear() 
    {
        if (Targets)        SimpleMatrix::deleteColocArray(Targets); 
        if (DataToDelete)   SimpleMatrix::deleteColocArray(DataToDelete);
    }

    void PrintAllDistributions(std::ostream& out, TargetPatternDef::TargetOutputType type)
    {
        auto te = TrainEnd();
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

        std::vector<unsigned> dist(TgtPtrnDef.NumTargetClasses);

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
                <<  std::setprecision(4) << std::setw(6) << std::showpoint << std::left << std::fixed
                << "\t|" << std::string(unsigned(200.f * dist[i] / numSamples),'*') << "|\n";

        out << "********\n";
    }

    void SetDataToDelete(TI ptr) {
        DataToDelete = ptr; 
    }


private:
    unsigned DataSize, TestSetsize, VldnSize, TrainSize;
    double**     Targets;
    TI           DataToDelete;
    std::vector<Pattern>    Patterns;
    TargetPatternDef        TgtPtrnDef;
    

};

extern void  ReadDataSplitsFromFile();

PatternSet<double*> LoadMnistData(unsigned& InputSize, unsigned& OutputSize); // one dimension output

PatternSet<unsigned char***> LoadMnistData2(Vec::Size3& insize, unsigned& outsize, Vec::Vec2<double> highlo, unsigned N = MNISTReader::NumImages);

PatternSet<unsigned char***> LoadCifarData10(Vec::Size3& insize, unsigned& outsize, Vec::Vec2<double> highlo, unsigned N = CIFAR::NumImages);
#endif
