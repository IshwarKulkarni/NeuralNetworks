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

#ifndef __ERROR_FUNCTION_INCLUDED__
#define __ERROR_FUNCTION_INCLUDED__

#include "utils/Utils.hxx"
#include "utils/SimpleMatrix.hxx"
#include "memory"

typedef SimpleMatrix::Matrix3<double> Volume;
typedef SimpleMatrix::Matrix<double> Frame;

struct ErrorFunctionType {
    virtual void Prime(const Volume& out, const double* target, Volume& res) const {};
    virtual void Apply(const Volume& out, const double* target, Volume& res) const {};
    virtual std::string Name() const { return ""; };
};

struct MeanSquareErrorType : public ErrorFunctionType
{
    virtual std::string Name() const override { return "MeanSquareError"; };

    virtual inline void Prime(const Volume& out, const double* target, Volume& res) const override
    {
        for (size_t i = 0; i < out.size(); ++i)
            res[i] = out[i] - target[i];
    }
    virtual inline void Apply(const Volume& out, const double* target, Volume& res) const override
    {
        for (size_t i = 0; i < out.size(); ++i)
        {
            double r = target[i] - out[i];
            res[i] = 0.5 * r*r;
        }
    }
};

// TODO : fix this!
struct CrossEntropyType  : ErrorFunctionType{
public:
    virtual std::string Name() const override { return "CrossEntropyError"; };

    virtual inline void Prime(const Volume& out, const double* target, Volume& res) const override
    {
        for (size_t i = 0; i < out.size(); ++i)
            res[i] = (out[i] - target[i]) / (out[i] * (1. - target[i]));            
    }

    inline void  Apply(const Volume& out, const double* target, Volume& res) const override
    {
        for (size_t i = 0; i < out.size(); ++i)
            res[i] = -target[i] * std::log(out[i]) - (1. - target[i]) * std::log(1. - out[i]);
    }
};

inline std::shared_ptr<ErrorFunctionType> GetErrofFunctionByName(std::string name)
{
    if (name == "CrossEntropyError")
        return std::make_shared<CrossEntropyType>();

    if (name == "MeanSquareError")
        return std::make_shared<MeanSquareErrorType>();

    return nullptr;
}


#endif