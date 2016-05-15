#ifndef __ACTIVATION_HXX__
#define __ACTIVATION_HXX__

#include <string>
#include <vector>
#include "utils/Exceptions.hxx"
#include "utils/utils.hxx"
#include <algorithm>

double SigmoidActivation(double p, double& grad)
{
    p = 1.f / (1.f + exp(-p));
    grad = p * (1 - p);
    return p;
}

double TanHActivation(double p, double& grad) 
{
    p = tanh(p);
    grad = 1 - p*p;
    return p;
}


typedef decltype(&SigmoidActivation)        ActivationFunction;
typedef decltype(&Utils::RoundedCompare)    ResultCmpPredicate;

struct Activation
{
    std::string Name;
    ActivationFunction Function;
    ResultCmpPredicate ResultCmpPredicate;
    double Eta;
    std::pair<double, double> MinMax;
};

inline const Activation* GetActivationByName(std::string name)
{
    static Activation List[] = {
        {"Sigmoid", SigmoidActivation, Utils::RoundedCompare,  0.25f, { 0.f, 1.f} },
        {"TanH",    TanHActivation,    Utils::SameSign,        0.5f, {-1.f, 1.f} }
    };

    for (unsigned i = 0; i < ARRAY_LENGTH(List); ++i)
        if (List[i].Name == name) return &(List[i]);
    
    THROW(InvalidOptionException, "Activation function by name of \"%s\" does not exist", name.c_str());
}

#endif
