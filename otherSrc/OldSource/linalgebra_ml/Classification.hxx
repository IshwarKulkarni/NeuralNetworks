#ifndef CLASSIFICATION_INCLUDED
#define CLASSIFICATION_INCLUDED

#include <array>
#include <vector>
#include "LinearAlgebra.hxx"

static const double ChangeFracToStop = 0.01;

template <typename DataType, uint K>
class KMeans
{
public:
    template<typename TupleContainer>
    KMeans(const TupleContainer& samples, std::vector<uint>& classes) : m_SamplesAreTrainingData ( classes.size() != 0 )
    {
        if(!m_SamplesAreTrainingData)
        {
            srand((time_t)time(NULL));
            classes.resize(samples.size());
            std::generate(classes.begin(), classes.end(), []{return Rand<uint>(K,0);});
        }
        
        m_NumSamples.fill(0);

        while(UpdateMeansAndContinue(samples,classes))
            for(uint i = 0 ; i < samples.size(); ++i)
                classes[i] = Classify(samples[i]);
    }

    template <typename Tuple1>
    uint ClassifyAndUpdate(Tuple1& sample)
    {
        auto c = Classify(sample);
        auto num = ++m_NumSamples[c];
        for (uint i = 0 ; i < m_Means.size(); ++i)
            m_Means[c][i] += sample[i]/num;

        return c;
    }


    template <typename Tuple1>
    uint Classify(const Tuple1& sample) const
    {
        std::vector<double> dist(K);
        for(uint c = 0 ; c < K; ++c) 
            dist[c] = L2Norm( m_Means[c] , sample);

        return std::min_element(dist.begin(), dist.end())-dist.begin();
    }

protected:
    std::array<uint, K> m_NumSamples; // samples in each cluster
    std::array<std::vector<DataType>,K> m_Means;
    bool m_SamplesAreTrainingData ;

public:

    const std::array<std::vector<DataType>,K> GetMeans() const { return m_Means; }

private:

    template<typename TupleContainer>
    bool UpdateMeansAndContinue( const TupleContainer &samples, const std::vector<uint>& classes )
    {
        auto prevNumSamples = m_NumSamples;

        m_NumSamples.fill(0);
        for(uint i = 0 ; i < K; ++i) 
            m_Means[i].clear(), m_Means[i].resize(samples[0].size());
        
        for(uint i = 0 ; i < samples.size(); ++i) 
        {
            m_Means[classes[i]] += samples[i];
            ++m_NumSamples[classes[i]];
        }

        uint allowedChanges = (uint)std::max(ChangeFracToStop *samples.size(),1.);

        for(uint i = 0 ; i < K; ++i) 
        {
            m_Means[i] /= m_NumSamples[i];
            allowedChanges -= abs((int)(prevNumSamples[i] - m_NumSamples[i]));
        }
        
        return allowedChanges < 1 && !m_SamplesAreTrainingData ;
    }
};


class NaiveBayesClassifier
{
    NaiveBayesClassifier(double** trainingData, int* classes, int numData, int dataWidth);

    int GetClass(double** trainingData);

    int GetClassAndUpdate(double** trainingData);

private:

    std::vector< std::vector<double> > m_Means;
    std::vector< std::vector<double> > m_SDs;
};


#endif