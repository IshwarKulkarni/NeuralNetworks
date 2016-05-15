#include "Classification.hxx"
#include <fstream>
#include <set>  

using namespace std;

NaiveBayesClassifier::NaiveBayesClassifier(double** trainingData, int* classes, int numData, int dataWidth)
{
    int numClasses = std::set<int>(classes,classes+numData).size();
    m_Means.resize(numClasses);
    m_SDs.resize(numClasses);

    std::vector<int> numInst(numClasses);

    for (int i = 0; i < numClasses; ++i)
        m_Means[i].resize(dataWidth), m_SDs[i].resize(dataWidth);

    for (int i = 0; i < numData; ++i)
        m_Means[j] += trainingData[i];
   
    for (int j = 0; j < dataWidth; ++j)
        m_Means[j] /= numData;

    for (int i = 0; i < numData; ++i)
        for (int j = 0; j < dataWidth; ++j)
            m_SDs[j] += L2Norm(trainingData[i],m_Means);

    for (int j = 0; j < dataWidth; ++j)
        m_SDs[j] /= numData;
}

int NaiveBayesClassifier::GetClassAndUpdate( double** trainingData )
{

}

int NaiveBayesClassifier::GetClass( double** trainingData )
{

}


int TestKMeans(int argc, char** argv)
{
    Matrix2D<float> full(argv[1]);
    full.RandPerm();

    auto x1 = full[0];
    Matrix2D<float> Split1(&x1);

    std::vector<uint> classes(Split1.size());
    KMeans<float, 3> km(Split1,classes);

    array<std::vector<double>,3> means;
    array<uint,3> samples;

    samples.fill(0);
    for(uint i = 0 ; i < 3; ++i) 
        means[i].clear(), means[i].resize(full.Width());

    for(uint i = 0 ; i < full.size(); ++i) 
    {
        means[(size_t)full(0,i)-1] += full[i];
        ++samples[(size_t)full(0,i)-1];
    }

    MatrixBlockIter<decltype(full)>  x2(full,1,full.Height()/2,full.Width()-1,full.Height()/2);
    Matrix2D<float> Split2(&x1);

    for (uint i = 0; i < full.Height()/2; ++i)
        km.ClassifyAndUpdate(Split2[i]);
    
  

    Log << "Actual:\n";
    for(uint i = 0 ; i < 3; ++i)  
    {
        means[i] /= samples[i];

        means[i].erase(means[i].begin());

        Log << LogEndl;
        for(auto m : means[i])
            Log << std::setw(8) << std::setprecision(4) << m << "\t" ;
    }
  
    auto returned = km.GetMeans();
    Log << LogEndl << LogEndl;
    for (uint i = 0; i < returned.size(); ++i)
    {
        for(uint j = 0 ; j < returned[i].size(); ++j)
            Log << std::setw(8) << std::setprecision(4) << returned[i][j] << "\t" ;
        Log << LogEndl ;
    }
        
    Log << LogEndl << LogEndl;
    for(uint i = 0 ; i < 3; ++i)  
        for(uint c = 0 ; c < 3; ++c)
            Log << "( " << i  << " , " << c << " ):" << L2Norm(means[i],km.GetMeans()[c]) << LogEndl; 


    return 0;
}

int TestWaterShed(int argc, char** argv)
{
    ImageGeneric ws(argv[1]);
    ImageProcessing::WaterShed(ws);
    return 0;
}
