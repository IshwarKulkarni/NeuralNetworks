/*
Written by Ishwar Kulkarni on 23-Jun-2013
*/

#include "Utils.hxx"

template <typename Type, uint Dim>
struct MatrixStructure
{
    typedef MatrixStructure<Type,Dim>    ThisType;
    typedef MatrixStructure<Type, Dim-1> LowerType;
    
    LowerType** SubMatrix;
    uint Size;
    uint LocalDim;
    
    // Constructor
    MatrixStructure(Type* data, uint* sizes) : LocalDim(Dim), Size(*sizes)
    {
        SubMatrix = new LowerType*[Size];
        for (uint i = 0; i < Size; ++i)
        {
            SubMatrix[i] = new LowerType(data, sizes+1);
            data += SubMatrix[i]->GetNumElem();
        }
    }

    void Slice(UintPair* boundaries)
    {
        uint begin  = boundaries->first;
        uint end    = std::min(boundaries->second,Size);

        for (uint i = 0; i < boundaries->first; i++)
            delete SubMatrix[i];

        for (uint i = end; i < Size; i++)
            delete SubMatrix[i];

        auto newSize = boundaries->second - boundaries->first;

        LowerType** newSub = newSize > 0 ? nullptr : new LowerType*[Size];

        for (uint i =  boundaries->first; i <  boundaries->second; i++)
            newSub[i- boundaries->first] = SubMatrix[i];
        
        if(newSub)
        {
            Size = newSize;
            delete[] SubMatrix;
            SubMatrix = newSub;
        }
        
        for (uint i = 0; i < newSize; i++)
            SubMatrix[i]->Slice(++boundaries);

    }

    inline uint GetSize() const { return Size; }
    inline uint GetNumElem() const { return Dim*SubMatrix[0]->GetNumElem(); }
    inline const LowerType& operator[](uint index) const { return *(SubMatrix[index]); }

    ~MatrixStructure()
    {
        for (uint i = 0; i < Size; ++i)
            delete SubMatrix[i];
        
        delete[] SubMatrix;
    }
};

template <typename Type>
struct MatrixStructure<Type, 1>
{
    uint Size;
    Type* SubMatrix;
    uint LocalDim;
    
    MatrixStructure(Type* data, uint* sizes): LocalDim(1), Size(*sizes), SubMatrix(data)
    {
    }

    void Slice(UintPair* boundaries)
    {
        uint newSize = boundaries->second - boundaries->first;

        Type* newSub = newSize > 0 ? nullptr : new Type[Size];

        for (uint  i = boundaries->first; i < boundaries->second; i++)
            newSub[i - boundaries->first] = SubMatrix[i];
        
        if(newSub)
        {
            Size = newSize;
            delete[] SubMatrix;
            SubMatrix = newSub;
        }
    }


    inline const Type& operator[](uint index) const{ return SubMatrix[index]; }
    inline uint GetNumElem() const { return Size; }
    inline uint GetSize()    const { return Size; }

    ~MatrixStructure()
    {
        delete[] SubMatrix;
    }
};


int main(int argc, char** argv)
{
    double array[24] = // sums to 156
    { 
        1,2,3,4,
        5,6,7,8,
        9,10,11,12,  

        // Sums to 78           
        13,14,15,16,
        17,18,19,20, // Sum to 26 
        21,22,23,24
    };

    unsigned size[] = {2,3,4};
    MatrixStructure<double,3> M(array,size);

    UintPair b[] = {std::make_pair(0,1), std::make_pair(1,2), std::make_pair(2,4)};
    M.Slice(b);


    
    return 0;
}




