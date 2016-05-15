#ifndef IMAGE_HXX_INCLUDED

#include <vector>
#include <fstream>
#include <algorithm> 
#include <exception>
#include <iostream>

template <typename Type>
Type Clip(Type v, Type sup = 0, Type inf = 0)
{
    if(v < inf) return inf;
    if(v > sup) return sup;
    return v;
}

template <typename Type>
void WriteBin( std::vector<std::vector<Type> > raster , char* name)
{

    if(raster.size() == 0 || raster[0].size()==0)
        return;

    Type maxVal = 0;
    for_each(raster.begin(), raster.end(),[&](std::vector<Type>& row) 
    { 
        maxVal = Clip(*(std::max_element(row.begin(), row.end())), numeric_limits<Type>::max(), maxVal) ;
    });

    int width = raster[0].size();
    std::ofstream outfile;
    
    try
    {
        outfile.open(name, std::ofstream::binary);

        outfile << "P2 " << width << " " << raster.size() << " " << (int) maxVal << " \n #Written by Ishwar \n\n";

        for(unsigned i = 0; i < raster.size(); ++i)
        {
            if(raster[0].size() != width )
                throw new std::exception("Irregular array width");

            for (unsigned j = 0; j < raster[i].size(); ++j)
                outfile << " " << (int) raster[i][j];

            outfile << std::endl;
        }
    }
    catch(std::exception& e)
    {
        std::cerr << "Exceptions happen " << e.what() << std::endl;
        outfile.close();
    }
    
    outfile.flush();
    outfile.close();
}

#endif