texture<float,2> Data;
texture<float,2> Centroids;

typedef unsigned int uint;
void hostCompute(uint width, uint height, vector<float>& rawDataVec, uint numClasses, float* classes, float* compare)
{

    float** output = new float*[numClasses];
    uint numChanges;

    do
    {
        numChanges = 0;
        for (uint i = 0; i < numClasses; i++)
        {
            output[i] = new float[width];
            for (uint j = 0; j < width; j++)
                output[i][j] = 0.f;
        }


        for (uint i = 0; i < height; i++)
        {
            uint c = (uint)(classes[i]);
            ++output[c][0] ;
            for(uint j = 1; j < width; j++)
                output[c][j] += rawDataVec[i*width + j];
        }

        for (uint i = 0; i < numClasses; i++)
            for(uint j = 1; j < width; j++)
                if(output[i][0]!=0)
                    output[i][j] /= output[i][0];
    

        float minDist = numeric_limits<float>::max();
        uint c = numClasses + 1;
        for (uint i = 0; i < height; i++)
        {
            for (uint j = 0; j < numClasses; j++)
            {
                float d = 0;
                for (uint k = 1; k < width; k++)
                {
                    float absd = (rawDataVec[k + i * width] - output[j][k]);
                    d += absd*absd;
                }
                if(minDist > d)
                {
                    minDist = d;
                    c = j;
                }
            }
            if(classes[i] != c)
                ++numChanges; classes[i] = c;
        }

    } while( numChanges == 0 );

    auto cleanup = [&]() {
        for (uint i = 0; i < numClasses; i++)
            delete[] output[i];
        
        delete[] output;

        if(compare != 0)
            delete[] compare;
    };



    try
    {    
        if( compare != 0) 
            for (uint i = 0; i < numClasses; i++)
                for (uint j = 0; j < width; j++)
                     if(!FLT_CMPR(output[i][j], compare[i*width + j]))
                     {
                        TwoDPrinter.Out(Log, output, width, numClasses, "CPU Out");
                        THROW_IF( true, DataException, " output[%d][%d], %f != compare[%d], %f", i,j, output[i][j], i*width + j, compare[i*width + j]);
                     }
    }
    catch(...)
    {
        cleanup();
        throw;
    }

    cleanup();

}

template<typename T>
void hostComputePartial(const uint gridDim, const uint height, const uint width, const uint numClasses, uint rows, const T& data, const bool firstIter, float* classes, float* compare)
{
    uint origRows  = rows;
    float* out = new float[width*numClasses*gridDim];
    //Log << "\n---------\t CPU In," << height << "\n";
    //TwoDPrinter.Out(Log,data,width, height,  origRows);
    for (uint blockIdx = 0; blockIdx  < gridDim; blockIdx ++)
    {
        float* stats = new float[numClasses*width];

        for (uint threadIdx = 0 ; threadIdx < width; threadIdx++)
            for (uint i = 0; i < numClasses; i++)
                stats[i*width + threadIdx] = 0.f; // counts and stats are reset to zero here.
        
        for (uint threadIdx = 0 ; threadIdx < width; threadIdx++)
        {
            if( (blockIdx + 1 ) * rows > height)
                rows = height % rows;

            uint offset = numClasses * blockIdx;

            for (uint i = 0; i < rows; i++) 
            {
                uint y = origRows*blockIdx + i;

                uint c = firstIter ?  classes[y] : ((blockIdx*origRows + i)%numClasses);

                if( firstIter && threadIdx == 0 )
                    ++stats[c*width + threadIdx];
                else
                    stats[c*width + threadIdx] += data[threadIdx +  y*width];
            }

            offset = numClasses * blockIdx * width;
    
            for (uint i = 0; i < numClasses; i++) 
                out[threadIdx+ width* i + offset] = stats[i*width + threadIdx];    
        }

        delete[] stats;
    }
    
    
    if( gridDim == 1 )
        for (uint threadIdx = 1 ; threadIdx < width; threadIdx++)
            for (uint i = 0; i < numClasses; i++) 
                if( out[i*width] != 0 )
                    out[width*i + threadIdx] /= out[i*width]; 

    Log << "CPU Out," << numClasses*gridDim;
    TwoDPrinter.Out(Log,out,width,numClasses*gridDim);

    for (uint i = 0; i < width*numClasses*gridDim; i++)
        THROW_IF( out[i] != compare[i], ISystemExeption, "Mismatch at %d", i);

    delete[] classes;
    delete[] out;
}

uint hostUpdateClasses(uint gridDim, uint blockDim , float* devClasses, uint height, uint width, uint numClasses, float* centroids, vector<float> data)
{
    uint numHClassesChanged = 0 ;
    for (uint blockIdx = 0; blockIdx < gridDim; ++blockIdx)
    {
        float* stats = new float[numClasses*width];
        uint extent = (blockIdx == gridDim - 1 ? height%blockDim : blockDim);
        for(uint threadIdx = 0 ; threadIdx < extent; ++threadIdx)
        {
            uint y = blockIdx*blockDim + threadIdx;

            float minDist = FLT_MAX;
            uint minClass = numClasses + 1;
            for (uint c = 0; c < numClasses; c++)
            {
                float dist = 0.f;
                for (uint x = 1; x < width; x++)
                {
                    float poi = data[x + y*width];
                    float cen = centroids[x + c*width];
                    float d = (poi - cen) ;
                    dist += d*d;
                }

                if( minDist > dist )
                {
                    minDist = dist;
                    minClass = c;
                }
            }

            stats[threadIdx] = minClass;
        }
        
        
        uint numChanges = 0;
        for (uint i = 0; i < extent; i++)
        {
            uint y = blockIdx * blockDim + i;
            if (stats[i]  != devClasses[y])
                ++numChanges;
            devClasses[y] = stats[i];
        }
        numHClassesChanged  += numChanges;
    }

    //TwoDPrinter.Out(Log,devClasses,height,1, "Updated Host Classes");
    return numHClassesChanged;
}

__device__ int numClassesChanged = 0 ;

__device__ uint callnum = 0;

__shared__ float stats[]; // will be numClasses*w long for each block.

__global__ void deviceComputePartial(const uint height, const uint numClasses, uint rows, size_t pitch, const bool firstIter, const float* classes, float* out)
{
    const uint width = blockDim.x;
    uint origRows  = rows;

    if( (blockIdx.x + 1 ) * rows > height )
        rows = height % rows;

    for (uint i = 0; i < numClasses; i++)
        stats[i*width + threadIdx.x] = 0.f; // counts and stats are reset to zero here.
        
    for (uint i = 0; i < rows; i++) 
    {
        uint y = origRows*blockIdx.x + i;
        
        uint c = firstIter ?  classes[y] : ((blockIdx.x*origRows + i)%numClasses);

        if( firstIter && threadIdx.x == 0 )
            ++stats[c*width + threadIdx.x];
        else
            stats[c*width + threadIdx.x] += tex2D(Data, threadIdx.x, y); 
    }
    
    uint offset = numClasses * blockIdx.x * pitch + threadIdx.x;
    
    for (uint i = 0; i < numClasses; i++) 
        out[pitch * i + offset] = stats[i*width + threadIdx.x];    

    __syncthreads();

    if( gridDim.x == 1 && threadIdx.x != 0 )
    { // this is last call
        for (uint i = 0; i < numClasses; i++) 
            if( stats[i*width] != 0)
                out[pitch * i + offset] /= stats[i*width]; // divide by first column to get centroid
    }
}

__global__ void updateClasses(float* devClasses, uint height, uint width, uint numClasses)
{
    uint y = IMAD(blockIdx.x, blockDim.x, threadIdx.x);

    float minDist = FLT_MAX;
    uint minClass = numClasses + 1;
    for (uint c = 0; c < numClasses; c++)
    {
        float dist = 0.f;
        for (int x = 1; x < width; x++)
        {
            float poi = tex2D(Data, x,y);
            float cen = tex2D(Centroids,x,c);
            float d = (poi - cen) ;
            dist += d*d;
        }

        if( minDist > dist )
        {
            minDist = dist;
            minClass = c;
        }
    }

    stats[threadIdx.x] = minClass;
    
    __syncthreads();

    if( threadIdx.x == 0 )
    {
        uint rows = blockDim.x;
        if( (blockIdx.x == gridDim.x-1 ) && gridDim.x != 1)
            rows = height % rows; 

        uint numChanges = 0;
        for (int i = 0; i < rows; i++)
        {
            uint y = blockIdx.x * blockDim.x + i;
            if (stats[i]  != devClasses[y])
                ++numChanges;
            devClasses[y] = stats[i];
        }
        atomicAdd(&numClassesChanged, numChanges);
    }
}

void deviceCompute(vector<float>& rawDataVec, uint width, uint height, uint rowsPerBlock, uint numClasses)
{
    const size_t s = sizeof(float);
    uint origRowsPerBlock = rowsPerBlock;
    float* classes = new float[height];
    
    for (uint i = 0; i < height; i++)
        classes[i] = rawDataVec[i*width];

    float* devClasses = cudaAllocCopy(classes,height);
    delete[] classes;    

    cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
    float* centroids = 0;
    uint changed = height+1 ;
    size_t pitch;
    do
    {
        ScopedTimer("Do-While");
        uint h = height;
        uint numBlocks = 0;
        uint prevNumBlocks = 0;
        rowsPerBlock = origRowsPerBlock;
        bool firstCall = true;

        auto data = cudaPitchedAllocCopy(&(rawDataVec[0]), width,height,pitch);
        THROW_IF( cudaBindTexture2D(0,Data,data,desc,width,h, pitch), CUDABindException ,"CUDA texture bind failed");
    
        while( (numBlocks = iDivUp(h, rowsPerBlock)) != prevNumBlocks || prevNumBlocks > 1)
        {
            if(prevNumBlocks == numBlocks)
            {
                rowsPerBlock *= 2; // merge each block with next block.
                numBlocks      = iDivUp(h, rowsPerBlock);
                if( numBlocks == 2) // => no further reduction possible.
                {
                    numBlocks = 1;
                    rowsPerBlock = h;
                }
            }

            uint outHeight = numClasses*numBlocks;

            auto stats = cudaPitchedAllocCopy<float>(0,width,outHeight,pitch);

            deviceComputePartial<<<numBlocks, width, s*numClasses*width>>>(h, numClasses, rowsPerBlock, pitch/s, firstCall, devClasses , stats);
            /*
            Log<< "\n\nHeight: " << outHeight;
            TwoDPrinter.Cuda(Log,stats,pitch, width,outHeight,"  GPU Out: ");

            if( firstCall )
                hostComputePartial(numBlocks,h,width,numClasses,rowsPerBlock, rawDataVec ,                               firstCall, cudaCopyOut(devClasses,h), cudaCopyOut(stats,pitch,width,outHeight));
            else
                hostComputePartial(numBlocks,h,width,numClasses,rowsPerBlock, cudaCopyOut(centroids,pitch, width, h) , firstCall, cudaCopyOut(devClasses,h), cudaCopyOut(stats,pitch,width,outHeight));
            */
            
            h = outHeight;
            
            cudaUnbindTexture(Data);

            THROW_IF( cudaBindTexture2D(0,Data,stats,desc,width,h, pitch), CUDABindException ,"CUDA texture bind failed");

            if( centroids != 0 )
                cudaFree(centroids);

            centroids = stats; 
            
            firstCall = false;
            prevNumBlocks = numBlocks;

        }
        
        TwoDPrinter.Cuda(Log, centroids, pitch, width, numClasses, "Centroid:");
        TwoDPrinter.Cuda(Log, devClasses  , height, 1, "Classes:");
        hostCompute(width, height,rawDataVec,numClasses,cudaCopyOut(devClasses,height),cudaCopyOut(centroids, pitch, width, numClasses));

        cudaUnbindTexture(Data);
        
        THROW_IF( cudaBindTexture2D(0,Data,        data,     desc,width,height,     pitch), CUDABindException ,"CUDA texture bind failed for Data");
        THROW_IF( cudaBindTexture2D(0,Centroids,centroids,desc,width,numClasses, pitch), CUDABindException ,"CUDA texture bind failed for Centroids");
                
        void* __devPtr  = 0 ;
        THROW_IF(cudaGetSymbolAddress(&__devPtr, numClassesChanged), CUDAException, "Failed");
        THROW_IF(cudaMemset(__devPtr,0,sizeof(numClassesChanged)), CUDAException, "CudaMemset failed"); 

        cudaThreadSynchronize();
        
        uint numChangedFromHost = hostUpdateClasses(iDivUp(height,origRowsPerBlock), origRowsPerBlock , cudaCopyOut(devClasses,height), height, width, numClasses, cudaCopyOut(centroids,pitch,width,numClasses), rawDataVec);
        updateClasses<<<iDivUp(height,origRowsPerBlock), origRowsPerBlock, origRowsPerBlock >>>(devClasses, height, width, numClasses);
                
        THROW_IF( cudaMemcpyFromSymbol(&changed,numClassesChanged, sizeof(uint),0,cudaMemcpyDeviceToHost) , CUDAException, "Copy singleton variable failed");

        TwoDPrinter.Cuda(Log, devClasses  , height, 1, "Updated Classes:");

        THROW_IF( changed != numChangedFromHost, ISystemExeption, "Failed");
        
        
        Log << "# Changes : " << changed << LogEndl;
    } while( changed!= 0 );

}

