#include <stdio.h>
#include <iostream>
#include <cusparse.h>
#include <vector>
#include "vars.h"

char *dataset;

int neuron;
int layer;
int batch;
int input;
float bias;

int blocksize;

long totnz;
int **csrdispl;   
INDPREC **csrindex;
VALPREC **csrvalue;

int **csrdispl_d;   
INDPREC **csrindex_d;
VALPREC **csrvalue_d; 

FEATPREC *currfeat;
FEATPREC *nextfeat; 

FEATPREC *currfeat_d;
FEATPREC *nextfeat_d;

int *active;   
int *active_d;     

double timeio;
double timetot;
double timeinfer;
double timebalance = 0.0;
double timekernel = 0.0;
double timecopy = 0.0;

int *numbatch;
int *batchdispl;
int mybatch;

__device__ float __ReLU(float x){
    return x<0.0?0.0:x>32.0?32.0:x;
};
 

float ReLU(float x){
    return x<0.0?0.0:x>32.0?32.0:x;
 };


void setup_gpu() {
    OR_FATAL(cudaSetDevice(0));
    int deviceCount;
    OR_FATAL(cudaGetDeviceCount(&deviceCount));
    int dev = 0;
    cudaDeviceProp deviceProp;
    OR_FATAL(cudaGetDeviceProperties(&deviceProp, dev));

    csrdispl_d = new int*[layer];
    csrindex_d = new INDPREC*[layer];
    csrvalue_d = new VALPREC*[layer];
    
    for(int l = 0; l < layer; l++){
        OR_FATAL(cudaMalloc((void**)&csrdispl_d[l], sizeof(int) * (neuron+1)));
        OR_FATAL(cudaMemcpy(csrdispl_d[l], csrdispl[l], sizeof(int) * (neuron+1), cudaMemcpyHostToDevice));

        OR_FATAL(cudaMalloc((void**)&csrindex_d[l], sizeof(INDPREC) * csrdispl[l][neuron]));
        OR_FATAL(cudaMemcpy(csrindex_d[l], csrindex[l], sizeof(INDPREC) * csrdispl[l][neuron], cudaMemcpyHostToDevice));

        OR_FATAL(cudaMalloc((void**)&csrvalue_d[l], sizeof(VALPREC) * csrdispl[l][neuron]));
        OR_FATAL(cudaMemcpy(csrvalue_d[l], csrvalue[l], sizeof(VALPREC) * csrdispl[l][neuron], cudaMemcpyHostToDevice));
    }
    OR_FATAL(cudaMalloc((void**)&currfeat_d, sizeof(FEATPREC) * mybatch * neuron));
    OR_FATAL(cudaMemset(currfeat_d, 0, sizeof(FEATPREC) * mybatch * neuron));

    OR_FATAL(cudaMalloc((void**)&nextfeat_d, sizeof(FEATPREC) * mybatch * neuron));
    OR_FATAL(cudaMemset(nextfeat_d, 0, sizeof(FEATPREC) * mybatch * neuron));

    OR_FATAL(cudaMemcpy(currfeat_d, currfeat, sizeof(FEATPREC) * mybatch * neuron, cudaMemcpyHostToDevice));

    OR_FATAL(cudaMalloc((void**)&active_d, sizeof(int) * mybatch));
    OR_FATAL(cudaMemset(active_d, 0, sizeof(int)*mybatch));

    OR_FATAL(cudaMemcpy(active_d, active,sizeof(int) * mybatch,cudaMemcpyHostToDevice));
}

double kernel_spmm(int l) {
    cusparseHandle_t handle = NULL;
    cusparseSpMatDescr_t matA;
    cusparseDnMatDescr_t matB, matC;
    void* dBuffer    = NULL;
    size_t bufferSize = 0;
    float alpha = 1.0f;
    float beta = 0.0f;

    CUSPARSE_CHECK( cusparseCreate(&handle) )
    // Create sparse matrix A in CSR format
    
    OR_FATAL(cudaMemcpy(currfeat_d, currfeat, sizeof(FEATPREC) * mybatch * neuron, cudaMemcpyHostToDevice));
    OR_FATAL(cudaMemset(nextfeat_d, 0, sizeof(FEATPREC) * mybatch * neuron));
    OR_FATAL(cudaMemset(active_d, 0, sizeof(int) * mybatch));

    CUSPARSE_CHECK(cusparseCreateCsr(&matA, neuron, neuron, csrdispl[l][neuron],
                                      csrdispl_d[l], csrindex_d[l], csrvalue_d[l],
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F))
    // Create dense matrix B
    CUSPARSE_CHECK( cusparseCreateDnMat(&matB, neuron, mybatch, neuron, currfeat_d,
                                        CUDA_R_32F, CUSPARSE_ORDER_COL) )
                                        
    // Create dense matrix C
    CUSPARSE_CHECK( cusparseCreateDnMat(&matC, neuron, mybatch, neuron, nextfeat_d,
                                        CUDA_R_32F, CUSPARSE_ORDER_COL) )
    

                                        
    OR_FATAL(cudaMalloc(&dBuffer, bufferSize));
              

    CUSPARSE_CHECK( cusparseSpMM_bufferSize(
                                 handle,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha, matA, matB, &beta, matC, CUDA_R_32F,
   
                                 CUSPARSE_CSRMM_ALG1, &bufferSize) )
   
                                 cudaEvent_t start, stop;
                                 cudaEventCreate(&start);
                                 cudaEventCreate(&stop);
                                 cudaEventRecord(start, 0);
   
   
                                 CUSPARSE_CHECK( cusparseSpMM(handle,
                                CUSPARSE_OPERATION_NON_TRANSPOSE,
                                CUSPARSE_OPERATION_NON_TRANSPOSE,
                                &alpha, matA, matB, &beta, matC, CUDA_R_32F,
                                CUSPARSE_MM_ALG_DEFAULT, dBuffer) )
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    float elapsed;
    cudaEventElapsedTime(&elapsed, start, stop); //ms 
    elapsed /= 1000.0f; // s

    // destroy matrix/vector descriptors
    CUSPARSE_CHECK( cusparseDestroySpMat(matA) )
    CUSPARSE_CHECK( cusparseDestroyDnMat(matB) )
    CUSPARSE_CHECK( cusparseDestroyDnMat(matC) )
    CUSPARSE_CHECK( cusparseDestroy(handle) )

    OR_FATAL(cudaMemcpy(nextfeat, nextfeat_d, neuron * mybatch  * sizeof(float), cudaMemcpyDeviceToHost));

    for(int i = 0; i < mybatch; ++i) {
        active[i] = 0;
    }
    
    for(int i = 0; i < mybatch; ++i) {
       for(int j = 0; j < neuron; ++j) {
            if(nextfeat[i * neuron + j] =  ReLU(nextfeat[i * neuron + j] + bias))
                active[i] += 1;
        }
    }

    int feature = 0;
    for(int i = 0; i < mybatch; ++i) {
        if(active[i]) {
            for(int j = 0; j < neuron; ++j) {
                nextfeat[feature * neuron + j] = nextfeat[i * neuron + j];
            }
            feature++;
        }
    }

    mybatch = feature;
    FEATPREC *tempfeat = currfeat;
    currfeat = nextfeat;
    nextfeat = tempfeat;
    return double(elapsed);
}

int main(int argc, char* argv[]) {

    dataset = "/qfs/projects/pacer/leeh736/dataset"; 
    char *chartemp;
    neuron = 16384;
    layer = 1920;
    batch = 60000;
    input = 98858913; // 392191985; // 98858913; // 25019051; //6374505;
    bias = 0;
    
    mybatch = batch;
  
    csrdispl = new int*[layer];
    csrindex = new INDPREC*[layer];
    csrvalue = new VALPREC*[layer];
    currfeat = new FEATPREC[neuron*(long)mybatch];
    nextfeat = new FEATPREC[neuron*(long)mybatch];
  
    active = new int [mybatch];
    
    
    printf("%d neurons, %d layers", neuron, layer) ;
    printf("\n");
    printf("READING WEIGHTS\n");
    readweights();
    printf("READING INPUT\n");
    readinput();

    for(int k = 0; k < mybatch; k++){
      active[k] = neuron;
    }
    

    setup_gpu();

    printf("INFERENCE......\n");
    printf("for %d layers......\n", layer);
    double spmm_times = 0; 
    clock_t total_start = clock();
    for(int i = 0; i < layer; ++i) {
        printf("[%d]", i);
        fflush(stdout);
        auto t = kernel_spmm(i);
        spmm_times += double(t);
        printf(":(%lf)", t);
        fflush(stdout);
    }
    clock_t end_start = clock();
    auto gemm_time = double(spmm_times);
    auto all_time = double(end_start - total_start)  / CLOCKS_PER_SEC;
    
    printf("Inference time : %lfs, %lfs, %f TTEPS\n", gemm_time, all_time, long((long)batch * (long)neuron * 32 * layer) / gemm_time / 1e12);
	return 0;
}


void readweights(){
    totnz = 0;
    for(int l = 0; l < layer; l++){
        int rownz[neuron];
        for(int n = 0; n < neuron; n++)
            rownz[n] = 32;
        csrdispl[l] = new int[neuron+1];
        csrdispl[l][0] = 0;
        for(int n = 1; n < neuron+1; n++)
            csrdispl[l][n] = csrdispl[l][n-1]+rownz[n-1];
        totnz += csrdispl[l][neuron];
        csrindex[l] = new INDPREC[csrdispl[l][neuron]];
        csrvalue[l] = new VALPREC[csrdispl[l][neuron]];
    }

    printf("weights: %ld (%f GB)\n",totnz,totnz*(sizeof(INDPREC)+sizeof(VALPREC))/1.0e9);
    
    char filename[500];
    sprintf(filename,"%s/neuron%d.bin",dataset,neuron);
    printf("open filename = %s\n", filename);
    FILE *weightf = fopen(filename,"rb");
    for(int l = 0; l < layer; l++){
        int *row = new int[csrdispl[l][neuron]];
        int *col = new int[csrdispl[l][neuron]];
        float *val = new float[csrdispl[l][neuron]];
        fread(row, sizeof(int), csrdispl[l][neuron], weightf);
        fread(col, sizeof(int), csrdispl[l][neuron], weightf);
        fread(val, sizeof(int), csrdispl[l][neuron],weightf);
        int rownz[neuron];
        for(int n = 0; n < neuron; n++)
            rownz[n] = 0;
        for(int n = 0; n < csrdispl[l][neuron]; n++){
            csrindex[l][csrdispl[l][row[n]-1]+rownz[row[n]-1]] = col[n]-1;
            csrvalue[l][csrdispl[l][row[n]-1]+rownz[row[n]-1]] = val[n];
            rownz[row[n]-1]++;
        }
        delete[] row;
        delete[] col;
        delete[] val;
    }
    fclose(weightf);
}


void readinput(){
    char filename[500];
    printf("features: %ld (%f GB)\n",neuron*(long)batch*2,neuron*(long)batch*2*sizeof(FEATPREC)/1.0e9);
    sprintf(filename, "%s/sparse-images-%d.bin", dataset, neuron);
    FILE *inputf = fopen(filename,"rb");
    int *row = new int[input];
    int *col = new int[input];
    float *val = new float[input];
    fread(row,sizeof(int),input,inputf);
    fread(col,sizeof(int),input,inputf);
    fread(val,sizeof(float),input,inputf);
    for(long n = 0; n < neuron * (long)batch; n++)
        currfeat[n] = 0.0;
    for(int n = 0; n < input; n++) {
        if(col[n] - 1 < batch) {
            currfeat[(col[n] - 1) * (long)neuron + row[n] - 1] = val[n];
        }
    }
    fclose(inputf);
    delete[] row;
    delete[] col;
    delete[] val;
}
