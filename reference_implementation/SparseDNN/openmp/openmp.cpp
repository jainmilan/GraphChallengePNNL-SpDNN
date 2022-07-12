#include <stdio.h>
#include <iostream>
#include <omp.h>
#include <vector>
#include <string>
#include <fstream>
#include "vars.h"


static char *dataset;
static int neuron;
static int layer;
static int batch;
static int input;
static float bias;
static std::string inputFileName;

int blocksize;

long totnz;
int **csrdispl;   
INDPREC **csrindex;
VALPREC **csrvalue;

FEATPREC *currfeat;
FEATPREC *nextfeat; 

int *active;   

double timeio;
double timetot;
double timeinfer;
double timebalance = 0.0;
double timekernel = 0.0;
double timecopy = 0.0;

int *numbatch;
int *batchdispl;
int mybatch;

float ReLU(float x){
    return x<0.0?0.0:x>32.0?32.0:x;
 };

void setup_gpu() {
    for(int l = 0; l < layer; l++){
#pragma omp target enter data map(alloc:csrdispl[l][0:neuron+1])
#pragma omp target enter data map(alloc:csrindex[l][0:csrdispl[l][neuron]])
#pragma omp target enter data map(alloc:csrvalue[l][0:csrdispl[l][neuron]])
    }
#pragma omp target enter data map(alloc:currfeat[0:mybatch*neuron])
#pragma omp target enter data map(alloc:nextfeat[0:mybatch*neuron])
#pragma omp target enter data map(alloc:active[0:mybatch])
}

void final_gpu() {
    for(int l = 0; l < layer; l++){
#pragma omp target exit data map(delete:csrdispl[l][0:neuron+1])
#pragma omp target exit data map(delete:csrindex[l][0:csrdispl[l][neuron]])
#pragma omp target exit data map(delete:csrvalue[l][0:csrdispl[l][neuron]])
    }
#pragma omp target exit data map(delete:currfeat[0:mybatch*neuron])
#pragma omp target exit data map(delete:nextfeat[0:mybatch*neuron])
#pragma omp target exit data map(delete:active[0:mybatch])
}

double kernel_spmm(int l) {

   double t0 = omp_get_wtime();
#pragma omp target teams distribute parallel for \
   collapse(2)
    for (int i = 0; i < neuron; i++) {
      for (int j = 0; j < mybatch; j++) {
        float result = 0;
        for (int p = csrdispl[l][i]; p < csrdispl[l][i+1]; p++) {
          const int k = csrindex[l][p];
          result += csrvalue[l][p] * currfeat[k*neuron + j];
        }
        nextfeat[i*neuron+j] = result;
      }
    }
   double t1 = omp_get_wtime();
                                        
#pragma omp target teams distribute parallel for 
   for(int i = 0; i < mybatch; ++i) {
        active[i] = 0;
       for(int j = 0; j < neuron; ++j) {
            if(nextfeat[i * neuron + j] =  ReLU(nextfeat[i * neuron + j] + bias))
                active[i] += 1;
        }
    }

#pragma omp target update from(currfeat[0:mybatch*neuron], nextfeat[0:mybatch*neuron], active[0:mybatch])
    
    int feature = 0, fet = 0;
#pragma omp parallel for default(shared) schedule(static)
    for(int i = 0; i < mybatch; ++i) {
        if(active[i]) {
#pragma omp atomic read
            fet = feature;
            for(int j = 0; j < neuron; ++j) {
                nextfeat[fet * neuron + j] = nextfeat[i * neuron + j];
            }
#pragma omp atomic update
            feature++;
        }
    }

    mybatch = feature;
    FEATPREC *tempfeat = currfeat;
    currfeat = nextfeat;
    nextfeat = tempfeat;
    return double(t1-t0);
}

int main(int argc, char* argv[]) {
    parseCommandLine(argc, argv);
    if (inputFileName.empty() || !neuron)
    {
      std::cout << "Input arguments missing...exiting!!!" << std::endl;
      return 0;
    }

    std::ifstream f(inputFileName.c_str());
    if (!f.good())
    {
      std::cout << "File path not found...exiting!!!" << std::endl;
      return 0;
    }
    dataset = (char*)inputFileName.c_str();
    /*
    dataset = "/lus/grand/projects/GRACE/spdnn/dataset";///qfs/projects/pacer/leeh736/dataset"; 
    char *chartemp;
    neuron = 65536;
    layer = 1920;
    batch = 60000;
    input = 392191985; // 392191985; // 98858913; // 25019051; //6374505;
    bias = 0;
    */
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
        printf(":(%lf)\n", t);
        fflush(stdout);
    }
    clock_t end_start = clock();
    auto gemm_time = double(spmm_times);
    auto all_time = double(end_start - total_start)  / CLOCKS_PER_SEC;
    final_gpu(); 
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

void parseCommandLine(int argc, char** const argv)
{
  int ret;
  optind = 1;

  while ((ret = getopt(argc, argv, "f:l:b:i:n:a:h")) != -1) {
    switch (ret) {
      case 'f':
        inputFileName.assign(optarg);
        break;
      case 'a':
        batch = atoi(optarg);
        break;
      case 'l':
        layer = atoi(optarg);
        break;
      case 'n':
        neuron = atoi(optarg);
        break;
      case 'i':
        input = atoi(optarg);
        break;
      case 'b':
        bias = atoi(optarg);
        break;
      case 'h':
        std::cout << "./inference -f <file-path> -i <input> -a <#batches> -n <#neurons> -l <#layers> -b <bias>" << std::endl;
        break;  
     default:
        assert(0 && "Should not reach here!!");
        break;
    }
  }
} // parseCommandLine
