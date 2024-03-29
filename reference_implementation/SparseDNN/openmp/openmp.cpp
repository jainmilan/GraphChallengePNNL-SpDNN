#include <stdio.h>
#include <iostream>
#include <vector>
#include <string>
#include <cstring>
#include <fstream>

#include <omp.h>
#include "vars.h"


static char *dataset;
static INDPREC neuron;
static INDPREC layer;
static INDPREC batch;
static INDPREC input;
static VALPREC bias;
static std::string inputFileName;
static std::string outFilePath;

int blocksize;

long totnz;
INDPREC **csrdispl;   
INDPREC **csrindex;
VALPREC **csrvalue;

FEATPREC *currfeat;
FEATPREC *nextfeat; 

INDPREC *active;   
INDPREC crbatch;

double timeio;
double timetot;
double timeinfer;
double timebalance = 0.0;
double timekernel = 0.0;
double timecopy = 0.0;

INDPREC *numbatch;
INDPREC *batchdispl;

INDPREC *categories;

FEATPREC ReLU(FEATPREC x){
    return x<0.0?0.0:x>32.0?32.0:x;
 };

void setup_gpu() {
    for(INDPREC l = 0; l < layer; l++){
#if defined(USE_OMP_HOST)
#else
#pragma omp target enter data map(to:csrdispl[l][0:neuron+1])
#pragma omp target enter data map(to:csrindex[l][0:csrdispl[l][neuron]])
#pragma omp target enter data map(to:csrvalue[l][0:csrdispl[l][neuron]])
#endif
    }
#if defined(USE_OMP_HOST)
#else
#pragma omp target enter data map(to:currfeat[0:crbatch*neuron])
#pragma omp target enter data map(to:nextfeat[0:crbatch*neuron])
#pragma omp target enter data map(to:active[0:crbatch])
#endif
}

void final_gpu() {
    for(INDPREC l = 0; l < layer; l++){
#if defined(USE_OMP_HOST)
#else
#pragma omp target exit data map(from:csrdispl[l][0:neuron+1])
#pragma omp target exit data map(from:csrindex[l][0:csrdispl[l][neuron]])
#pragma omp target exit data map(from:csrvalue[l][0:csrdispl[l][neuron]])
#endif
    }
#if defined(USE_OMP_HOST)
#else
#pragma omp target exit data map(from:currfeat[0:crbatch*neuron])
#pragma omp target exit data map(from:nextfeat[0:crbatch*neuron])
#pragma omp target exit data map(from:active[0:crbatch])
#endif
}

double kernel_spmm(INDPREC l) {

  std::memset(nextfeat, 0, sizeof(FEATPREC)*crbatch*neuron);
  std::memset(active, 0, sizeof(INDPREC)*crbatch);

#if defined(USE_OMP_HOST)
#else
#pragma omp target update to(currfeat[0:crbatch*neuron], nextfeat[0:crbatch*neuron], active[0:crbatch])
#endif

   double t0 = omp_get_wtime();
#if defined(USE_OMP_HOST)
#pragma omp parallel for default(shared) \
   collapse(2)
#else
#pragma omp target teams loop \
  collapse(2) \
  map(to: currfeat[0:crbatch*neuron], nextfeat[0:crbatch*neuron], active[0:crbatch]) \
  map(to: csrdispl[l][0:neuron+1], csrindex[l][0:csrdispl[l][neuron]], csrvalue[l][0:csrdispl[l][neuron]])
#endif
   for (INDPREC i = 0; i < crbatch; i++) {
     for (INDPREC j = 0; j < neuron; j++) {
       VALPREC result = 0;
       for (INDPREC p = csrdispl[l][j]; p < csrdispl[l][j+1]; p++) {
         const INDPREC k = csrindex[l][p];
         result += csrvalue[l][p] * currfeat[i * neuron + k];
       }
       nextfeat[i * neuron + j] = ReLU(result + bias);
       if (nextfeat[i * neuron + j])
         active[i] += 1;
     }
   }
   double t1 = omp_get_wtime();

#if defined(USE_OMP_HOST)
#else
#pragma omp target update from(nextfeat[0:crbatch*neuron])
#pragma omp target update from(active[0:crbatch])
#endif                                       
    
   INDPREC feature = 0;
    for(INDPREC i = 0; i < crbatch; i++) {
        if(active[i]) {
            for(INDPREC j = 0; j < neuron; j++) {
                nextfeat[feature * neuron + j] = nextfeat[i * neuron + j];
            }
            categories[feature] = categories[i];
            feature++;
        }
    }

    crbatch = feature;
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
      std::cout << "Input file path not found...exiting!!!" << std::endl;
      return 0;
    }

    if (!outFilePath.empty())
    {
      std::ifstream f(outFilePath.c_str());
      if (!f.good())
      {
        std::cout << "Output file path not found...exiting!!!" << std::endl;
        return 0;
      }
    }

    dataset = (char*)inputFileName.c_str();
    std::cout << "Input data path: " << dataset << std::endl;
    std::cout << "#Neurons: " << neuron << std::endl;
    std::cout << "#Layers: " << layer << std::endl;
    std::cout << "#Batches: " << batch << std::endl;
    std::cout << "#Inputs: " << input << std::endl;
    std::cout << "Bias: " << bias << std::endl;

    /*
    dataset = "/lus/grand/projects/GRACE/spdnn/dataset";///qfs/projects/pacer/leeh736/dataset"; 
    char *chartemp;
    neuron = 65536;
    layer = 1920;
    batch = 60000;
    input = 392191985; // 392191985; // 98858913; // 25019051; //6374505;
    bias = 0;
    */
    crbatch = batch;
    csrdispl = new INDPREC*[layer];
    csrindex = new INDPREC*[layer];
    csrvalue = new VALPREC*[layer];
    currfeat = new FEATPREC[neuron*(long)crbatch];
    nextfeat = new FEATPREC[neuron*(long)crbatch];
  
    active = new int [crbatch];
    categories = new INDPREC[crbatch];

    printf("%d neurons, %d layers", neuron, layer) ;
    printf("\n");
    printf("READING WEIGHTS\n");
    readweights();
    printf("READING INPUT\n");
    readinput();

    for(int k = 0; k < crbatch; k++){
      active[k] = neuron;
      categories[k] = k;
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
	 
    std::string slayer = std::to_string(layer);
    std::string sneuron = std::to_string(neuron);
    std::string sbatch = std::to_string(batch);
#if defined(USE_OMP_HOST)
    std::string outfilename = outFilePath + "/" + slayer + "-" + sneuron + "-" + sbatch + "cpu-results.txt";
#else
    std::string outfilename = outFilePath + "/" + slayer + "-" + sneuron + "-" + sbatch + "gpu-results.txt";
#endif

    std::cout << "Storing output results in: " << outfilename << std::endl;
    std::ofstream ofile(outfilename);
    if (ofile.is_open())
    {
      for (INDPREC i = 0; i < crbatch; i++)
        ofile << categories[i] + 1 << "\n";
    }
    ofile.close();

    return 0;
}

void readweights(){
    totnz = 0;
    for(INDPREC l = 0; l < layer; l++){
        INDPREC rownz[neuron];
        for(INDPREC n = 0; n < neuron; n++)
            rownz[n] = 32;
        csrdispl[l] = new INDPREC[neuron+1];
        csrdispl[l][0] = 0;
        for(INDPREC n = 1; n < neuron+1; n++)
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
    for(INDPREC l = 0; l < layer; l++){
        INDPREC *row = new INDPREC[csrdispl[l][neuron]];
        INDPREC *col = new INDPREC[csrdispl[l][neuron]];
        VALPREC *val = new VALPREC[csrdispl[l][neuron]];
        fread(row, sizeof(INDPREC), csrdispl[l][neuron], weightf);
        fread(col, sizeof(INDPREC), csrdispl[l][neuron], weightf);
        fread(val, sizeof(VALPREC), csrdispl[l][neuron],weightf);
        INDPREC rownz[neuron];
        for(INDPREC n = 0; n < neuron; n++)
            rownz[n] = 0;
        for(INDPREC n = 0; n < csrdispl[l][neuron]; n++){
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
    INDPREC *row = new INDPREC[input];
    INDPREC *col = new INDPREC[input];
    VALPREC *val = new VALPREC[input];
    fread(row,sizeof(INDPREC),input,inputf);
    fread(col,sizeof(INDPREC),input,inputf);
    fread(val,sizeof(VALPREC),input,inputf);
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

  while ((ret = getopt(argc, argv, "f:o:l:b:i:n:a:h")) != -1) {
    switch (ret) {
      case 'f':
        inputFileName.assign(optarg);
        break;
      case 'o':
        outFilePath.assign(optarg);
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
        bias = atof(optarg);
        break;
      case 'h':
        std::cout << "./inference -f <file-path> -i <input> -o <output path> -a <#batches> -n <#neurons> -l <#layers> -b <bias>" << std::endl;
        break;  
     default:
        assert(0 && "Should not reach here!!");
        break;
    }
  }
} // parseCommandLine
