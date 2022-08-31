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
static INDPREC nparts;
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
FEATPREC *currfeat_ptr;
FEATPREC *nextfeat_ptr; 

INDPREC *active;   
INDPREC *active_ptr;   
static INDPREC pbatch;

double timeio;
double timetot;
double timeinfer;
double timebalance = 0.0;
double timekernel = 0.0;
double timecopy = 0.0;

INDPREC *numbatch;
INDPREC *batchdispl;

INDPREC *categories;
INDPREC *mycategories;

static INDPREC saved=0;
INDPREC offset=0;

FEATPREC ReLU(FEATPREC x){
    return x<0.0?0.0:x>32.0?32.0:x;
 };

void setup_gpu() {
    for(INDPREC l = 0; l < layer; l++){
#if defined(USE_OMP_HOST)
#else
#pragma omp target enter data map(alloc:csrdispl[l][0:neuron+1])
#pragma omp target enter data map(alloc:csrindex[l][0:csrdispl[l][neuron]])
#pragma omp target enter data map(alloc:csrvalue[l][0:csrdispl[l][neuron]])
#endif
    }
#if defined(USE_OMP_HOST)
#else
#pragma omp target enter data map(alloc:currfeat_ptr[:pbatch*neuron])
#pragma omp target enter data map(alloc:nextfeat_ptr[:pbatch*neuron])
#pragma omp target enter data map(alloc:active_ptr[:pbatch])
#endif
}

void final_gpu() {
    for(INDPREC l = 0; l < layer; l++){
#if defined(USE_OMP_HOST)
#else
#pragma omp target exit data map(delete:csrdispl[l][0:neuron+1])
#pragma omp target exit data map(delete:csrindex[l][0:csrdispl[l][neuron]])
#pragma omp target exit data map(delete:csrvalue[l][0:csrdispl[l][neuron]])
#endif
    }
#if defined(USE_OMP_HOST)
#else
#pragma omp target exit data map(delete:currfeat_ptr[:pbatch*neuron])
#pragma omp target exit data map(delete:nextfeat_ptr[:pbatch*neuron])
#pragma omp target exit data map(delete:active_ptr[:pbatch])
#endif
}

double kernel_spmm(INDPREC l) {

   std::memset(active_ptr, 0, sizeof(INDPREC)*pbatch);
   std::memset(nextfeat_ptr, 0, sizeof(FEATPREC)*pbatch*neuron);

#if defined(USE_OMP_HOST)
#else
#pragma omp target update to(currfeat_ptr[:pbatch*neuron], nextfeat_ptr[:pbatch*neuron], active_ptr[:pbatch])
#endif

   double t0 = omp_get_wtime();
#if defined(USE_OMP_HOST)
#pragma omp parallel for default(shared) \
   collapse(2)
#else
#pragma omp target teams loop \
   collapse(2) \
   map(to: currfeat_ptr[:pbatch*neuron], nextfeat_ptr[:pbatch*neuron], active_ptr[:pbatch]) \
   map(to: csrdispl[l][0:neuron+1], csrindex[l][0:csrdispl[l][neuron]], csrvalue[l][0:csrdispl[l][neuron]])
#endif
   for (INDPREC i = 0; i < pbatch; i++) {
     for (INDPREC j = 0; j < neuron; j++) {
       VALPREC result = 0;
       for (INDPREC p = csrdispl[l][j]; p < csrdispl[l][j+1]; p++) {
         const INDPREC k = csrindex[l][p];
         result += csrvalue[l][p] * currfeat_ptr[i * neuron + k];
       }
       nextfeat_ptr[i * neuron + j] = ReLU(result + bias);
       if (nextfeat_ptr[i * neuron + j])
         active_ptr[i] += 1;
     }
   }
   double t1 = omp_get_wtime();

#if defined(USE_OMP_HOST)
#else
#pragma omp target update from (nextfeat_ptr[:pbatch*neuron])
#pragma omp target update from (active_ptr[:pbatch])
#endif  

   INDPREC feature = 0;
   for(INDPREC i = 0; i < pbatch; i++) {
     if(active_ptr[i]) {
       for(INDPREC j = 0; j < neuron; j++) {
         nextfeat_ptr[feature * neuron + j] = nextfeat_ptr[i * neuron + j];
       }
       mycategories[feature] = mycategories[i];
       feature++;
     }
   }

   pbatch = feature;
   FEATPREC *tempfeat = currfeat_ptr;
   currfeat_ptr = nextfeat_ptr;
   nextfeat_ptr = tempfeat;

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
    csrdispl = new INDPREC*[layer];
    csrindex = new INDPREC*[layer];
    csrvalue = new VALPREC*[layer];
    currfeat = new FEATPREC[neuron*(long)batch];
    nextfeat = new FEATPREC[neuron*(long)batch];
  
    active = new int[batch];
    categories = new INDPREC[batch];

    printf("%d neurons, %d layers", neuron, layer) ;
    printf("\n");
    printf("READING WEIGHTS\n");
    readweights();
    printf("READING INPUT\n");
    readinput();

    for(int k = 0; k < batch; k++){
      active[k] = neuron;
      categories[k] = k;
    }
    
    double spmm_times = 0; 
    clock_t total_start = clock();

    pbatch = batch / nparts;
    
    for(offset = 0; offset < batch; offset += pbatch) {     
      
      mycategories = new INDPREC[pbatch];
      for(INDPREC k = 0; k < pbatch; k++) {
        mycategories[k] = k + offset;
      }

      currfeat_ptr = currfeat + offset * neuron;
      nextfeat_ptr = nextfeat + offset * neuron;
      active_ptr = active + offset;

      printf("INFERENCE......\n");
      setup_gpu();
      printf("for %d layers of batch size %d......\n", layer, pbatch);
      for(int i = 0; i < layer; ++i) {        
        printf("[%d], [%d]", i, pbatch);
        fflush(stdout);
        auto t = kernel_spmm(i);
        spmm_times += double(t);
        printf(":(%lf)\n", t);
        fflush(stdout);
      }

      for (INDPREC k = 0; k < pbatch; k++) {
        categories[saved+k] = mycategories[k];
      }

      saved += pbatch;
      pbatch = batch / nparts;
      final_gpu();
    }
    
    pbatch = saved;

    clock_t end_start = clock();
    auto gemm_time = double(spmm_times);
    auto all_time = double(end_start - total_start)  / CLOCKS_PER_SEC;
    printf("Inference time (#%d parts): %lfs, %lfs, %f TTEPS\n", nparts, gemm_time, all_time, long((long)batch * (long)neuron * 32 * layer) / gemm_time / 1e12);

    std::string slayer = std::to_string(layer);
    std::string sneuron = std::to_string(neuron);
    std::string sbatch = std::to_string(batch);
    std::string snparts = std::to_string(nparts);
    
    std::string outfilename = outFilePath + "/" + slayer + "-" + sneuron + "-" + sbatch + "-" + snparts + "parts-results.txt";

    std::cout << "Storing output results in: " << outfilename << std::endl;
    std::cout << "Saved features: " << saved << std::endl;
    std::ofstream ofile(outfilename);
    if (ofile.is_open())
    {
      for (INDPREC i = 0; i < pbatch; i++)
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

  while ((ret = getopt(argc, argv, "f:o:l:b:i:n:a:p:h")) != -1) {
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
      case 'p':
        nparts = atoi(optarg);
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
        std::cout << "./inference -f <file-path> -i <input> -o <output path> -a <#batches> -g <#gpus> -n <#neurons> -l <#layers> -b <bias>" << std::endl;
        break;  
     default:
        assert(0 && "Should not reach here!!");
        break;
    }
  }
} // parseCommandLine
