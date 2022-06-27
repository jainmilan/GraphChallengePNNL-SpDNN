# ----- Make Macros -----

CXX = g++
CXXFLAGS = -std=c++11 -fopenmp
OPTFLAGS = -O3

NVCC = nvcc
NVCCFLAGS = -lineinfo -O3 -std=c++11 -gencode arch=compute_80,code=sm_80 -ccbin=g++ -Xcompiler -fopenmp -I${CUDA_HOME}/include -L${CUDA_HOME}/lib64 -lcudart

LD_FLAGS = -ccbin=${CXX} -Xcompiler -fopenmp -lcusparse_static

TARGETS = inference
OBJECTS = cusparse.o

# ----- Make Rules -----

all:	$(TARGETS)

%.o : %.cu vars.h
	${NVCC} ${NVCCFLAGS} $< -c -o $@

inference: $(OBJECTS)
	$(NVCC) -o $@ $(OBJECTS) $(LD_FLAGS)

clean:
	rm -f $(TARGETS) *.o *.o.* *.txt *.bin core *.html *.xml
