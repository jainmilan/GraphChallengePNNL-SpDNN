# ----- Make Macros -----

ENABLE_GPU=0
ifeq ($(ENABLE_GPU),1)
CXX = nvc++ 
CXXFLAGS = -std=c++11 -DINDPREC=int -DVALPREC=float -DFEATPREC=float -mp=gpu -Minfo=mp
else
CXX = g++ 
CXXFLAGS = -g -std=c++11 -DUSE_OMP_HOST -DINDPREC=int -DVALPREC=float -DFEATPREC=float -fopenmp
endif
OPTFLAGS = -O3

TARGETS = inference
OBJECTS = openmp.o

# ----- Make Rules -----

all:	$(TARGETS)

%.o : %.cpp vars.h
	$(CXX) $(OPTFLAGS) $(CXXFLAGS) $< -c -o $@

inference: $(OBJECTS)
	$(CXX) $(OPTFLAGS) $(CXXFLAGS) -o $@ $(OBJECTS) $(LDFLAGS)

clean:
	rm -f $(TARGETS) *.o *.o.* *.txt *.bin core *.html *.xml
