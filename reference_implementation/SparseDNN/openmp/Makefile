# ----- Make Macros -----

CXX = nvc++ 
CXXFLAGS = -std=c++11 -mp=gpu -Minfo=mp
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
