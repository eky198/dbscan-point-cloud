CXX = g++
CXXFLAGS = -Wall -O3 -std=c++17 -m64 -I. -fopenmp -Wno-unknown-pragmas
# CXXFLAGS = -Wall -g -std=c++17 -m64 -I. -fopenmp -Wno-unknown-pragmas
NVCC = nvcc
NVCCFLAGS = -O3 -m64 -Xcompiler -fopenmp --gpu-architecture compute_61 -ccbin /usr/bin/gcc

all: dbscan dbscan_seq dbscan_cu

dbscan: dbscan.o
	$(CXX) $(CXXFLAGS) -o $@ dbscan.o

dbscan_seq: dbscan_seq.o
	$(CXX) $(CXXFLAGS) -o $@ dbscan_seq.o

dbscan_cu: dbscan_cu.o
	$(CXX) $(CXXFLAGS) -o $@ dbscan_cu.o -L/usr/local/cuda-11.7/lib64/ -lcudart

dbscan.o: dbscan.cpp dbscan.h
	$(CXX) $(CXXFLAGS) -c $<

dbscan_seq.o: dbscan_seq.cpp dbscan.h
	$(CXX) $(CXXFLAGS) -c $<

dbscan_cu.o: dbscan.cu dbscan.h
	$(NVCC) $< $(NVCCFLAGS) -c -o $@

clean:
	/bin/rm -rf *~ *.o $(APP_NAME) *.class