CXX = g++
CXXFLAGS = -Wall -O3 -std=c++17 -m64 -I. -fopenmp -Wno-unknown-pragmas
# CXXFLAGS = -Wall -g -std=c++17 -m64 -I. -fopenmp -Wno-unknown-pragmas

all: dbscan dbscan_seq

dbscan: dbscan.o
	$(CXX) $(CXXFLAGS) -o $@ dbscan.o

dbscan_seq: dbscan_seq.o
	$(CXX) $(CXXFLAGS) -o $@ dbscan_seq.o

dbscan.o: dbscan.cpp dbscan.h
	$(CXX) $(CXXFLAGS) -c $<

dbscan_seq.o: dbscan_seq.cpp dbscan.h
	$(CXX) $(CXXFLAGS) -c $<

clean:
	/bin/rm -rf *~ *.o $(APP_NAME) *.class