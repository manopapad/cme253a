.PHONY:
	default clean

default: el3.exec

%.exec: %.cu
	nvcc --std=c++11 -O3 -o $@ $< -lhdf5

clean:
	$(RM) *.hdf *.exec *.dat
