.PHONY:
	clean

%.exec: %.cu
	nvcc --std=c++11 -arch=sm_53 -O3 -o $@ $< -lhdf5

clean:
	$(RM) *.hdf *.exec *.dat
