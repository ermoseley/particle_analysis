# Build part2cube from mini-ramses-dev (particle output -> 3D density cube)
FC = gfortran
FFLAGS = -O2 -Wall
TARGET = part2cube

all: $(TARGET)

$(TARGET): part2cube.f90
	$(FC) $(FFLAGS) -o $(TARGET) part2cube.f90

clean:
	rm -f $(TARGET) *.o *.mod

.PHONY: all clean
