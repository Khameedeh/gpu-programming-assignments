NVCC = nvcc
NVCC_FLAGS = -O2 -arch=sm_75

TARGET = color_matrix
SRCS = main.cu color_the_matrix.cu
OBJS = main.o color_the_matrix.o

all: $(TARGET)

$(TARGET): $(OBJS)
	$(NVCC) $(NVCC_FLAGS) -o $@ $(OBJS)

main.o: main.cu color_the_matrix.h
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

color_the_matrix.o: color_the_matrix.cu color_the_matrix.h
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET) *.ppm

.PHONY: all clean
