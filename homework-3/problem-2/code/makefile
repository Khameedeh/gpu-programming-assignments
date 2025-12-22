NVCC = nvcc
NVCC_FLAGS = -O2 -arch=sm_75

TARGET = masked_computation
SRCS = main.cu masked_computation.cu
OBJS = main.o masked_computation.o

all: $(TARGET)

$(TARGET): $(OBJS)
	$(NVCC) $(NVCC_FLAGS) -o $@ $(OBJS)

main.o: main.cu masked_computation.h
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

masked_computation.o: masked_computation.cu masked_computation.h
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)

.PHONY: all clean
