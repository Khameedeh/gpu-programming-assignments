NVCC = nvcc
NVCC_FLAGS = -O2 -arch=sm_75
CC = gcc
CFLAGS = -O2

TARGET = rasterization
CU_SRCS = main.cu rasterization.cu
C_SRCS = draw_shape.c
CU_OBJS = main.o rasterization.o
C_OBJS = draw_shape.o

all: $(TARGET)

$(TARGET): $(CU_OBJS) $(C_OBJS)
	$(NVCC) $(NVCC_FLAGS) -o $@ $(CU_OBJS) $(C_OBJS)

main.o: main.cu rasterization.h
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

rasterization.o: rasterization.cu rasterization.h
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

draw_shape.o: draw_shape.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(CU_OBJS) $(C_OBJS) $(TARGET) output.ppm

.PHONY: all clean