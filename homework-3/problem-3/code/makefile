CC      = gcc
CFLAGS  = -Wall -O2

TARGET  = test_viewer
SRCS    = main.c draw_shape.c
OBJS    = $(SRCS:.c=.o)

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $(OBJS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET) output.ppm

.PHONY: all clean
