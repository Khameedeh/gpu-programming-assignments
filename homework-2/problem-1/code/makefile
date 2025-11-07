

CC = gcc
CFLAGS = -Wall -Wextra -pthread -O2

TARGET = producer_consumer

SRC = producer_consumer.c

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $(TARGET) $(SRC)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)

rebuild: clean all
