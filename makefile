# Makefile

NVCC = nvcc
CFLAGS = -arch=sm_50 -O2
TARGET = ray_tracer

all: $(TARGET)

$(TARGET): main.cu scene.cu camera.cu
	$(NVCC) $(CFLAGS) -o $(TARGET) main.cu scene.cu camera.cu

clean:
	rm -f $(TARGET)
