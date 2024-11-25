#include <stdio.h>

void printDevProp(cudaDeviceProp devProp) {

	// Nombre del dispositivo
	printf("Nombre del dispositivo: %s\n", devProp.name);

	// La computer capability (major.minor computer capability).
	printf("Major revision number: %d\n", devProp.major);
	printf("Minor revision number: %d\n", devProp.minor);

	// El numero de multiprocesadores (SMs) en el dispositivo.
	printf("Multiprocessor count: %d\n", devProp.multiProcessorCount);

	// El maximo numero de CUDA threads residentes por SM.
	printf("Max threads per SM: %d\n", devProp.maxThreadsPerMultiProcessor);

	// El tamano maximo de cada dimension de un Grid.
	printf("Max dimension size of a thread block (x,y,z): (%d, %d, %d)\n", devProp.maxGridSize[0], devProp.maxGridSize[1], devProp.maxGridSize[2]);

	// El maximo numero de hilos por bloque.
	printf("Max threads per block: %d\n", devProp.maxThreadsPerBlock);

	// El tamano maximo de cada dimension de un bloque
	printf("Max dimension size of a block (x, y, z): (%d, %d, %d)\n", devProp.maxThreadsDim[0], devProp.maxThreadsDim[1], devProp.maxThreadsDim[2]);

	// El numero de registros de 32 bits disponibles por SM.
	printf("Registers per block: %d\n", devProp.regsPerMultiprocessor);

	// El numero de registros de 32 bits disponibles por bloque
	printf("Registers per block: %d\n", devProp.regsPerBlock);

	// El tamano de la memoria compartida disponible por procesador.
	printf("Shared memory per multiprocessor: %zu\n", devProp.sharedMemPerMultiprocessor);

	// El tamano de la memoria compartida disponible por bloque.
	printf("Shared memory per block: %zu\n", devProp.sharedMemPerBlock);

	// El tamano de la memoria global disponible en el dispositivo.
	printf("Total global memory: %zu\n", devProp.totalGlobalMem);

	// La frecuencia pico de la memoria
	printf("Memory clock rate: %d\n", devProp.memoryClockRate);

	// El ancho del bus de memoria global.
	printf("Memory bus width: %d\n", devProp.memoryBusWidth);

	// BWpeak (GiB/s) = (F*(ancho/8)*2)/1e9
	printf("Peak memory bandwidth (GB/s): %f\n", 2.0 * devProp.memoryClockRate * (devProp.memoryBusWidth / 8) / 1.0e6);

	// Obten el numero total de CUDA cores, a partir del numero de SMs y del numero de cores por SM en la arquitectura.
	int cores = -1;
	int mp = devProp.multiProcessorCount;

	switch (devProp.major) {
		case 8: // Volta
			cores = mp * 64;
			break;
		case 7: // Turing
			cores = mp * 64;
			break;
		case 6: // Pascal
			cores = mp * 128;
			break;
		default: // No matching
			cores = -1;
			break;
	}

	printf("Number of CUDA cores: %d\n", cores);

}

int main(int argc, char *argv[]) {

	int numDevs;
	cudaDeviceProp prop;
	cudaError_t error;

	error = cudaGetDeviceCount(&numDevs);

	if(error != cudaSuccess) {

		fprintf(stderr, "Error obteniendo numero de dispositivos: %s en %s linea %d\n", cudaGetErrorString(error), __FILE__, __LINE__);
		exit(EXIT_FAILURE);

	}

	printf("Numero de dispositivos = %d\n", numDevs);

	for(int i=0; i < numDevs; i++) {

		error = cudaGetDeviceProperties(&prop, i);
		if(error != cudaSuccess) {

			fprintf(stderr, "Error obteniendo propiedades del dispositivo %d: %s en %s linea %d\n", i, cudaGetErrorString(error), __FILE__, __LINE__);
			exit(EXIT_FAILURE);
		}

		printf("\nDispositivo #%d\n", i);
		printDevProp(prop);

	}

	return(EXIT_SUCCESS);

}
