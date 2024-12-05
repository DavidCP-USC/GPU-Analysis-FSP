/**
 * Suma dos vectores: C = A + B.
 */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define checkError(ans) { asserError((ans), __FILE__, __LINE__); }
inline void asserError(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"Error: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

#define TSET(time)  clock_gettime( CLOCK_MONOTONIC, &(time) )
#define TINT(ts,te) { ( (double) 1000.*( (te).tv_sec - (ts).tv_sec ) + ( (te).tv_nsec - (ts).tv_nsec )/(double) 1.e6 ) }

// Numero maximo de threads por bloque
#define MAX_TH_PER_BLOCK 1024

// Tamanho por defecto de los vectores
#define NELDEF 1000

// Numero de threads por bloque por defecto
#define TPBDEF 256

// Numwero de repeticiones
#define NREPDEF 1

// Tipo de datos
typedef float basetype;

/**
 * Codigo host
 */
__host__ void
h_vectorAdd(const basetype *A, const basetype *B, basetype *C, unsigned int numElements)
{
    for (unsigned int i = 0; i < numElements; ++i)
    {
        C[i] = A[i] + B[i];
    }
}

/**
 * Codigo CUDA
 */
__global__ void
vectorAdd(const basetype *A, const basetype *B, basetype *C, unsigned int numElements)
{
    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements)
    {
        C[i] = A[i] + B[i];
    }
}

/**
 * Funcion main en el host
 * Parametros: nElementos threadsPerBlock nreps
 */
int
main(int argc, char *argv[])
{
    basetype *A=NULL, *B=NULL, *C=NULL;
    unsigned int numElements = 0, tpb = 0, nreps=1;
    size_t size = 0;

    // Valores para la medida de tiempos
    struct timespec tstart, tend;
    double tint;

    // Tamanho de los vectores
    char *end = NULL;
    numElements = (argc > 1) ? strtoul(argv[1], &end, 10):NELDEF;
    // Tamanho de los vectores en bytes
    size = numElements * sizeof(basetype);

    // Numero de threads por bloque
    tpb = (argc > 2) ? atoi(argv[2]):TPBDEF;
    // Comprueba si es superior al máximo
    tpb = (tpb > MAX_TH_PER_BLOCK) ? TPBDEF:tpb;

    // Numero de repeticiones de la suma
    nreps = (argc > 3) ? atoi(argv[3]):NREPDEF;

    // Caracteristicas del Grid
    dim3 threadsPerBlock( tpb );
    // blocksPerGrid = ceil(numElements/threadsPerBlock)
    dim3 blocksPerGrid( (numElements + threadsPerBlock.x - 1) / threadsPerBlock.x );
    printf("Suma de vectores de %u elementos (%u reps), con %u bloques de %u threads\n",
      numElements, nreps, blocksPerGrid.x, threadsPerBlock.x);

    // Reserva memoria unificada
    TSET(tstart);
    checkError(cudaMallocManaged(&A, size));
    checkError(cudaMallocManaged(&B, size));
    checkError(cudaMallocManaged(&C, size));
    TSET(tend);
    tint = TINT(tstart, tend);
    printf("A: Tiempo reserva memoria unificada: %lf ms\n", tint);

    // Inicializa los vectores
    for (unsigned int i = 0; i < numElements; ++i)
    {
        A[i] = rand() / (basetype)RAND_MAX;
        B[i] = rand() / (basetype)RAND_MAX;
    }

    /*
    * Hace la suma en el host
    */
    TSET(tstart);
    for(unsigned int r = 0; r < nreps; ++r)
        h_vectorAdd(A, B, C, numElements);
    TSET(tend);
    tint = TINT(tstart, tend);
    printf("HOST: Tiempo para hacer %u sumas de vectores de tamaño %u: %lf ms\n", nreps, numElements, tint);

    /*
    * Hace la suma en el dispositivo
    */
    TSET(tstart);
    for(unsigned int r = 0; r < nreps; ++r) {
        vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, numElements);
        checkError(cudaDeviceSynchronize());
    }
    TSET(tend);
    tint = TINT(tstart, tend);
    printf("C: Tiempo ejecucion kernel GPU: %lf ms\n", tint);

    // Verifica que la suma es correcta
    for (unsigned int i = 0; i < numElements; ++i)
    {
        if (fabs(C[i] - (A[i] + B[i])) > 1e-5)
        {
            fprintf(stderr, "Verificacion de resultados falla en el elemento %d!\n", i);
            exit(EXIT_FAILURE);
        }
    }

    // Libera memoria unificada
    checkError(cudaFree(A));
    checkError(cudaFree(B));
    checkError(cudaFree(C));

    return 0;

}
