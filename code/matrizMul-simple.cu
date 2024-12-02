#include <stdio.h>
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
#define TINT(ts,te) ( (double) 1000.*( (te).tv_sec - (ts).tv_sec ) + ( (te).tv_nsec - (ts).tv_nsec )/(double) 1.e6 )

// Número máximo de threads por cada dimensión del bloque
#define MAX_TH_PER_BLOCK_DIM 32

// Dimensiones por defecto de las matrices
#define ROWS_DEF 1000
#define COLS_DEF 1000

// Número de threads por cada dimensión del bloque por defecto
#define TPB_X_DEF 4
#define TPB_Y_DEF 4

// Tipo de datos
typedef float basetype;

void check_memoria(size_t numElem);

/**
 * Código CUDA
 * Cada thread computa un elemento de la matriz C
 */
__global__ void
matrizMul(const basetype *A, const basetype *B, basetype *C, unsigned int rowsA, unsigned int colsA, unsigned int colsB)
{
    // Índices de fila y columna en la matriz C
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rowsA && col < colsB)
    {
        basetype sum = 0.0;
        for (unsigned int k = 0; k < colsA; ++k)
        {
            sum += A[row * colsA + k] * B[k * colsB + col];
        }
        C[row * colsB + col] = sum;
    }
}

/**
 * Función main en el host
 * argv[1]: número de filas de la matriz A
 * argv[2]: número de columnas de la matriz A
 * argv[3]: número de columnas de la matriz B
 * argv[4]: número de threads por bloque en la dimensión x
 * argv[5]: número de threads por bloque en la dimensión y
 */
int main(int argc, char *argv[])
{
    basetype *h_A = NULL, *h_B = NULL, *h_C = NULL, *h_C2 = NULL;
    basetype *d_A = NULL, *d_B = NULL, *d_C = NULL;
    size_t sizeA, sizeB, sizeC;
    unsigned int rowsA, colsA, rowsB, colsB;
    unsigned int tpb_x, tpb_y;
    struct timespec tstart, tend;
    double tint;

    // Dimensiones de las matrices
    rowsA = (argc > 1) ? atoi(argv[1]) : ROWS_DEF;
    colsA = (argc > 2) ? atoi(argv[2]) : COLS_DEF;
    rowsB = colsA; // Por definición, las columnas de A deben coincidir con las filas de B
    colsB = (argc > 3) ? atoi(argv[3]) : COLS_DEF;

    // Tamaños de las matrices en bytes
    sizeA = rowsA * colsA * sizeof(basetype);
    sizeB = rowsB * colsB * sizeof(basetype);
    sizeC = rowsA * colsB * sizeof(basetype);

    // Dimensiones del bloque de hilos
    tpb_x = (argc > 4) ? atoi(argv[4]) : TPB_X_DEF;
    tpb_y = (argc > 5) ? atoi(argv[5]) : TPB_Y_DEF;

    tpb_x = (tpb_x > MAX_TH_PER_BLOCK_DIM) ? MAX_TH_PER_BLOCK_DIM : tpb_x;
    tpb_y = (tpb_y > MAX_TH_PER_BLOCK_DIM) ? MAX_TH_PER_BLOCK_DIM : tpb_y;

    check_memoria(rowsA * colsA + rowsB * colsB + rowsA * colsB);

    // Configuración de Grid y Block
    dim3 threadsPerBlock(tpb_x, tpb_y, 1);
    dim3 blocksPerGrid((colsB + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (rowsA + threadsPerBlock.y - 1) / threadsPerBlock.y, 1);

    printf("Multiplicación de matrices A(%u,%u) * B(%u,%u) = C(%u,%u)\n", rowsA, colsA, rowsB, colsB, rowsA, colsB);
    printf("Grid de bloques (%u,%u), bloque de threads (%u,%u)\n",
           blocksPerGrid.x, blocksPerGrid.y, threadsPerBlock.x, threadsPerBlock.y);

    // Reserva memoria en el host
    h_A = (basetype *)malloc(sizeA);
    h_B = (basetype *)malloc(sizeB);
    h_C = (basetype *)malloc(sizeC);
    h_C2 = (basetype *)malloc(sizeC);

    if (!h_A || !h_B || !h_C || !h_C2)
    {
        fprintf(stderr, "Error reservando memoria en el host\n");
        exit(EXIT_FAILURE);
    }

    // Inicializa las matrices
    for (unsigned int i = 0; i < rowsA * colsA; ++i)
        h_A[i] = rand() / (basetype)RAND_MAX;
    for (unsigned int i = 0; i < rowsB * colsB; ++i)
        h_B[i] = rand() / (basetype)RAND_MAX;

    // Host: multiplicación de matrices
    TSET(tstart);
    for (unsigned int i = 0; i < rowsA; ++i)
        for (unsigned int j = 0; j < colsB; ++j)
        {
            basetype sum = 0.0;
            for (unsigned int k = 0; k < colsA; ++k)
                sum += h_A[i * colsA + k] * h_B[k * colsB + j];
            h_C[i * colsB + j] = sum;
        }
    TSET(tend);
    tint = TINT(tstart, tend);
    printf("HOST: Tiempo multiplicación: %lf ms\n", tint);

    // GPU: reserva memoria
    checkError(cudaMalloc((void **)&d_A, sizeA));
    checkError(cudaMalloc((void **)&d_B, sizeB));
    checkError(cudaMalloc((void **)&d_C, sizeC));

    // Copia matrices al dispositivo
    checkError(cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice));
    checkError(cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice));

    // Lanza el kernel
    TSET(tstart);
    matrizMul<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, rowsA, colsA, colsB);
    checkError(cudaDeviceSynchronize());
    TSET(tend);
    tint = TINT(tstart, tend);
    printf("DEVICE: Tiempo multiplicación: %lf ms\n", tint);

    // Copia el resultado de vuelta
    checkError(cudaMemcpy(h_C2, d_C, sizeC, cudaMemcpyDeviceToHost));

    // Verifica la multiplicación
    for (unsigned int i = 0; i < rowsA * colsB; ++i)
    {
        if (fabs(h_C[i] - h_C2[i]) > 1e-3)
        {
            fprintf(stderr, "Error en el resultado en el índice %u\n", i);
            exit(EXIT_FAILURE);
        }
    }

    printf("Multiplicación correcta.\n");

    // Libera memoria
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C2);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}

void check_memoria(size_t numElem)
{
    cudaDeviceProp prop;
    checkError(cudaGetDeviceProperties(&prop, 0));

    size_t gmem = prop.totalGlobalMem;
    double gib = 1073741824.0;

    printf("Memoria global disponible: %g GiB\n", gmem / gib);
    if (gmem < numElem * sizeof(basetype))
    {
        fprintf(stderr, "No hay suficiente memoria en la GPU\n");
        exit(EXIT_FAILURE);
    }
}
