#include <iostream>
#include <math.h>

// Função kernel para somar os elementos de dois arrays
__global__
void add(int n, float *x, float *y)
{
  for (int i = 0; i < n; i++)
    y[i] = x[i] + y[i];
}

int main(void)
{
  int N = 1<<20; // 1M elementos (2^20 = 1.048.576)

  float *x, *y;

  // Aloca Memória Unificada - acessível pela CPU ou GPU
  cudaMallocManaged(&x, N*sizeof(float));
  cudaMallocManaged(&y, N*sizeof(float));

  // Inicializa os arrays x e y no host (CPU)
  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  // Executa o kernel para 1M elementos na GPU
  add<<<1, 1>>>(N, x, y);

  // Aguarda a GPU terminar antes de acessar no host
  cudaDeviceSynchronize();

  // Verifica erros (todos os valores devem ser 3.0f)
  float maxError = 0.0f;
  for (int i = 0; i < N; i++)
    maxError = fmax(maxError, fabs(y[i]-3.0f));
  std::cout << "Erro máximo: " << maxError << std::endl;

  // Libera a memória
  cudaFree(x);
  cudaFree(y);
  
  return 0;
}
