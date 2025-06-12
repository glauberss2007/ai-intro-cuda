#include <iostream>
#include <math.h>

// Função para somar os elementos de dois arrays
void add(int n, float *x, float *y)
{
  for (int i = 0; i < n; i++)
      y[i] = x[i] + y[i];
}

int main(void)
{
  int N = 1<<20; // 1M elementos (2^20 = 1.048.576)

  // Alocação dinâmica de memória para os arrays x e y
  float *x = new float[N];
  float *y = new float[N];

  // Inicialização dos arrays x e y no host (CPU)
  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  // Executa a função de adição para 1M elementos na CPU
  add(N, x, y);

  // Verifica erros (todos os valores devem ser 3.0f)
  float maxError = 0.0f;
  for (int i = 0; i < N; i++)
    maxError = fmax(maxError, fabs(y[i]-3.0f));
  std::cout << "Erro máximo: " << maxError << std::endl;

  // Libera a memória alocada
  delete [] x;
  delete [] y;

  return 0;
}
