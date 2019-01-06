#define N 9
#include <time.h>
#include <math.h>
#include <stdio.h>
/* #include <immintrin.h> */
#include <time.h>
#include <sys/time.h> // for timing
#include <pthread.h>
#include <immintrin.h>


struct thread_data
{
  unsigned int thread_id;
  float *W;
  float *U;
  float a;
  long end;
};

double now(){
  // Retourne l'heure actuelle en secondes
  struct timeval t; double f_t;
  gettimeofday(&t, NULL);
  f_t = t.tv_usec; f_t = f_t/1000000.0; f_t +=t.tv_sec;
  return f_t;
}

/* https://www.codeproject.com/Articles/874396/%2FArticles%2F874396%2FCrunching-Numbers-with-AVX-and-AVX */

//Ici on dÃ©clare tous nos vecteurs en prenant soin de les aligner
float U[N] __attribute__((aligned(16)));
float W[N] __attribute__((aligned(16)));
float a;
int k, n;

float gm(float*U, float*W, float a, int k, int n){
  float r = 0;
  float sum_w = 0;
  for ( int i = 0; i < n; i++){
    r += pow((W[i] * U[i] - a), k);
    sum_w += W[i];
  }
  return(r/sum_w);
}

/* Sum : https://www.moreno.marzolla.name/teaching/high-performance-computing/2018-2019/L08-SIMD.pdf */

float vect_gm(float*U, float*W, float a, int k, int n){
  __m256 mm_U, mm_W, mm_a, mm_step_r, mm_r, mm_sum_w;
  mm_a = _mm256_set1_ps(a);
  mm_r = _mm256_setzero_ps();
  mm_sum_w = _mm256_setzero_ps();

  for ( int i = 0; i < n/8; i++ ){
    mm_U = _mm256_load_ps( &U[8*i] );
    mm_W = _mm256_load_ps( &W[8*i] );

    mm_U = _mm256_fmsub_ps(mm_U, mm_W, mm_a);
    mm_step_r = mm_U;
    /* Replace by fast exponentation? */
    for (int j=1; j<k; j++){
      mm_step_r = _mm256_mul_ps(mm_step_r, mm_U);
    }
    mm_r = _mm256_add_ps(mm_r, mm_step_r);

    /* W sum */
    mm_sum_w = _mm256_add_ps(mm_sum_w, mm_W);
  }

  float* sum_r = (float*)&mm_r;
  float* sum_w = (float*)&mm_sum_w;
  float r = 0.;
  float w = 0.;

  // Edge case where n is not a multiple of 8
  for ( int i = 8*(n/8) ; i < n ; i++ ){
    r += pow((W[i] * U[i] - a), k);
    w += W[i];
  }

  for ( int i = 0; i < 8; i++){
    r += sum_r[i];
    w += sum_w[i];
  }
  return r/w;
}

/* void parallel_gm(float *U, float *W, float a, int k, int n, int mode, int nb_threads){ */
/*  /\* https://www.geeksforgeeks.org/sum-array-using-pthreads/   *\/ */

/*   pthread_t threads[nb_threads]; */

/* } */


void init(){
  unsigned int i;
  for( i = 0; i < N; i++ ){
    U[i] = (float)rand () / RAND_MAX ;
    W[i] = (float)rand () / RAND_MAX ;
  }
  /* for( i = 0; i < N; i++ ){ */
  /*   U[i] = (float) i + 1; */
  /*   W[i] = 1.; */
  /* } */

}

int main(){
  a = 0;
  int k = 4;
  init();

  double t;
  float rs, rv;


  t = now();
  rs = gm(U, W, a, k, N);
  t = now()-t;
  printf("S = %10.9f Temps du code scalaire   : %f seconde(s)\n",rs,t);

  t = now();
  rv = vect_gm(U, W, a, k, N);
  t = now()-t;
  printf("S = %10.9f Temps du code vectoriel 1: %f seconde(s)\n",rv,t);


  return 0;

}
