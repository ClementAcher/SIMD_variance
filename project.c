#define N 100000 
#include <time.h>
#include <math.h>
#include <stdio.h>
/* #include <immintrin.h> */
#include <time.h>
#include <sys/time.h> // for timing
#include <pthread.h>
#include <immintrin.h>



double now(){
  // Retourne l'heure actuelle en secondes
  struct timeval t; double f_t;
  gettimeofday(&t, NULL);
  f_t = t.tv_usec; f_t = f_t/1000000.0; f_t +=t.tv_sec;
  return f_t;
}

/* https://www.codeproject.com/Articles/874396/%2FArticles%2F874396%2FCrunching-Numbers-with-AVX-and-AVX */

//Ici on dÃ©clare tous nos vecteurs en prenant soin de les aligner
float U[N] __attribute__((aligned(32)));
float W[N] __attribute__((aligned(32)));
float a;
int k, n;

typedef struct result {
    float r;
    float sum_w;
} result_t;

result_t sliced_gm(float *U, float *W, float a, int k, int n, int start) {
    /* Computes separatly the two sums, for n values, starting at index start. 
     * Returning a raw structure instead of a pointer is worth it as long as this function is not
     * called too many times (in which case this is less expensive than malloc/free).
     * */
    float r = 0;
    float sum_w = 0;
    int j;
    for (int i = 0; i < n; i++){
        j = start + i;
        float p = pow((W[j] * U[j] - a), k);
        r += p;
        sum_w += W[j];
    }
    return (result_t){r, sum_w};
}

float gm(float*U, float*W, float a, int k, int n){
    result_t result = sliced_gm(U, W, a, k, n, 0);
    return result.r / result.sum_w;
}


/* Sum : https://www.moreno.marzolla.name/teaching/high-performance-computing/2018-2019/L08-SIMD.pdf */
result_t sliced_vect_gm(float *U, float *W, float a, int k, int n, int start) {
    /* Computes separately the two sums, for n values, starting at start, using AVX */
  __m256 mm_U, mm_W, mm_a, mm_step_r, mm_r, mm_sum_w;
  mm_a = _mm256_set1_ps(a);
  mm_r = _mm256_setzero_ps();
  mm_sum_w = _mm256_setzero_ps();

  for ( int i = 0; i < n/8; i++ ){
    mm_U = _mm256_load_ps( &U[start + 8*i] );
    mm_W = _mm256_load_ps( &W[start + 8*i] );

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
    r += pow((W[start + i] * U[start + i] - a), k);
    w += W[start + i];
  }

  for ( int i = 0; i < 8; i++){
    r += sum_r[i];
    w += sum_w[i];
  }
  return (result_t){r, w};
}

float vect_gm(float*U, float*W, float a, int k, int n){
    result_t result = sliced_vect_gm(U, W, a, k, n, 0);
    return result.r / result.sum_w;
}

float *p_w_sum;
float *p_u_w_sum;

typedef struct shared_data {
    float *W;
    float *U;
    float a;
    int k;
} shared_data_t;

typedef struct thread_data {
  unsigned int thread_id;
  int start;
  int slice_size;
  int mode;
  shared_data_t *s_data;
} thread_data_t;

void *worker_func(void *data) {
    // convert back data
    thread_data_t *d = (thread_data_t*) data;
    shared_data_t *s_d = d->s_data;
    // perform calculations
    result_t result;
    if (d->mode == 0)
        result = sliced_gm(s_d->U, s_d->W, s_d->a, s_d->k, d->slice_size, d->start);
    else
        result = sliced_vect_gm(s_d->U, s_d->W, s_d->a, s_d->k, d->slice_size, d->start);
    p_w_sum[d->thread_id] = result.sum_w;
    p_u_w_sum[d->thread_id] = result.r;
}

float parallel_gm(float *U, float *W, float a, int k, int n, int mode, int nb_threads) {
    int slice_size = n / nb_threads;
    int overflow = n % nb_threads;
    // allocate memory for partial sums
    p_w_sum = malloc(nb_threads * sizeof(float));
    p_u_w_sum = malloc(nb_threads * sizeof(float));
    // spawn all workers
    pthread_t threads[nb_threads];
    shared_data_t s_data = { W, U, a, k };
    for (int i = 0; i < nb_threads; i++) {
        thread_data_t *data = malloc(sizeof(thread_data_t));
        data->s_data = &s_data;
        data->thread_id = i;
        data->start = i * slice_size;
        data->mode = mode;
        if (i == nb_threads - 1)
            slice_size += overflow;
        data->slice_size = slice_size;
        pthread_create(&threads[i], NULL, worker_func, (void*)data);
    }
    // join all
    for (int i = 0; i < nb_threads; i++)
        pthread_join(threads[i], NULL);
    // perform calculations
    float w_sum, u_w_sum;
    for (int i = 0; i < nb_threads; i++) {
        w_sum += p_w_sum[i];
        u_w_sum += p_u_w_sum[i];
    }
    return u_w_sum / w_sum;
}


void init(){
  unsigned int i;
  for( i = 0; i < N; i++ ){
    U[i] = (float)rand () / RAND_MAX ;
    W[i] = (float)rand () / RAND_MAX ;
  }
}

int main(){
  a = 0;
  int k = 4;
  init();

  double t;
  float rs, rv, rp;

  t = now();
  rs = gm(U, W, a, k, N);
  t = now()-t;
  printf("S = %10.9f Temps du code scalaire             : %f seconde(s)\n",rs,t);

  t = now();
  rv = vect_gm(U, W, a, k, N);
  t = now()-t;
  printf("S = %10.9f Temps du code vectoriel 1          : %f seconde(s)\n",rv,t);

  t = now();
  rp = parallel_gm(U, W, a, k, N, 0, 4);
  t = now() - t;
  printf("S = %10.9f Temps du code multi-thread (mode 0): %f seconde(s)\n", rp, t);

  t = now();
  rp = parallel_gm(U, W, a, k, N, 1, 4);
  t = now() - t;
  printf("S = %10.9f Temps du code multi-thread (mode 1): %f seconde(s)\n", rp, t);

  return 0;

}
