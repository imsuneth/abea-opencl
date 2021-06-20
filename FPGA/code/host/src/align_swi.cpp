#include "f5c.h"
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <math.h>
#include <cstring>
#include "CL/opencl.h"
#include "AOCLUtils/aocl_utils.h"

#include "dump_functions.h"

using namespace aocl_utils;

#include "error.h"
#include "f5cmisc_cu.h"
#include "f5cmisc.h"

const char *binary_name = "align";
int print_results = false;
#define VERBOSITY 0

#define AOCL_ALIGNMENT 64

// #define CPU_GPU_PROC
// #define DEBUG_ADAPTIVE

#define STRING_BUFFER_LEN 1024

#define bandwidth ALN_BANDWIDTH
#define half_bandwidth ALN_BANDWIDTH / 2

#define max_gap_threshold 50
#define min_average_log_emission -5.0
#define epsilon 1e-10

#define FROM_D 0
#define FROM_U 1
#define FROM_L 2

// OpenCL runtime configuration
static cl_platform_id platform = NULL;
static cl_device_id device = NULL;
static cl_context context = NULL;
static cl_command_queue queue = NULL;
static cl_kernel align_kernel_single = NULL;

static cl_program program = NULL;

cl_int status;

// Function prototypes
bool init();
void cleanup();
static void device_info_ulong(cl_device_id device, cl_device_info param, const char *name);
static void device_info_uint(cl_device_id device, cl_device_info param, const char *name);
static void device_info_bool(cl_device_id device, cl_device_info param, const char *name);
static void device_info_string(cl_device_id device, cl_device_info param, const char *name);
static void display_device_info(cl_device_id device);

static void align_ocl(core_t *core, db_t *db);

void host_post_processing(AlignedPair *event_align_pairs,
                          int32_t *n_event_align_pairs,
                          char *read, int32_t *read_len,
                          ptr_t *read_ptr, event_t *event_table,
                          int32_t *n_events1, ptr_t *event_ptr,
                          scalings_t *scalings, model_t *models,
                          int32_t n_bam_rec, uint32_t *kmer_ranks1,
                          float *band, uint8_t *traces,
                          EventKmerPair *band_lower_lefts);

double total_time_pre_kernel = 0;
double total_time_core_kernel = 0;
double total_time_post_kernel = 0;
double host_to_device_transfer_time = 0;
double device_to_host_transfer_time = 0;

double align_kernel_time;
double align_pre_kernel_time;
double align_core_kernel_time;
double align_post_kernel_time;
double align_cl_malloc;
double align_cl_memcpy;
double align_cl_memcpy_back;
double align_cl_postprocess;
double align_cl_preprocess;
double align_cl_total_kernel;

// Entry point.
int main(int argc, char *argv[])
{

  if (!init())
  {
    fprintf(stderr, "init() unsuccessful\n");
    return -1;
  }

  fprintf(stderr, "init() successful\n\n");

  // Load dump files
  const char *dump_dir = argv[1];

  int32_t no_of_batches = load_no_of_batches(dump_dir);
  // int32_t no_of_batches = 1;

  fprintf(stderr, "no_of_batches: %d\n", no_of_batches);
  int32_t batch_no = 0;

  int32_t total_no_of_reads = 0;

  for (batch_no = 0; batch_no < no_of_batches; batch_no++)
  {

    fprintf(stderr, "batch_no: %d/%d\t", batch_no, no_of_batches);

    char batch_dir[50];
    snprintf(batch_dir, sizeof(batch_dir), "%s/%ld", dump_dir, batch_no);

    db_t *db;
    db = (db_t *)malloc(sizeof(db_t));
    // db = new db_t();

    core_t *core;
    core = (core_t *)malloc(sizeof(core_t));
    core->opt.verbosity = VERBOSITY;

    load_n_bam_rec(db, batch_dir);
    // db->n_bam_rec = 5;

    // core->model = (model_t*)malloc(sizeof(model_t)*db->n_bam_rec);
    posix_memalign((void **)&core->model, AOCL_ALIGNMENT, sizeof(model_t) * NUM_KMER);

    load_core(core, batch_dir);

    fprintf(stderr, "reads:\t%ld\n", db->n_bam_rec);
    total_no_of_reads += db->n_bam_rec;

    db->n_event_align_pairs = (int32_t *)malloc(sizeof(int32_t) * db->n_bam_rec);
    db->event_align_pairs = (AlignedPair **)malloc(sizeof(AlignedPair *) * db->n_bam_rec);
    // db->read_len = (int32_t *)malloc(sizeof(int32_t) * db->n_bam_rec);
    posix_memalign((void **)&db->read_len, AOCL_ALIGNMENT, sizeof(int32_t) * db->n_bam_rec);
    db->read = (char **)malloc(sizeof(char *) * db->n_bam_rec);
    db->et = (event_table *)malloc(sizeof(event_table) * db->n_bam_rec);
    // db->scalings = (scalings_t *)malloc(sizeof(scalings_t) * db->n_bam_rec);
    posix_memalign((void **)&db->scalings, AOCL_ALIGNMENT, sizeof(scalings_t) * db->n_bam_rec);
    db->f5 = (fast5_t **)malloc(sizeof(fast5_t *) * db->n_bam_rec);

    db_t *db_out;
    db_out = (db_t *)malloc(sizeof(db_t));
    db_out->n_event_align_pairs = (int32_t *)malloc(sizeof(int32_t) * db->n_bam_rec);
    db_out->event_align_pairs = (AlignedPair **)malloc(sizeof(AlignedPair *) * db->n_bam_rec);

    // printf("db, core initialized\n");

    for (int i = 0; i < db->n_bam_rec; i++)
    {

      load_read_inputs(db, i, batch_dir);
      load_read_outputs(db_out, i, batch_dir);
      // posix_memalign((void **)&db->event_align_pairs[i], AOCL_ALIGNMENT, sizeof(AlignedPair) * db_out->n_event_align_pairs[i]);
      // db->event_align_pairs[i] = (AlignedPair *)malloc(sizeof(AlignedPair) * db_out->n_event_align_pairs[i]);
    }

    align_ocl(core, db);

    if (print_results)
    {
      for (int i = 0; i < db->n_bam_rec; i++)
      {
        // compare with original output
        int32_t n_event_align_pairs = db->n_event_align_pairs[i];
        int32_t n_event_align_pairs_out = db_out->n_event_align_pairs[i];

        if (n_event_align_pairs != n_event_align_pairs_out)
        {
          fprintf(stderr, "%d=\toutput: %d (%d)\tFailed\n", i, n_event_align_pairs, n_event_align_pairs_out);
          //break;
        }
        else
        {
          fprintf(stderr, "%d=\toutput: %d (%d)\tPassed ", i, n_event_align_pairs, n_event_align_pairs_out);
          // if (check_event_align_pairs(db->event_align_pairs[i], db_out->event_align_pairs[i], n_event_align_pairs) == 0)
          // {
          //   // fprintf(stderr, "%d=\t Found conflict in event_align_pairs\n", i);
          //   fprintf(stderr, "%d=\toutput: %d, expected: %d\tFailed\n", i, n_event_align_pairs, n_event_align_pairs_out);
          // }
          // else
          // {
          //   // fprintf(stderr, "%d=\t Run pass\n", i);
          //   fprintf(stderr, "%d=\toutput: %d, expected: %d\tPassed\n", i, n_event_align_pairs, n_event_align_pairs_out);
          // }
          check_event_align_pairs(db->event_align_pairs[i], db_out->event_align_pairs[i], n_event_align_pairs);
          fprintf(stderr, "\n");
        }

        // printf("readpos:%d, refpos:%d\n",db_out->event_align_pairs[0]->read_pos, db_out->event_align_pairs[0]->ref_pos);
      }
    }

    // free
    for (int i = 0; i < db->n_bam_rec; i++)
    {
      free(db->et[i].event);
    }
    free(core->model);
    free(db->read_len);
    free(db->scalings);

    free(db->f5);
    free(db->event_align_pairs);
    free(db->read);

    free(db_out->event_align_pairs);
    // free(db_out->event_align_pairs);
    free(db_out->n_event_align_pairs);
    free(db_out);

    // free(db->f5);

    free(db->et);
    // free(db->read);

    // free(db->event_align_pairs);
    free(db->n_event_align_pairs);

    free(core);
    free(db);
  }

  fprintf(stderr, "Pre processing: %.3f seconds\n", align_pre_kernel_time);
  fprintf(stderr, "Core kernel %.3f seconds\n", align_core_kernel_time);
  fprintf(stderr, "Post processing %.3f seconds\n", align_post_kernel_time);
  align_cl_total_kernel = align_pre_kernel_time + align_core_kernel_time + align_post_kernel_time;
  fprintf(stderr, "Total execution  %.3f seconds\n", align_cl_total_kernel);
  double align_cl_total_data = align_cl_memcpy + align_cl_memcpy_back;
  fprintf(stderr, "Data transfer: %.3f seconds\n", align_cl_total_data);
  fprintf(stderr, "Total number of reads %ld\n\n\n", total_no_of_reads);

  cleanup();
  return 0;
}

void align_ocl(core_t *core, db_t *db)
{

  int32_t i;
  int32_t n_bam_rec = db->n_bam_rec;
  double realtime1;

  /**cuda pointers*/
  char *read_host;      //flattened reads sequences
  ptr_t *read_ptr_host; //index pointer for flattedned "reads"

  int64_t sum_read_len;
  int32_t *n_events_host;
  event_t *event_table_host;
  ptr_t *event_ptr_host;
  int64_t sum_n_events;
  AlignedPair *event_align_pairs_host;
  int32_t *n_event_align_pairs_host;
  float *bands_host;
  uint8_t *trace_host;
  EventKmerPair *band_lower_left_host;
  uint32_t *kmer_rank_host;

  /*time measurements*/
  cl_event event;
  // cl_double host_to_device_transfer_time = 0;
  // cl_double device_to_host_transfer_time = 0;
  cl_ulong start = 0, end = 0;

  realtime1 = realtime();

  // read_ptr_host = (ptr_t *)malloc(sizeof(ptr_t) * n_bam_rec);
  posix_memalign((void **)&read_ptr_host, AOCL_ALIGNMENT, n_bam_rec * sizeof(ptr_t));
  MALLOC_CHK(read_ptr_host);

  // #endif
  sum_read_len = 0;

  int32_t max_read_len = 0;

  //read sequences : needflattening
  for (i = 0; i < n_bam_rec; i++)
  {
    if (max_read_len < db->read_len[i])
      max_read_len = db->read_len[i];

    read_ptr_host[i] = sum_read_len;
    sum_read_len += (db->read_len[i] + 1); //with null term
                                           //printf("sum_read_len:%d += (db->read_len[i]: %d + 1)\n", sum_read_len, db->read_len[i]);
  }

  if (core->opt.verbosity > 1)
  {
    fprintf(stderr, "num of bases:\t%lu\n", sum_read_len - n_bam_rec);
    fprintf(stderr, "mean read len:\t%d\n", (sum_read_len - n_bam_rec) / n_bam_rec);
    fprintf(stderr, "max read len:\t%lu\n", max_read_len);
  }
  // fprintf(stderr, "n_bam_rec %d, sum_read_len %d\n", n_bam_rec, sum_read_len);
  //form the temporary flattened array on host
  // read_host = (char *)malloc(sizeof(char) * sum_read_len);
  posix_memalign((void **)&read_host, AOCL_ALIGNMENT, sizeof(char) * sum_read_len);
  MALLOC_CHK(read_host);

  for (i = 0; i < n_bam_rec; i++)
  {
    ptr_t idx = read_ptr_host[i];
    strcpy(&read_host[idx], db->read[i]);
  }

  if (core->opt.verbosity > 1)
    fprintf(stderr, "here1\n");
  //now the events : need flattening
  //num events : need flattening
  //get the total size and create the pointers

  // n_events_host = (int32_t *)malloc(sizeof(int32_t) * n_bam_rec);
  posix_memalign((void **)&n_events_host, AOCL_ALIGNMENT, n_bam_rec * sizeof(int32_t));
  MALLOC_CHK(n_events_host);

  // event_ptr_host = (ptr_t *)malloc(sizeof(ptr_t) * n_bam_rec);
  posix_memalign((void **)&event_ptr_host, AOCL_ALIGNMENT, n_bam_rec * sizeof(ptr_t));
  MALLOC_CHK(event_ptr_host);
  // #endif

  sum_n_events = 0;
  for (i = 0; i < n_bam_rec; i++)
  {
    n_events_host[i] = (int32_t)db->et[i].n;
    event_ptr_host[i] = sum_n_events;
    sum_n_events += db->et[i].n;
    // fprintf(stderr, "read_i:%d, sum_n_events: %d\n", i, db->et[i].n);
  }
  if (core->opt.verbosity > 1)
    fprintf(stderr, "here2\n");

  //event table flatten
  //form the temporary flattened array on host
  // event_table_host = (event_t *)malloc(sizeof(event_t) * sum_n_events);

  // fprintf(stderr, "sum_n_events: %d\n", sum_n_events);

  posix_memalign((void **)&event_table_host, AOCL_ALIGNMENT, sum_n_events * sizeof(event_t));
  if (core->opt.verbosity > 1)
    fprintf(stderr, "mallocd\n");
  MALLOC_CHK(event_table_host);

  if (core->opt.verbosity > 1)
    fprintf(stderr, "here3\n");

  for (i = 0; i < n_bam_rec; i++)
  {
    ptr_t idx = event_ptr_host[i];
    memcpy(&event_table_host[idx], db->et[i].event,
           sizeof(event_t) * db->et[i].n);

    // fprintf(stderr, "n_events:%d, event_table:\n", db->et[i].n);
    // for(int j=0; j<db->et[0].n; j++){
    //   fprintf(stderr, "%f ", db->et[i].event[j].mean);
    // }
    // fprintf(stderr, "\nend of event_table");
  }

  size_t sum_n_bands = sum_n_events + sum_read_len; //todo : can be optimised
  if (core->opt.verbosity > 1)
  {
    fprintf(stderr, "sum_n_events: %d\n", sum_n_events);
    fprintf(stderr, "sum_read_len: %d\n", sum_read_len);
    fprintf(stderr, "sum_n_bands: %d\n", sum_n_bands);
  }

  posix_memalign((void **)&event_align_pairs_host, AOCL_ALIGNMENT, 2 * sum_n_events * sizeof(AlignedPair));
  MALLOC_CHK(event_align_pairs_host);

  posix_memalign((void **)&kmer_rank_host, AOCL_ALIGNMENT, sum_read_len * sizeof(uint32_t));
  MALLOC_CHK(kmer_rank_host);

  posix_memalign((void **)&band_lower_left_host, AOCL_ALIGNMENT, sum_n_bands * sizeof(EventKmerPair));
  MALLOC_CHK(band_lower_left_host);

  posix_memalign((void **)&trace_host, AOCL_ALIGNMENT, sizeof(uint8_t) * sum_n_bands * ALN_BANDWIDTH);
  MALLOC_CHK(trace_host);

  posix_memalign((void **)&bands_host, AOCL_ALIGNMENT, sizeof(float) * sum_n_bands * ALN_BANDWIDTH);
  MALLOC_CHK(bands_host);

  posix_memalign((void **)&n_event_align_pairs_host, AOCL_ALIGNMENT, n_bam_rec * sizeof(int32_t));
  MALLOC_CHK(n_event_align_pairs_host);

  align_cl_preprocess += (realtime() - realtime1);

  /** Start OpenCL memory allocation**/
  realtime1 = realtime();

  //read_ptr
  if (core->opt.verbosity > 1)
    print_size("read_ptr array", n_bam_rec * sizeof(ptr_t));
  cl_mem read_ptr = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, n_bam_rec * sizeof(ptr_t), read_ptr_host, &status);
  checkError(status, "Failed clCreateBuffer");

  //read_len
  if (core->opt.verbosity > 1)
    print_size("read_lens", n_bam_rec * sizeof(int32_t));
  cl_mem read_len = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, n_bam_rec * sizeof(int32_t), db->read_len, &status);
  checkError(status, "Failed clCreateBuffer");

  //n_events
  if (core->opt.verbosity > 1)
    print_size("n_events", n_bam_rec * sizeof(size_t));
  cl_mem n_events = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, n_bam_rec * sizeof(int32_t), n_events_host, &status);
  checkError(status, "Failed clCreateBuffer");

  //event ptr
  if (core->opt.verbosity > 1)
    print_size("event ptr", n_bam_rec * sizeof(ptr_t));
  cl_mem event_ptr = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, n_bam_rec * sizeof(ptr_t), event_ptr_host, &status);
  checkError(status, "Failed clCreateBuffer");

  //scalings : already linear
  if (core->opt.verbosity > 1)
    print_size("Scalings", n_bam_rec * sizeof(scalings_t));
  cl_mem scalings = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, n_bam_rec * sizeof(scalings_t), db->scalings, &status);
  checkError(status, "Failed clCreateBuffer");

  //model : already linear
  if (core->opt.verbosity > 1)
    print_size("model", NUM_KMER * sizeof(model_t));
  cl_mem model = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, NUM_KMER * sizeof(model_t), core->model, &status);
  checkError(status, "Failed clCreateBuffer");

  //read
  if (core->opt.verbosity > 1)
    print_size("read array", sum_read_len * sizeof(char));
  cl_mem read = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sum_read_len * sizeof(char), read_host, &status);
  checkError(status, "Failed clCreateBuffer");

  //event_table
  if (core->opt.verbosity > 1)
    print_size("event table", sum_n_events * sizeof(event_t));
  cl_mem event_table = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sum_n_events * sizeof(event_t), event_table_host, &status);
  checkError(status, "Failed clCreateBuffer");

  //kmer_rank
  if (core->opt.verbosity > 1)
    print_size("kmer rank", sum_read_len * sizeof(uint32_t));
  cl_mem kmer_rank = clCreateBuffer(context, CL_MEM_READ_WRITE, sum_read_len * sizeof(uint32_t), NULL, &status);
  checkError(status, "Failed clCreateBuffer");

  //bands
  if (core->opt.verbosity > 1)
    print_size("bands", sizeof(float) * sum_n_bands * ALN_BANDWIDTH);
  cl_mem bands = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * sum_n_bands * ALN_BANDWIDTH, NULL, &status);
  checkError(status, "Failed clCreateBuffer");

  //trace
  // for (i = 0; i < sum_n_bands * ALN_BANDWIDTH; i++)
  // {
  //   trace_host[i] = 0;
  // }
  if (core->opt.verbosity > 1)
    print_size("trace", sizeof(uint8_t) * sum_n_bands * ALN_BANDWIDTH);
  cl_mem trace = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(uint8_t) * sum_n_bands * ALN_BANDWIDTH, NULL, &status);
  checkError(status, "Failed clCreateBuffer");

  // if (core->opt.verbosity > 1)
  //   print_size("trace", sizeof(uint8_t) * sum_n_bands * BLOCK_LEN_BANDWIDTH);
  // cl_mem trace = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(uint8_t) * sum_n_bands * BLOCK_LEN_BANDWIDTH, NULL, &status);
  // checkError(status, "Failed clCreateBuffer");

  // uint8_t zeros[n_bam_rec];
  // for (i = 0; i < n_bam_rec; i++)
  // {
  //   zeros[i] = 0;
  // }
  // status = clEnqueueWriteBuffer(queue, trace, CL_TRUE, 0, n_bam_rec * sizeof(uint8_t), zeros, 0, NULL, NULL);
  // checkError(status, "Failed clEnqueueWriteBuffer");

  //band_lower_left
  if (core->opt.verbosity > 1)
    print_size("band_lower_left", sizeof(EventKmerPair) * sum_n_bands);
  cl_mem band_lower_left = clCreateBuffer(context, CL_MEM_READ_WRITE, sum_n_bands * sizeof(EventKmerPair), NULL, &status);
  checkError(status, "Failed clCreateBuffer");

  // //read_ptr
  // if (core->opt.verbosity > 1)
  //   print_size("read_ptr array", n_bam_rec * sizeof(ptr_t));
  // cl_mem read_ptr = clCreateBuffer(context, CL_MEM_READ_ONLY, n_bam_rec * sizeof(ptr_t), NULL, &status);
  // checkError(status, "Failed clCreateBuffer");

  // //read_len
  // if (core->opt.verbosity > 1)
  //   print_size("read_lens", n_bam_rec * sizeof(int32_t));
  // cl_mem read_len = clCreateBuffer(context, CL_MEM_READ_ONLY, n_bam_rec * sizeof(int32_t), NULL, &status);
  // checkError(status, "Failed clCreateBuffer");

  // //n_events
  // if (core->opt.verbosity > 1)
  //   print_size("n_events", n_bam_rec * sizeof(int32_t));
  // cl_mem n_events = clCreateBuffer(context, CL_MEM_READ_ONLY, n_bam_rec * sizeof(int32_t), NULL, &status);
  // checkError(status, "Failed clCreateBuffer");

  // //event ptr
  // if (core->opt.verbosity > 1)
  //   print_size("event ptr", n_bam_rec * sizeof(ptr_t));
  // cl_mem event_ptr = clCreateBuffer(context, CL_MEM_READ_ONLY, n_bam_rec * sizeof(ptr_t), NULL, &status);
  // checkError(status, "Failed clCreateBuffer");

  // //scalings : already linear
  // if (core->opt.verbosity > 1)
  //   print_size("Scalings", n_bam_rec * sizeof(scalings_t));
  // cl_mem scalings = clCreateBuffer(context, CL_MEM_READ_ONLY, n_bam_rec * sizeof(scalings_t), NULL, &status);
  // checkError(status, "Failed clCreateBuffer");

  // //model : already linear
  // if (core->opt.verbosity > 1)
  //   print_size("model", NUM_KMER * sizeof(model_t));
  // cl_mem model = clCreateBuffer(context, CL_MEM_READ_ONLY, NUM_KMER * sizeof(model_t), NULL, &status);
  // checkError(status, "Failed clCreateBuffer");

  // //read
  // if (core->opt.verbosity > 1)
  //   print_size("read array", sum_read_len * sizeof(char));
  // cl_mem read = clCreateBuffer(context, CL_MEM_READ_ONLY, sum_read_len * sizeof(char), NULL, &status);
  // checkError(status, "Failed clCreateBuffer");

  // //event_table
  // if (core->opt.verbosity > 1)
  //   print_size("event table", sum_n_events * sizeof(event_t));
  // cl_mem event_table = clCreateBuffer(context, CL_MEM_READ_ONLY, sum_n_events * sizeof(event_t), NULL, &status);
  // checkError(status, "Failed clCreateBuffer");

  // //kmer_rank
  // if (core->opt.verbosity > 1)
  //   print_size("kmer rank", sum_read_len * sizeof(model_t));
  // cl_mem kmer_rank = clCreateBuffer(context, CL_MEM_READ_WRITE, sum_read_len * sizeof(size_t), NULL, &status);
  // checkError(status, "Failed clCreateBuffer");

  // //bands
  // if (core->opt.verbosity > 1)
  //   print_size("bands", sizeof(float) * sum_n_bands * ALN_BANDWIDTH);
  // cl_mem bands = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * sum_n_bands * ALN_BANDWIDTH, NULL, &status);
  // checkError(status, "Failed clCreateBuffer");

  // //trace
  // posix_memalign((void **)&trace_host, AOCL_ALIGNMENT, sizeof(uint8_t) * sum_n_bands * ALN_BANDWIDTH);
  // MALLOC_CHK(trace_host);
  // // for (i = 0; i < sum_n_bands * ALN_BANDWIDTH; i++)
  // // {
  // //   trace_host[i] = 0;
  // // }
  // if (core->opt.verbosity > 1)
  //   print_size("trace", sizeof(uint8_t) * sum_n_bands * ALN_BANDWIDTH);
  // cl_mem trace = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(uint8_t) * sum_n_bands * ALN_BANDWIDTH, NULL, &status);
  // checkError(status, "Failed clCreateBuffer");

  // // if (core->opt.verbosity > 1)
  // //   print_size("trace", sizeof(uint8_t) * sum_n_bands * BLOCK_LEN_BANDWIDTH);
  // // cl_mem trace = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(uint8_t) * sum_n_bands * BLOCK_LEN_BANDWIDTH, NULL, &status);
  // // checkError(status, "Failed clCreateBuffer");

  // // uint8_t zeros[n_bam_rec];
  // // for (i = 0; i < n_bam_rec; i++)
  // // {
  // //   zeros[i] = 0;
  // // }
  // // status = clEnqueueWriteBuffer(queue, trace, CL_TRUE, 0, n_bam_rec * sizeof(uint8_t), zeros, 0, NULL, NULL);
  // // checkError(status, "Failed clEnqueueWriteBuffer");

  // //band_lower_left
  // if (core->opt.verbosity > 1)
  //   print_size("band_lower_left", sizeof(EventKmerPair) * sum_n_bands);
  // cl_mem band_lower_left = clCreateBuffer(context, CL_MEM_READ_WRITE, sum_n_bands * sizeof(EventKmerPair), NULL, &status);
  // checkError(status, "Failed clCreateBuffer");

  // align_cl_malloc += (realtime() - realtime1);

  // /* cuda mem copys*/
  // realtime1 = realtime();

  // //read_ptr
  // status = clEnqueueWriteBuffer(queue, read_ptr, CL_TRUE, 0, n_bam_rec * sizeof(ptr_t), read_ptr_host, 0, NULL, NULL);
  // checkError(status, "Failed clEnqueueWriteBuffer");

  // //read
  // status = clEnqueueWriteBuffer(queue, read, CL_TRUE, 0, sum_read_len * sizeof(char), read_host, 0, NULL, NULL);
  // checkError(status, "Failed clEnqueueWriteBuffer");

  // //read_len
  // status = clEnqueueWriteBuffer(queue, read_len, CL_TRUE, 0, n_bam_rec * sizeof(int32_t), db->read_len, 0, NULL, NULL);
  // checkError(status, "Failed clEnqueueWriteBuffer");

  // //n_events
  // status = clEnqueueWriteBuffer(queue, n_events, CL_TRUE, 0, n_bam_rec * sizeof(int32_t), n_events_host, 0, NULL, NULL);
  // checkError(status, "Failed clEnqueueWriteBuffer");

  // //event_ptr
  // status = clEnqueueWriteBuffer(queue, event_ptr, CL_TRUE, 0, n_bam_rec * sizeof(ptr_t), event_ptr_host, 0, NULL, NULL);
  // checkError(status, "Failed clEnqueueWriteBuffer");

  // //event_table
  // status = clEnqueueWriteBuffer(queue, event_table, CL_TRUE, 0, sizeof(event_t) * sum_n_events, event_table_host, 0, NULL, NULL);
  // checkError(status, "Failed clEnqueueWriteBuffer");

  // //model
  // status = clEnqueueWriteBuffer(queue, model, CL_TRUE, 0, NUM_KMER * sizeof(model_t), core->model, 0, NULL, NULL);
  // checkError(status, "Failed clEnqueueWriteBuffer");

  // //scalings
  // status = clEnqueueWriteBuffer(queue, scalings, CL_TRUE, 0, sizeof(scalings_t) * n_bam_rec, db->scalings, 0, NULL, NULL);
  // checkError(status, "Failed clEnqueueWriteBuffer");

  align_cl_memcpy += (realtime() - realtime1);

  realtime1 = realtime();

  // status = clSetKernelArg(align_kernel_single, 0, sizeof(cl_mem), &event_align_pairs);
  // checkError(status, "Failed to set kernel args");

  // status = clSetKernelArg(align_kernel_single, 1, sizeof(cl_mem), &n_event_align_pairs);
  // checkError(status, "Failed to set kernel args");

  status = clSetKernelArg(align_kernel_single, 0, sizeof(cl_mem), &read);
  checkError(status, "Failed to set kernel args");

  status = clSetKernelArg(align_kernel_single, 1, sizeof(cl_mem), &read_len);
  checkError(status, "Failed to set kernel args");

  status = clSetKernelArg(align_kernel_single, 2, sizeof(cl_mem), &read_ptr);
  checkError(status, "Failed to set kernel args");

  status = clSetKernelArg(align_kernel_single, 3, sizeof(cl_mem), &event_table);
  checkError(status, "Failed to set kernel args");

  status = clSetKernelArg(align_kernel_single, 4, sizeof(cl_mem), &n_events);
  checkError(status, "Failed to set kernel args");

  status = clSetKernelArg(align_kernel_single, 5, sizeof(cl_mem), &event_ptr);
  checkError(status, "Failed to set kernel args");

  status = clSetKernelArg(align_kernel_single, 6, sizeof(cl_mem), &scalings);
  checkError(status, "Failed to set kernel args");

  status = clSetKernelArg(align_kernel_single, 7, sizeof(cl_mem), &model);
  checkError(status, "Failed to set kernel args");

  status = clSetKernelArg(align_kernel_single, 8, sizeof(int32_t), &n_bam_rec);
  checkError(status, "Failed to set kernel args");

  status = clSetKernelArg(align_kernel_single, 9, sizeof(cl_mem), &kmer_rank);
  checkError(status, "Failed to set kernel args");

  status = clSetKernelArg(align_kernel_single, 10, sizeof(cl_mem), &bands);
  checkError(status, "Failed to set kernel args");

  status = clSetKernelArg(align_kernel_single, 11, sizeof(cl_mem), &trace);
  checkError(status, "Failed to set kernel args");

  status = clSetKernelArg(align_kernel_single, 12, sizeof(cl_mem), &band_lower_left);
  checkError(status, "Failed to set kernel args");

  // assert(BLOCK_LEN_BANDWIDTH >= ALN_BANDWIDTH);

  const size_t gSize[3] = {1, 1, 1};  //global
  const size_t wgSize[3] = {1, 1, 1}; //local

  // if (core->opt.verbosity > 1)
  //   fprintf(stderr, "global %zu,%zu, work_group %zu,%zu\n", gSize[0], gSize[1], wgSize[0], wgSize[1]);

  // printf("Calling kernel\n");

  // clEnqueueNDRangeKernel(queue, align_kernel_single, 1, NULL, gSize, wgSize, 0, NULL, NULL);

  clEnqueueTask(queue, align_kernel_single, 0, NULL, NULL);

  status = clFinish(queue);
  checkError(status, "Failed to finish");

  //********** Pre-Kernel execution time *************************

  // if (core->opt.verbosity > 1)
  //   fprintf(stderr, "[%s::%.3f*%.2f] align-pre kernel done\n", __func__, realtime() - realtime1, cputime() / (realtime() - realtime1));
  align_kernel_time += (realtime() - realtime1);
  align_pre_kernel_time += (realtime() - realtime1);

  realtime1 = realtime();

  //bands_host,
  status = clEnqueueReadBuffer(queue, bands, CL_TRUE, 0, sizeof(float) * sum_n_bands * ALN_BANDWIDTH, bands_host, 0, NULL, NULL);
  checkError(status, "clEnqueueReadBuffer");

  //trace_host,
  status = clEnqueueReadBuffer(queue, trace, CL_TRUE, 0, sizeof(uint8_t) * sum_n_bands * ALN_BANDWIDTH, trace_host, 0, NULL, NULL);
  checkError(status, "clEnqueueReadBuffer");

  //band_lower_left_host
  status = clEnqueueReadBuffer(queue, band_lower_left, CL_TRUE, 0, sum_n_bands * sizeof(EventKmerPair), band_lower_left_host, 0, NULL, NULL);
  checkError(status, "clEnqueueReadBuffer");

  //kmer_ranks
  status = clEnqueueReadBuffer(queue, kmer_rank, CL_TRUE, 0, sum_read_len * sizeof(uint32_t), kmer_rank_host, 0, NULL, NULL);
  checkError(status, "clEnqueueReadBuffer");

  align_cl_memcpy_back += (realtime() - realtime1);

  realtime1 = realtime();

  status = clReleaseMemObject(read_ptr);
  checkError(status, "clReleaseMemObject failed!");
  status = clReleaseMemObject(read_len);
  checkError(status, "clReleaseMemObject failed!");
  status = clReleaseMemObject(n_events);
  checkError(status, "clReleaseMemObject failed!");
  status = clReleaseMemObject(event_ptr);
  checkError(status, "clReleaseMemObject failed!");
  status = clReleaseMemObject(scalings);
  checkError(status, "clReleaseMemObject failed!");
  status = clReleaseMemObject(model);
  checkError(status, "clReleaseMemObject failed!");
  status = clReleaseMemObject(read);
  checkError(status, "clReleaseMemObject failed!");
  status = clReleaseMemObject(event_table);
  checkError(status, "clReleaseMemObject failed!");
  status = clReleaseMemObject(kmer_rank);
  checkError(status, "clReleaseMemObject failed!");
  // status = clReleaseMemObject(event_align_pairs);
  // checkError(status, "clReleaseMemObject failed!");
  // status = clReleaseMemObject(n_event_align_pairs);
  // checkError(status, "clReleaseMemObject failed!");
  status = clReleaseMemObject(bands);
  checkError(status, "clReleaseMemObject failed!");
  status = clReleaseMemObject(trace);
  checkError(status, "clReleaseMemObject failed!");
  status = clReleaseMemObject(band_lower_left);
  checkError(status, "clReleaseMemObject failed!");

  align_cl_malloc += (realtime() - realtime1);

  /** post work**/
  realtime1 = realtime();

  if (core->opt.verbosity > 1)
    fprintf(stderr, "Read back done \n");

  //HOST POST PROCESSING ===========================================================
  host_post_processing(event_align_pairs_host,
                       n_event_align_pairs_host,
                       read_host, db->read_len,
                       read_ptr_host, event_table_host,
                       n_events_host, event_ptr_host,
                       db->scalings, core->model,
                       n_bam_rec, kmer_rank_host,
                       bands_host, trace_host,
                       band_lower_left_host);

  if (core->opt.verbosity > 1)
    fprintf(stderr, "Post kernel done: %.3f seconds\n", (realtime() - realtime1));
  //===============================================================================
  align_post_kernel_time += (realtime() - realtime1);

  /** post work**/
  realtime1 = realtime();

  //copy back
  for (i = 0; i < n_bam_rec; i++)
  {
    db->n_event_align_pairs[i] = n_event_align_pairs_host[i];
    db->event_align_pairs[i] = (AlignedPair *)malloc(sizeof(AlignedPair) * db->n_event_align_pairs[i]);
    ptr_t idx = event_ptr_host[i];
    memcpy(db->event_align_pairs[i], &event_align_pairs_host[idx * 2], sizeof(AlignedPair) * db->n_event_align_pairs[i]);
  }

  free(read_ptr_host);
  free(read_host);
  free(n_events_host);
  free(event_ptr_host);
  free(event_table_host);
  free(trace_host);
  free(bands_host);
  free(band_lower_left_host);
  free(kmer_rank_host);
  free(n_event_align_pairs_host);
  free(event_align_pairs_host);

  align_cl_postprocess += (realtime() - realtime1);
}

/////// HELPER FUNCTIONS ///////

bool init()
{
  cl_int status;

  if (!setCwdToExeDir())
  {
    return false;
  }

  // Get the OpenCL platform.
  platform = findPlatform("Intel(R) FPGA SDK for OpenCL(TM)");
  if (platform == NULL)
  {
    printf("ERROR: Unable to find Intel(R) FPGA OpenCL platform.\n");
    return false;
  }

  // Query the available OpenCL devices.
  scoped_array<cl_device_id> devices;
  cl_uint num_devices;

  devices.reset(getDevices(platform, CL_DEVICE_TYPE_ALL, &num_devices));

  // We'll just use the first device.
  device = devices[0];

  // Display some device information.
  // display_device_info(device);

  // Create the context.
  context = clCreateContext(NULL, 1, &device, &oclContextCallback, NULL, &status);
  checkError(status, "Failed to create context");

  // Create the command queue.
  queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &status);
  checkError(status, "Failed to create command queue");

  // Create the program.

  std::string binary_file = getBoardBinaryFile(binary_name, device);
  printf("Using AOCX: %s\n", binary_file.c_str());
  program = createProgramFromBinary(context, binary_file.c_str(), &device, 1);

  // Build the program that was just created.

  status = clBuildProgram(program, 0, NULL, "", NULL, NULL);
  checkError(status, "Failed to build program");

  // Create the kernel - name passed in here must match kernel name in the
  // original CL file, that was compiled into an AOCX file using the AOC tool
  const char *kernel1_name = "align_kernel_single"; // Kernel name, as defined in the CL file

  align_kernel_single = clCreateKernel(program, kernel1_name, &status);
  checkError(status, "Failed to pre create kernel");

  return true;
}

// Free the resources allocated during initialization
void cleanup()
{

  if (align_kernel_single)
  {
    clReleaseKernel(align_kernel_single);
  }

  if (program)
  {
    clReleaseProgram(program);
  }

  if (queue)
  {
    clReleaseCommandQueue(queue);
  }
  if (context)
  {
    clReleaseContext(context);
  }
}

// Helper functions to display parameters returned by OpenCL queries
static void device_info_ulong(cl_device_id device, cl_device_info param, const char *name)
{
  cl_ulong a = 99;
  clGetDeviceInfo(device, param, sizeof(cl_ulong), &a, NULL);
  printf("%-40s = %lu\n", name, a);
}
static void device_info_size_t(cl_device_id device, cl_device_info param, const char *name)
{
  size_t a;
  clGetDeviceInfo(device, param, sizeof(size_t), &a, NULL);
  printf("%-40s = %zu\n", name, a);
}

static void device_info_size_t_arr(cl_device_id device, cl_device_info param, const char *name)
{
  size_t a[3];
  clGetDeviceInfo(device, param, sizeof(size_t) * 3, &a, NULL);
  printf("%-40s = %zu, %zu, %zu\n", name, a[0], a[1], a[2]);
}
static void device_info_uint(cl_device_id device, cl_device_info param, const char *name)
{
  cl_uint a;
  clGetDeviceInfo(device, param, sizeof(cl_uint), &a, NULL);
  printf("%-40s = %u\n", name, a);
}
static void device_info_bool(cl_device_id device, cl_device_info param, const char *name)
{
  cl_bool a;
  clGetDeviceInfo(device, param, sizeof(cl_bool), &a, NULL);
  printf("%-40s = %s\n", name, (a ? "true" : "false"));
}
static void device_info_string(cl_device_id device, cl_device_info param, const char *name)
{
  char a[STRING_BUFFER_LEN];
  clGetDeviceInfo(device, param, STRING_BUFFER_LEN, &a, NULL);
  printf("%-40s = %s\n", name, a);
}

// Query and display OpenCL information on device and runtime environment
static void display_device_info(cl_device_id device)
{

  printf("Querying device for info:\n");
  printf("========================\n");
  device_info_string(device, CL_DEVICE_NAME, "CL_DEVICE_NAME");
  device_info_string(device, CL_DEVICE_VENDOR, "CL_DEVICE_VENDOR");
  device_info_uint(device, CL_DEVICE_VENDOR_ID, "CL_DEVICE_VENDOR_ID");
  device_info_string(device, CL_DEVICE_VERSION, "CL_DEVICE_VERSION");
  device_info_string(device, CL_DRIVER_VERSION, "CL_DRIVER_VERSION");
  device_info_uint(device, CL_DEVICE_ADDRESS_BITS, "CL_DEVICE_ADDRESS_BITS");
  device_info_bool(device, CL_DEVICE_AVAILABLE, "CL_DEVICE_AVAILABLE");
  device_info_bool(device, CL_DEVICE_ENDIAN_LITTLE, "CL_DEVICE_ENDIAN_LITTLE");
  device_info_ulong(device, CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, "CL_DEVICE_GLOBAL_MEM_CACHE_SIZE");
  device_info_ulong(device, CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE, "CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE");
  device_info_ulong(device, CL_DEVICE_GLOBAL_MEM_SIZE, "CL_DEVICE_GLOBAL_MEM_SIZE");
  device_info_bool(device, CL_DEVICE_IMAGE_SUPPORT, "CL_DEVICE_IMAGE_SUPPORT");
  device_info_ulong(device, CL_DEVICE_LOCAL_MEM_SIZE, "CL_DEVICE_LOCAL_MEM_SIZE");
  device_info_ulong(device, CL_DEVICE_MAX_CLOCK_FREQUENCY, "CL_DEVICE_MAX_CLOCK_FREQUENCY");
  device_info_ulong(device, CL_DEVICE_MAX_COMPUTE_UNITS, "CL_DEVICE_MAX_COMPUTE_UNITS");
  device_info_ulong(device, CL_DEVICE_MAX_CONSTANT_ARGS, "CL_DEVICE_MAX_CONSTANT_ARGS");
  device_info_ulong(device, CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, "CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE");
  device_info_uint(device, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, "CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS");
  device_info_uint(device, CL_DEVICE_MEM_BASE_ADDR_ALIGN, "CL_DEVICE_MEM_BASE_ADDR_ALIGN");
  device_info_uint(device, CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE, "CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE");
  device_info_uint(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR, "CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR");
  device_info_uint(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT, "CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT");
  device_info_uint(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT, "CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT");
  device_info_uint(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG, "CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG");
  device_info_uint(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT, "CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT");
  device_info_uint(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE, "CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE");
  device_info_size_t_arr(device, CL_DEVICE_MAX_WORK_ITEM_SIZES, "CL_DEVICE_MAX_WORK_ITEM_SIZES");
  device_info_size_t(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, "CL_DEVICE_MAX_WORK_GROUP_SIZE");

  {
    cl_command_queue_properties ccp;
    clGetDeviceInfo(device, CL_DEVICE_QUEUE_PROPERTIES, sizeof(cl_command_queue_properties), &ccp, NULL);
    printf("%-40s = %s\n", "Command queue out of order? ", ((ccp & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE) ? "true" : "false"));
    printf("%-40s = %s\n", "Command queue profiling enabled? ", ((ccp & CL_QUEUE_PROFILING_ENABLE) ? "true" : "false"));
  }
}
//#endif

static inline uint32_t get_rank(char base)
{
  if (base == 'A')
  {
    return 0;
  }
  else if (base == 'C')
  {
    return 1;
  }
  else if (base == 'G')
  {
    return 2;
  }
  else if (base == 'T')
  {
    return 3;
  }
  else
  {
    WARNING("A None ACGT base found : %c", base);
    return 0;
  }
}

// return the lexicographic rank of the kmer amongst all strings of
// length k for this alphabet
static inline uint32_t get_kmer_rank(const char *str, uint32_t k)
{

  // fprintf(stderr, "substring:%s\n", str); //uint32_t p = 1;
  uint32_t r = 0;

  // from last base to first
  for (uint32_t i = 0; i < k; ++i)
  {
    //r += rank(str[k - i - 1]) * p;
    //p *= size();
    r += get_rank(str[k - i - 1]) << (i << 1);
  }
  return r;
}

//copy a kmer from a reference
static inline void kmer_cpy(char *dest, char *src, uint32_t k)
{
  uint32_t i = 0;
  for (i = 0; i < k; i++)
  {
    dest[i] = src[i];
  }
  dest[i] = '\0';
}
static inline float log_normal_pdf(float x, float gp_mean, float gp_stdv,
                                   float gp_log_stdv)
{
  /*INCOMPLETE*/
  float log_inv_sqrt_2pi = -0.918938f; // Natural logarithm
  float a = (x - gp_mean) / gp_stdv;
  return log_inv_sqrt_2pi - gp_log_stdv + (-0.5f * a * a);
  // return 1;
}

static inline float log_probability_match_r9(scalings_t scaling,
                                             model_t *models,
                                             event_t *events, int event_idx,
                                             uint32_t kmer_rank, uint8_t strand)
{
  // event level mean, scaled with the drift value
  strand = 0;
  assert(kmer_rank < 4096);
  //float level = read.get_drift_scaled_level(event_idx, strand);

  //float time =
  //    (events.event[event_idx].start - events.event[0].start) / sample_rate;
  float unscaledLevel = events[event_idx].mean;
  float scaledLevel = unscaledLevel;
  //float scaledLevel = unscaledLevel - time * scaling.shift;

  //fprintf(stderr, "level %f\n",scaledLevel);
  //GaussianParameters gp = read.get_scaled_gaussian_from_pore_model_state(pore_model, strand, kmer_rank);
  float gp_mean =
      scaling.scale * models[kmer_rank].level_mean + scaling.shift;
  float gp_stdv = models[kmer_rank].level_stdv * 1; //scaling.var = 1;
// float gp_stdv = 0;
// float gp_log_stdv = models[kmer_rank].level_log_stdv + scaling.log_var;
// if(models[kmer_rank].level_stdv <0.01 ){
// 	fprintf(stderr,"very small std dev %f\n",models[kmer_rank].level_stdv);
// }
#ifdef CACHED_LOG
  float gp_log_stdv = models[kmer_rank].level_log_stdv;
#else
  float gp_log_stdv =
      log(models[kmer_rank].level_stdv); // scaling.log_var = log(1)=0;
#endif

  float lp = log_normal_pdf(scaledLevel, gp_mean, gp_stdv, gp_log_stdv);
  return lp;
}

#define event_kmer_to_band(ei, ki) (ei + 1) + (ki + 1)
#define band_event_to_offset(bi, ei) band_lower_left[bi].event_idx - (ei)
#define band_kmer_to_offset(bi, ki) (ki) - band_lower_left[bi].kmer_idx
#define is_offset_valid(offset) (offset) >= 0 && (offset) < bandwidth
#define event_at_offset(bi, offset) band_lower_left[(bi)].event_idx - (offset)
#define kmer_at_offset(bi, offset) band_lower_left[(bi)].kmer_idx + (offset)

#define move_down(curr_band)                    \
  {                                             \
    curr_band.event_idx + 1, curr_band.kmer_idx \
  }
#define move_right(curr_band)                   \
  {                                             \
    curr_band.event_idx, curr_band.kmer_idx + 1 \
  }

#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#define MAX(a, b) (((a) > (b)) ? (a) : (b))

#ifdef ALIGN_2D_ARRAY
#define BAND_ARRAY(r, c) (bands[(r)][(c)])
#define TRACE_ARRAY(r, c) (trace[(r)][(c)])
#else
#define BAND_ARRAY(r, c) (bands[((r) * (ALN_BANDWIDTH) + (c))])
#define TRACE_ARRAY(r, c) (trace[((r) * (ALN_BANDWIDTH) + (c))])
#endif

#define event_kmer_to_band(ei, ki) (ei + 1) + (ki + 1)
#define band_event_to_offset(bi, ei) band_lower_left[bi].event_idx - (ei)
#define band_kmer_to_offset(bi, ki) (ki) - band_lower_left[bi].kmer_idx
#define is_offset_valid(offset) (offset) >= 0 && (offset) < bandwidth
#define event_at_offset(bi, offset) band_lower_left[(bi)].event_idx - (offset)
#define kmer_at_offset(bi, offset) band_lower_left[(bi)].kmer_idx + (offset)

void host_post_processing(AlignedPair *event_align_pairs,
                          int32_t *n_event_align_pairs,
                          char *read, int32_t *read_len,
                          ptr_t *read_ptr, event_t *event_table,
                          int32_t *n_events1, ptr_t *event_ptr,
                          scalings_t *scalings, model_t *models,
                          int32_t n_bam_rec, uint32_t *kmer_ranks1,
                          float *band, uint8_t *traces,
                          EventKmerPair *band_lower_lefts)
{
  for (int32_t ii = 0; ii < n_bam_rec; ii++)
  {
    // fprintf(stderr, "%s\n", sequence);
    // fprintf(stderr, "Scaling %f %f", scaling.scale, scaling.shift);
    // printf("read:%lu\n", ii);
    // AlignedPair* out_2 = db->event_align_pairs[i];
    // char* sequence = db->read[i];
    // int32_t sequence_len = db->read_len[i];
    // event_table events = db->et[i];
    // model_t* models = core->model;
    // scalings_t scaling = db->scalings[i];
    // float sample_rate = db->f5[i]->sample_rate;

    AlignedPair *out_2 = &event_align_pairs[event_ptr[ii] * 2];
    char *sequence = &read[read_ptr[ii]];
    int32_t sequence_len = read_len[ii];
    // printf("read_len[%lu] = %d\n", ii, read_len[ii]);
    event_t *events = &event_table[event_ptr[ii]];
    int32_t n_event = n_events1[ii];

    scalings_t scaling = scalings[ii];

    float *bands =
        &band[(read_ptr[ii] + event_ptr[ii]) * ALN_BANDWIDTH];

    uint8_t *trace =
        &traces[(read_ptr[ii] + event_ptr[ii]) * ALN_BANDWIDTH];

    EventKmerPair *band_lower_left =
        &band_lower_lefts[read_ptr[ii] + event_ptr[ii]];

    uint32_t *kmer_ranks = &kmer_ranks1[read_ptr[ii]];

    // size_t n_events = events[strand_idx].n;
    int32_t n_events = n_event; // <------ diff
    // printf("n_events= %ld\n", n_events);
    int32_t n_kmers = sequence_len - KMER_SIZE + 1;

    // printf("n_kmers : %lu\n", n_kmers);

    // transition penalties
    double events_per_kmer = (double)n_events / n_kmers;
    double p_stay = 1 - (1 / (events_per_kmer + 1));

    // setting a tiny skip penalty helps keep the true alignment within the
    // adaptive band this was empirically determined

    double lp_skip = log(epsilon);
    double lp_stay = log(p_stay);
    double lp_step = log(1.0 - exp(lp_skip) - exp(lp_stay));
    double lp_trim = log(0.01);
    // printf("lp_step %lf \n", lp_step);

    // dp matrix
    int32_t n_rows = n_events + 1;
    int32_t n_cols = n_kmers + 1;
    int32_t n_bands = n_rows + n_cols;

    //
    // Backtrack to compute alignment
    //
    float sum_emission = 0;
    float n_aligned_events = 0;

    //>>>>>>>>>>>>>> New replacement begin
    // std::vector<AlignedPair> out;

    int outIndex = 0;
    //<<<<<<<<<<<<<<<<New Replacement over

    double max_score = -INFINITY;
    int curr_event_idx = 0;
    int curr_kmer_idx = n_kmers - 1;

    // Find best score between an event and the last k-mer. after trimming the remaining evnets
    for (int32_t event_idx = 0; event_idx < n_events; ++event_idx)
    {
      int band_idx = event_kmer_to_band(event_idx, curr_kmer_idx);

      //>>>>>>>New  replacement begin
      /*assert(band_idx < bands.size());*/
      // assert((size_t)band_idx < n_bands);
      //<<<<<<<<New Replacement over
      int offset = band_event_to_offset(band_idx, event_idx);
      // fprintf(stderr, "%lu ", event_idx);
      if (is_offset_valid(offset))
      {
        float s =
            BAND_ARRAY(band_idx, offset) + (n_events - event_idx) * lp_trim;
        if (s > max_score)
        {
          max_score = s;
          curr_event_idx = event_idx;
        }
      }
    }
    // fprintf(stderr, "post:1");
#ifdef DEBUG_ADAPTIVE
    fprintf(stderr, "[adaback] ei: %d ki: %d s: %.2f\n", curr_event_idx,
            curr_kmer_idx, max_score);
#endif

    int curr_gap = 0;
    int max_gap = 0;
    while (curr_kmer_idx >= 0 && curr_event_idx >= 0)
    {
      // emit alignment
      //>>>>>>>New Repalcement begin
      // assert(outIndex < (n_events * 2));
      out_2[outIndex].ref_pos = curr_kmer_idx;
      out_2[outIndex].read_pos = curr_event_idx;
      outIndex++;
      // out.push_back({curr_kmer_idx, curr_event_idx});
      //<<<<<<<<<New Replacement over

#ifdef DEBUG_ADAPTIVE
      fprintf(stderr, "[adaback] ei: %d ki: %d\n", curr_event_idx,
              curr_kmer_idx);
#endif
      // qc stats
      //>>>>>>>>>>>>>>New Replacement begin
      char *substring = &sequence[curr_kmer_idx];
      uint32_t kmer_rank = get_kmer_rank(substring, KMER_SIZE);
      //<<<<<<<<<<<<<New Replacement over
      float tempLogProb = log_probability_match_r9(
          scaling, models, events, curr_event_idx, kmer_rank, 0);

      sum_emission += tempLogProb;
      //fprintf(stderr, "lp_emission %f \n", tempLogProb);
      //fprintf(stderr,"lp_emission %f, sum_emission %f, n_aligned_events %d\n",tempLogProb,sum_emission,outIndex);

      n_aligned_events += 1;

      int band_idx = event_kmer_to_band(curr_event_idx, curr_kmer_idx);
      int offset = band_event_to_offset(band_idx, curr_event_idx);

      // fprintf(stderr, "%d ", band_idx);

      // assert(band_kmer_to_offset(band_idx, curr_kmer_idx) == offset);

      uint8_t from = TRACE_ARRAY(band_idx, offset);
      if (from == FROM_D)
      {
        curr_kmer_idx -= 1;
        curr_event_idx -= 1;
        curr_gap = 0;
      }
      else if (from == FROM_U)
      {
        curr_event_idx -= 1;
        curr_gap = 0;
      }
      else
      {
        curr_kmer_idx -= 1;
        curr_gap += 1;
        max_gap = MAX(curr_gap, max_gap);
      }
    }
// fprintf(stderr, "post:2");
    //>>>>>>>>New replacement begin
    // std::reverse(out.begin(), out.end());
    int c;
    int end = outIndex - 1;
    for (c = 0; c < outIndex / 2; c++)
    {
      int ref_pos_temp = out_2[c].ref_pos;
      int read_pos_temp = out_2[c].read_pos;
      out_2[c].ref_pos = out_2[end].ref_pos;
      out_2[c].read_pos = out_2[end].read_pos;
      out_2[end].ref_pos = ref_pos_temp;
      out_2[end].read_pos = read_pos_temp;
      end--;
    }
// fprintf(stderr, "post:3");
    // if(outIndex>1){
    //   AlignedPair temp={out_2[0].ref_pos,out[0].read_pos};
    //   int i;
    //   for(i=0;i<outIndex-1;i++){
    //     out_2[i]={out_2[outIndex-1-i].ref_pos,out[outIndex-1-i].read_pos};
    //   }
    //   out[outIndex-1]={temp.ref_pos,temp.read_pos};
    // }
    //<<<<<<<<<New replacement over

    // QC results
    double avg_log_emission = sum_emission / n_aligned_events;
    //fprintf(stderr,"sum_emission %f, n_aligned_events %f, avg_log_emission %f\n",sum_emission,n_aligned_events,avg_log_emission);
    //>>>>>>>>>>>>>New replacement begin
    bool spanned = out_2[0].ref_pos == 0 &&
                   out_2[outIndex - 1].ref_pos == int(n_kmers - 1);
    // bool spanned = out.front().ref_pos == 0 && out.back().ref_pos == n_kmers - 1;
    //<<<<<<<<<<<<<New replacement over
    //bool failed = false;
    if (avg_log_emission < min_average_log_emission || !spanned ||
        max_gap > max_gap_threshold)
    {
      //failed = true;
      //>>>>>>>>>>>>>New replacement begin
      outIndex = 0;
      // out.clear();
      //free(out_2);
      //out_2 = NULL;
      //<<<<<<<<<<<<<New replacement over
    }
// fprintf(stderr, "post:4");
    //fprintf(stderr, "ada\t%s\t%s\t%.2lf\t%zu\t%.2lf\t%d\t%d\t%d\n", read.read_name.substr(0, 6).c_str(), failed ? "FAILED" : "OK", events_per_kmer, sequence.size(), avg_log_emission, curr_event_idx, max_gap, fills);
    //outSize=outIndex;
    //if(outIndex>500000)fprintf(stderr, "Max outSize %d\n", outIndex);
    n_event_align_pairs[ii] = outIndex;
  }
}