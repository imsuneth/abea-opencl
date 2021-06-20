
#include "f5c.h"
// #include <assert.h>
// #include <assert.h>
// #include "f5cmisc.cuh"
#define OFFSET_LOOP_UNROLL_FACTOR 4 // for de5net a7; try higher for arria 10
//#define DEBUG_ESTIMATED_SCALING 1
//#define DEBUG_RECALIB_SCALING 1
//#define DEBUG_ADAPTIVE 1
// #define CACHED_LOG 1

// From f5cmisc.cuh*****************************
// #define ALIGN_KERNEL_FLOAT 1 //(for 2d kernel only)
// #define WARP_HACK 1 // whether the kernels are  performed in 1D with a warp
// hack (effective only  if specific TWODIM_ALIGN is not defined)
// #define REVERSAL_ON_CPU \
//   1 // reversal of the backtracked array is performed on the CPU instead of
//   the
//     // GPU

//*********************************************

// todo : can make more efficient using bit encoding
inline uint32_t get_rank(char base) {
  // printf("%c ", base);
  if (base == 'A') { // todo: do we neeed simple alpha?
    return 0;
  } else if (base == 'C') {
    return 1;
  } else if (base == 'G') {
    return 2;
  } else if (base == 'T') {
    return 3;
  } else {
    // WARNING("A None ACGT base found : %c", base);
    return 0;
  }
}

// return the lexicographic rank of the kmer amongst all strings of
// length k for this alphabet
inline uint32_t get_kmer_rank(__global char *str) {
  // uint32_t p = 1;
  uint32_t r = 0;

// from last base to first
#pragma unroll
  for (uint32_t i = 0; i < KMER_SIZE; ++i) {
    // r += rank(str[k - i - 1]) * p;
    // p *= size();
    r += get_rank(str[KMER_SIZE - i - 1]) << (i << 1);
  }
  // printf("%d ", r);
  return r;
}

// copy a kmer from a reference
inline void kmer_cpy(char *dest, char *src, uint32_t k) {
  uint32_t i = 0;
  for (i = 0; i < k; i++) {
    dest[i] = src[i];
  }
  dest[i] = '\0';
}

#define log_inv_sqrt_2pi -0.918938f // Natural logarithm

inline float log_normal_pdf(float x, float gp_mean, float gp_stdv,
                            float gp_log_stdv) {
  /*INCOMPLETE*/
  // float log_inv_sqrt_2pi = -0.918938f; // Natural logarithm
  float a = (x - gp_mean) / gp_stdv;
  return log_inv_sqrt_2pi - gp_log_stdv + (-0.5f * a * a);
  // return 1;
}

inline float log_probability_match_r9(scalings_t scaling,
                                      __global model_t *restrict models,
                                      __global event1_t *restrict events,
                                      int event_idx, uint32_t kmer_rank) {
  // event level mean, scaled with the drift value
  // strand = 0;
  // assert(kmer_rank < 4096);
  // float level = read.get_drift_scaled_level(event_idx, strand);

  // float time =
  //    (events.event[event_idx].start - events.event[0].start) / sample_rate;
  // float unscaledLevel = events.event[event_idx].mean;
  float unscaledLevel = events[event_idx].mean;
  float scaledLevel = unscaledLevel;
  // printf("%f ", scaledLevel);
  // float scaledLevel = unscaledLevel - time * scaling.shift;

  // fprintf(stderr, "level %f\n",scaledLevel);
  // GaussianParameters gp =
  // read.get_scaled_gaussian_from_pore_model_state(pore_model, strand,
  // kmer_rank);
  float gp_mean = scaling.scale * models[kmer_rank].level_mean + scaling.shift;
  float gp_stdv = models[kmer_rank].level_stdv * 1; // scaling.var = 1;
                                                    // float gp_stdv = 0;
  // float gp_log_stdv = models[kmer_rank].level_log_stdv + scaling.log_var;
  // if(models[kmer_rank].level_stdv <0.01 ){
  // 	fprintf(stderr,"very small std dev %f\n",models[kmer_rank].level_stdv);
  // }

  float gp_log_stdv =
      log(models[kmer_rank].level_stdv); // scaling.log_var = log(1)=0;

  float lp = log_normal_pdf(scaledLevel, gp_mean, gp_stdv, gp_log_stdv);
  return lp;
}

#define event_kmer_to_band(ei, ki) (ei + 1) + (ki + 1)
#define band_event_to_offset(bi, ei) band_lower_left[bi].event_idx - (ei)
#define band_kmer_to_offset(bi, ki) (ki) - band_lower_left[bi].kmer_idx
#define is_offset_valid(offset) (offset) >= 0 && (offset) < bandwidth
#define event_at_offset(bi, offset) band_lower_left[(bi)].event_idx - (offset)
#define kmer_at_offset(bi, offset) band_lower_left[(bi)].kmer_idx + (offset)

// #define move_down(curr_band)                                                   \
//     { curr_band.event_idx + 1, curr_band.kmer_idx }
// #define move_right(curr_band)                                                  \
//     { curr_band.event_idx, curr_band.kmer_idx + 1 }

#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#define MAX(a, b) (((a) > (b)) ? (a) : (b))

// #ifdef ALIGN_2D_ARRAY
// #define BAND_ARRAY(r, c) (bands[(r)][(c)])
// #define TRACE_ARRAY(r, c) (trace[(r)][(c)])
// #else
#define BAND_ARRAY(r, c) (bands[((r) * (ALN_BANDWIDTH) + (c))])
#define TRACE_ARRAY(r, c) (trace[((r) * (ALN_BANDWIDTH) + (c))])
// #endif

#define FROM_D 0
#define FROM_U 1
#define FROM_L 2

#define max_gap_threshold 50
#define bandwidth ALN_BANDWIDTH
#define half_bandwidth ALN_BANDWIDTH / 2

#ifndef ALIGN_KERNEL_FLOAT
#define min_average_log_emission -5.0
#define epsilon 1e-10
#else
#define min_average_log_emission -5.0f
#define epsilon 1e-10f
#endif

// #define DEBUG_KERNEL
// __attribute__((max_work_group_size(1024)))
//
// __attribute__((reqd_work_group_size(1, 1, 1)))
// __attribute__((num_compute_units(1)))
__attribute__((task)) __kernel void align_kernel_single(
    // __global AlignedPair *restrict event_align_pairs,
    // __global int32_t *restrict n_event_align_pairs,
    __global char *restrict read, __global int32_t *restrict read_len,
    __global ptr_t *restrict read_ptr, __global event1_t *restrict event_table,
    __global int32_t *restrict n_events1, __global ptr_t *restrict event_ptr,
    __global scalings_t *restrict scalings, __global model_t *restrict models,
    int32_t n_bam_rec, __global uint32_t *restrict kmer_ranks1,
    __global float *restrict bands1, __global uint8_t *restrict trace1,
    __global EventKmerPair *restrict band_lower_lefts) {

// size_t ii = get_global_id(0);
// printf("START!!!!!!!!!!!!!!!\n");

// #pragma ivdep array(read_len)
// #pragma ivdep array(read_ptr)
// #pragma ivdep array(event_table)
// #pragma ivdep array(n_events1)
// #pragma ivdep array(event_ptr)
// #pragma ivdep array(scalings)
// #pragma ivdep array(models)
// #pragma ivdep array(kmer_ranks1)
// #pragma ivdep array(bands1)
// #pragma ivdep array(trace1)
// #pragma ivdep array(band_lower_lefts)
#pragma ii 1
#pragma ivdep
  for (int32_t ii = 0; ii < n_bam_rec; ii++) {
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

    // __global AlignedPair *out_2 = &event_align_pairs[event_ptr[ii] * 2];
    // __global char *sequence = &read[read_ptr[ii]];
    __global char *sequence = read + read_ptr[ii];
    int32_t sequence_len = read_len[ii];
    // printf("read_len[%lu] = %d\n", ii, read_len[ii]);

    // __global event1_t *events = &event_table[event_ptr[ii]];
    __global event1_t *events = event_table + event_ptr[ii];

    // printf("start %d \n", events->start);

    int32_t n_events = n_events1[ii];
    // printf("n_events %d\n",n_events);
    scalings_t scaling = scalings[ii];

    // __global float *bands =
    //     &bands1[(read_ptr[ii] + event_ptr[ii]) * ALN_BANDWIDTH];
    __global float *bands =
        bands1 + (read_ptr[ii] + event_ptr[ii]) * ALN_BANDWIDTH;

    // __global uint8_t *trace =
    //     &trace1[(read_ptr[ii] + event_ptr[ii]) * ALN_BANDWIDTH];
    __global uint8_t *trace =
        trace1 + (read_ptr[ii] + event_ptr[ii]) * ALN_BANDWIDTH;

    // __global EventKmerPair *band_lower_left =
    //     &band_lower_lefts[read_ptr[ii] + event_ptr[ii]];
    __global EventKmerPair *band_lower_left =
        band_lower_lefts + read_ptr[ii] + event_ptr[ii];

    // __global uint32_t *kmer_ranks = &kmer_ranks1[read_ptr[ii]];
    __global uint32_t *kmer_ranks = kmer_ranks1 + read_ptr[ii];

    // int32_t n_events = n_event; // <------ diff
    int32_t n_kmers = sequence_len - KMER_SIZE + 1;
    // printf("n_kmers %d\n",n_kmers);
    // transition penalties
    double events_per_kmer = (double)n_events / n_kmers;
    // printf("events_per_kmer %lf \n",events_per_kmer);
    double p_stay = 1 - (1 / (events_per_kmer + 1));

    // setting a tiny skip penalty helps keep the true alignment within the
    // adaptive band this was empirically determined

    // printf("n_events:%d, event_table:\n",n_events);
    // for(int j=0; j<n_events; j++){
    //   printf("%f ", events[j].mean);
    // }
    // printf("\nend of event_table");

    // #ifndef ALIGN_KERNEL_FLOAT
    double lp_skip = log(epsilon);
    double lp_stay = log(p_stay);
    double lp_step = log(1.0 - exp(lp_skip) - exp(lp_stay));
    double lp_trim = log(0.01);
    // #else
    //     float lp_skip = logf(epsilon);
    //     float lp_stay = logf(p_stay);
    //     float lp_step = logf(1.0f - expf(lp_skip) - expf(lp_stay));
    //     float lp_trim = logf(0.01f);
    // #endif

    // dp matrix
    int32_t n_rows = n_events + 1;
    int32_t n_cols = n_kmers + 1;
    int32_t n_bands = n_rows + n_cols;

#ifdef DEBUG_KERNEL
    printf("lp_skip:%f\n", lp_skip);
    printf("lp_stay:%f\n", lp_stay);
    printf("lp_step:%f\n", lp_step);
    printf("lp_trim:%f\n", lp_trim);
    // printf("Sequcen:\n");
    // for (int kk = 0; kk < sequence_len; kk++) {
    //   printf("%c", sequence[kk]);
    // }
    // printf("\nSequcen end\n");
    printf("sequence_len:%d\n", sequence_len);
    printf("n_events:%d\n", n_events);
    printf("events[0].mean:%f\n", events[0].mean);
    printf("scaling.scale:%f\n", scaling.scale);
    printf("n_bands:%d\n", n_bands);
#endif

    // #pragma unroll
    for (int32_t i = 0; i < n_kmers; ++i) {
      //>>>>>>>>> New replacement begin
      __global char *substring = &sequence[i];
      kmer_ranks[i] = get_kmer_rank(substring);
      //<<<<<<<<< New replacement over
    }

#ifdef DEBUG_KERNEL
    // printf("kmer_ranks:\n");
    // for (int kk = 0; kk < n_kmers; kk++) {
    //   printf("%d ", kmer_ranks[kk]);
    // }
    // printf("\nkmer_ranks end\n");
#endif
#pragma ivdep array(bands)
#pragma ivdep array(trace)
    for (int32_t i = 0; i < n_bands; i++) {
#pragma unroll
      for (int32_t j = 0; j < bandwidth; j++) {
        BAND_ARRAY(i, j) = -INFINITY;
        TRACE_ARRAY(i, j) = 0;
      }
    }

    // initialize range of first two
    band_lower_left[0].event_idx = half_bandwidth - 1;
    band_lower_left[0].kmer_idx = -1 - half_bandwidth;
    // band_lower_left[1] = move_down(band_lower_left[0]);
    band_lower_left[1].kmer_idx = band_lower_left[0].kmer_idx;
    band_lower_left[1].event_idx = band_lower_left[0].event_idx + 1;

    int start_cell_offset = band_kmer_to_offset(0, -1);
    // assert(is_offset_valid(start_cell_offset));
    // assert(band_event_to_offset(0, -1) == start_cell_offset);
    BAND_ARRAY(0, start_cell_offset) = 0.0f;

    // band 1: first event is trimmed
    int first_trim_offset = band_event_to_offset(1, 0);
    // assert(kmer_at_offset(1, first_trim_offset) == -1);
    // assert(is_offset_valid(first_trim_offset));
    BAND_ARRAY(1, first_trim_offset) = lp_trim;
    TRACE_ARRAY(1, first_trim_offset) = FROM_U;

    // int fills = 0;
#ifdef DEBUG_ADAPTIVE
    printf("[trim] bi: %d o: %d e: %d k: %d s: %.2lf\n", 1, first_trim_offset,
           0, -1, BAND_ARRAY(1, first_trim_offset));
#endif

    // printf("INNER_LOOP!!!!!!!!!!!!!!!\n");
    // fill in remaining bands
    // printf("n_bands %lu\n", n_bands);
    // #pragma unroll
    // bool odd_band_idx = true;
    // #pragma ivdep array(events)
    // #pragma ivdep array(models)
    // #pragma ivdep array(kmer_ranks)
    // #pragma ivdep array(traces)
    // #pragma loop_coalesce 2

    bool odd_band_idx = true;

    // #pragma unroll 4
    for (int32_t band_idx = 2; band_idx < n_bands; ++band_idx) {
      odd_band_idx = !odd_band_idx;
      // if (band_idx < n_bands) {

      // Determine placement of this band according to Suzuki's adaptive
      // algorithm When both ll and ur are out-of-band (ob) we alternate
      // movements otherwise we decide based on scores
      float ll = BAND_ARRAY(band_idx - 1, 0);
      // printf("%f ",ll);
      float ur = BAND_ARRAY(band_idx - 1, bandwidth - 1);
      bool ll_ob = ll == -INFINITY;
      bool ur_ob = ur == -INFINITY;
#ifdef DEBUG_KERNEL
      // printf("band_idx:%d, ll:%f, ur:%f\n", band_idx, ll, ur);
#endif
      bool right = false;
      if (ll_ob && ur_ob) {
        // right = band_idx % 2 == 1;
        right = odd_band_idx;
      } else {
        right = ll < ur; // Suzuki's rule
      }
      EventKmerPair bbl = band_lower_left[band_idx - 1];
      if (right) {
        // band_lower_left[band_idx] = move_right(band_lower_left[band_idx -
        // 1]);

        // band_lower_left[band_idx] = band_lower_left[band_idx - 1];
        // band_lower_left[band_idx].kmer_idx++;

        bbl.kmer_idx++;

        // band_lower_left[band_idx].event_idx =
        //     band_lower_left[band_idx - 1].event_idx;
        // band_lower_left[band_idx].kmer_idx =
        //     band_lower_left[band_idx - 1].kmer_idx + 1;
        // printf("band_idx:%d, move_right\n", band_idx);

      } else {
        // band_lower_left[band_idx] = move_down(band_lower_left[band_idx -
        // 1]);

        // band_lower_left[band_idx] = band_lower_left[band_idx - 1];
        // band_lower_left[band_idx].event_idx++;

        bbl.event_idx++;

        // band_lower_left[band_idx].event_idx =
        //     band_lower_left[band_idx - 1].event_idx + 1;
        // band_lower_left[band_idx].kmer_idx =
        //     band_lower_left[band_idx - 1].kmer_idx;
        // printf("band_idx:%d, move_down\n", band_idx);
      }
      band_lower_left[band_idx] = bbl;

      // int trim_offset = band_kmer_to_offset(band_idx, -1);
      int trim_offset = (-1) - bbl.kmer_idx;
      // printf("%d ", trim_offset);
      if (is_offset_valid(trim_offset)) {
        // int64_t event_idx = event_at_offset(band_idx, trim_offset);
        int32_t event_idx = bbl.event_idx - (trim_offset);
        // printf("%ld ", event_idx);
        if (event_idx >= 0 && event_idx < (int64_t)n_events) {
          BAND_ARRAY(band_idx, trim_offset) = lp_trim * (event_idx + 1);
          TRACE_ARRAY(band_idx, trim_offset) = FROM_U;
        } else {
          BAND_ARRAY(band_idx, trim_offset) = -INFINITY;
        }
      }

      // Get the offsets for the first and last event and kmer
      // We restrict the inner loop to only these values
      // printf("%d ", band_lower_left[band_idx].kmer_idx);
      // printf("%d ", band_idx);

      // int kmer_min_offset = band_kmer_to_offset(band_idx, 0);
      int kmer_min_offset = 0 - bbl.kmer_idx;
      // int kmer_max_offset = band_kmer_to_offset(band_idx, n_kmers);
      int kmer_max_offset = n_kmers - bbl.kmer_idx;
      // int event_min_offset = band_event_to_offset(band_idx, n_events - 1);
      int event_min_offset = bbl.event_idx - (n_events - 1);
      // int event_max_offset = band_event_to_offset(band_idx, -1);
      int event_max_offset = bbl.event_idx - (-1);

      int min_offset = MAX(kmer_min_offset, event_min_offset);
      min_offset = MAX(min_offset, 0);

      int max_offset = MIN(kmer_max_offset, event_max_offset);
      max_offset = MIN(max_offset, bandwidth);
#ifdef DEBUG_KERNEL
// printf("band_idx:%d, toffest:%d, kmin:%d, kmax:%d, emin:%d, emax:%d,
// min:%d, max:%d\n", band_idx, trim_offset,
// kmer_min_offset,kmer_max_offset,event_min_offset,event_max_offset,
// min_offset, max_offset);
#endif

#pragma ivdep array(bands)
#pragma ivdep array(trace)
#pragma ivdep array(band_lower_left)
      // for (int offset = 0; offset < ALN_BANDWIDTH; ++offset) {

#pragma unroll                                                                 \
    OFFSET_LOOP_UNROLL_FACTOR // for de5net a7; try higher for arria 10
      for (int offset = min_offset; offset < max_offset; ++offset) {

        // if (offset >= min_offset && offset < max_offset) {

        // int event_idx = event_at_offset(band_idx, offset);
        // int kmer_idx = kmer_at_offset(band_idx, offset);
        int event_idx = bbl.event_idx - offset;
        int kmer_idx = bbl.kmer_idx + offset;

        int32_t kmer_rank = kmer_ranks[kmer_idx];
        // printf("%ld ", kmer_rank);

        int offset_up = band_event_to_offset(band_idx - 1, event_idx - 1);
        int offset_left = band_kmer_to_offset(band_idx - 1, kmer_idx - 1);
        int offset_diag = band_kmer_to_offset(band_idx - 2, kmer_idx - 1);

        float up = is_offset_valid(offset_up)
                       ? BAND_ARRAY(band_idx - 1, offset_up)
                       : -INFINITY;
        float left = is_offset_valid(offset_left)
                         ? BAND_ARRAY(band_idx - 1, offset_left)
                         : -INFINITY;
        float diag = is_offset_valid(offset_diag)
                         ? BAND_ARRAY(band_idx - 2, offset_diag)
                         : -INFINITY;

        float lp_emission = log_probability_match_r9(scaling, models, events,
                                                     event_idx, kmer_rank);
        // fprintf(stderr, "lp emiision : %f , event idx %d, kmer rank
        // %d\n", lp_emission,event_idx,kmer_rank);
        // printf("%f ",lp_emission);
        float score_d = diag + lp_step + lp_emission;
        float score_u = up + lp_stay + lp_emission;
        float score_l = left + lp_skip;

        float max_score = score_d;
        uint8_t from = FROM_D;

        max_score = score_u > max_score ? score_u : max_score;
        from = max_score == score_u ? FROM_U : from;
        max_score = score_l > max_score ? score_l : max_score;
        from = max_score == score_l ? FROM_L : from;

        BAND_ARRAY(band_idx, offset) = max_score;
        // printf("%f ", max_score);
        TRACE_ARRAY(band_idx, offset) = from;
        // fills += 1;
        // printf("[trim] bi: %d o: %d e: %d k: %d s: %.2lf\n", 1,
        // first_trim_offset, 0, -1, BAND_ARRAY(1, first_trim_offset));
        // }
        // } // loop offset

        // } // if (band_idx < n_bands)
        // else {
        //   break;
      }
    } // for (int32_t band_idx = 2; band_idx < 100000; ++band_idx)
#ifdef DEBUG_KERNEL
    printf("fills:%d\n", fills);
    printf("band_lower_left:\n");
    // for (int kk = 0; kk < sequence_len; kk++) {
    //   printf("%d ", band_lower_left[kk].event_idx);
    //   printf("%d ", band_lower_left[kk].kmer_idx);
    // }
    printf("\band_lower_left end\n");
#endif

    // //
    // // Backtrack to compute alignment
    // //
    // double sum_emission = 0;
    // double n_aligned_events = 0;

    // //>>>>>>>>>>>>>> New replacement begin
    // // std::vector<AlignedPair> out;

    // int outIndex = 0;
    // //<<<<<<<<<<<<<<<<New Replacement over

    // float max_score = -INFINITY;
    // int curr_event_idx = 0;
    // int curr_kmer_idx = n_kmers - 1;

    // // Find best score between an event and the last k-mer. after trimming
    // the
    // // remaining evnets
    // for (size_t event_idx = 0; event_idx < n_events; ++event_idx) {
    //   int band_idx = event_kmer_to_band(event_idx, curr_kmer_idx);

    //   //>>>>>>>New  replacement begin
    //   /*assert(band_idx < bands.size());*/
    //   // assert((size_t)band_idx < n_bands);
    //   //<<<<<<<<New Replacement over
    //   int offset = band_event_to_offset(band_idx, event_idx);
    //   if (is_offset_valid(offset)) {
    //     float s =
    //         BAND_ARRAY(band_idx, offset) + (n_events - event_idx) * lp_trim;
    //     if (s > max_score) {
    //       max_score = s;
    //       curr_event_idx = event_idx;
    //     }
    //   }
    // }
    //   if (is_offset_valid(offset)) {
    //     float s =
    //         BAND_ARRAY(band_idx, offset) + (n_events - event_idx) * lp_trim;
    //     if (s > max_score) {
    //       max_score = s;
    //       curr_event_idx = event_idx;
    //     }
    //   }
    // }

    // #ifdef DEBUG_ADAPTIVE
    //     fprintf(stderr, "[adaback] ei: %d ki: %d s: %.2f\n", curr_event_idx,
    //             curr_kmer_idx, max_score);
    // #endif

    //     int curr_gap = 0;
    //     int max_gap = 0;
    //     while (curr_kmer_idx >= 0 && curr_event_idx >= 0) {
    //       // emit alignment
    //       //>>>>>>>New Repalcement begin
    //       // assert(outIndex < (int)(n_events * 2));
    //       out_2[outIndex].ref_pos = curr_kmer_idx;
    //       out_2[outIndex].read_pos = curr_event_idx;
    //       outIndex++;
    //       // out.push_back({curr_kmer_idx, curr_event_idx});
    //       //<<<<<<<<<New Replacement over

    // #ifdef DEBUG_ADAPTIVE
    //       fprintf(stderr, "[adaback] ei: %d ki: %d\n", curr_event_idx,
    //               curr_kmer_idx);
    // #endif
    //       // qc stats
    //       //>>>>>>>>>>>>>>New Replacement begin
    //       __global char *substring = &sequence[curr_kmer_idx];
    //       size_t kmer_rank = get_kmer_rank(substring, KMER_SIZE);
    //       //<<<<<<<<<<<<<New Replacement over
    //       float tempLogProb = log_probability_match_r9(scaling, models,
    //       events,
    //                                                    curr_event_idx,
    //                                                    kmer_rank);

    //       sum_emission += tempLogProb;
    //       // fprintf(stderr, "lp_emission %f \n", tempLogProb);
    //       // fprintf(stderr,"lp_emission %f, sum_emission %f,
    //       n_aligned_events
    //       // %d\n",tempLogProb,sum_emission,outIndex);

    //       n_aligned_events += 1;

    //       int band_idx = event_kmer_to_band(curr_event_idx, curr_kmer_idx);
    //       int offset = band_event_to_offset(band_idx, curr_event_idx);
    //       // assert(band_kmer_to_offset(band_idx, curr_kmer_idx) == offset);

    //       uint8_t from = TRACE_ARRAY(band_idx, offset);
    //       if (from == FROM_D) {
    //         curr_kmer_idx -= 1;
    //         curr_event_idx -= 1;
    //         curr_gap = 0;
    //       } else if (from == FROM_U) {
    //         curr_event_idx -= 1;
    //         curr_gap = 0;
    //       } else {
    //         curr_kmer_idx -= 1;
    //         curr_gap += 1;
    //         max_gap = MAX(curr_gap, max_gap);
    //       }
    //     }

    //     //>>>>>>>>New replacement begin
    //     // std::reverse(out.begin(), out.end());
    //     int c;
    //     int end = outIndex - 1;
    //     for (c = 0; c < outIndex / 2; c++) {
    //       int ref_pos_temp = out_2[c].ref_pos;
    //       int read_pos_temp = out_2[c].read_pos;
    //       out_2[c].ref_pos = out_2[end].ref_pos;
    //       out_2[c].read_pos = out_2[end].read_pos;
    //       out_2[end].ref_pos = ref_pos_temp;
    //       out_2[end].read_pos = read_pos_temp;
    //       end--;
    //     }

    //     //<<<<<<<<<New replacement over

    //     // QC results
    //     double avg_log_emission = sum_emission / n_aligned_events;
    //     // fprintf(stderr,"sum_emission %f, n_aligned_events %f,
    //     avg_log_emission
    //     // %f\n",sum_emission,n_aligned_events,avg_log_emission);
    //     //>>>>>>>>>>>>>New replacement begin
    //     bool spanned = out_2[0].ref_pos == 0 &&
    //                    out_2[outIndex - 1].ref_pos == (int)(n_kmers - 1);
    //     // bool spanned = out.front().ref_pos == 0 && out.back().ref_pos ==
    //     n_kmers
    //     // - 1;
    //     //<<<<<<<<<<<<<New replacement over
    //     // bool failed = false;
    //     if (avg_log_emission < min_average_log_emission || !spanned ||
    //         max_gap > max_gap_threshold) {
    //       // failed = true;
    //       //>>>>>>>>>>>>>New replacement begin
    //       outIndex = 0;
    //       // out.clear();
    //       // free(out_2);
    //       // out_2 = NULL;
    //       //<<<<<<<<<<<<<New replacement over
    //     }

    // // "Max outSize %d\n", outIndex);
    // n_event_align_pairs[ii] = outIndex;
    // // return outIndex;

  } // for (int32_t ii = 0; ii < n_bam_rec; ii++)
}