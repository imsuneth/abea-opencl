#include "f5c.h"
// #include <assert.h>
// #include "f5cmisc.cuh"

//#define DEBUG_ESTIMATED_SCALING 1
//#define DEBUG_RECALIB_SCALING 1
//#define DEBUG_ADAPTIVE 1

// From f5cmisc.cuh*****************************
// #define ALIGN_KERNEL_FLOAT 1 //(for 2d kernel only)
// #define WARP_HACK 1          // whether the kernels are  performed in 1D with a warp
// hack (effective only  if specific TWODIM_ALIGN is not defined)
// #define REVERSAL_ON_CPU \
//   1 // reversal of the backtracked array is performed on the CPU instead of
//   the
//     // GPU

//*********************************************

// todo : can make more efficient using bit encoding
uint32_t get_rank(char base)
{
    if (base == 'A')
    { // todo: do we neeed simple alpha?
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
        // WARNING("A None ACGT base found : %c", base);
        return 0;
    }
}

// return the lexicographic rank of the kmer amongst all strings of
// length k for this alphabet
inline uint32_t get_kmer_rank(__global char *str, uint32_t k)
{
    // uint32_t p = 1;
    uint32_t r = 0;

    // from last base to first
    for (uint32_t i = 0; i < k; ++i)
    {
        // r += rank(str[k - i - 1]) * p;
        // p *= size();
        r += get_rank(str[k - i - 1]) << (i << 1);
    }
    return r;
}

// copy a kmer from a reference
inline void kmer_cpy(char *dest, char *src, uint32_t k)
{
    uint32_t i = 0;
    for (i = 0; i < k; i++)
    {
        dest[i] = src[i];
    }
    dest[i] = '\0';
}

#define log_inv_sqrt_2pi -0.918938f // Natural logarithm

static inline float log_normal_pdf(float x, float gp_mean, float gp_stdv,
                                   float gp_log_stdv)
{
    /*INCOMPLETE*/
    // float log_inv_sqrt_2pi = -0.918938f; // Natural logarithm
    float a = (x - gp_mean) / gp_stdv;
    return log_inv_sqrt_2pi - gp_log_stdv + (-0.5f * a * a);
    // return 1;
}

inline float
log_probability_match_r9(scalings_t scaling, __global model_t *models,
                         event_table events, int event_idx, uint32_t kmer_rank,
                         uint8_t strand, float sample_rate)
{
    // event level mean, scaled with the drift value
    // strand = 0;
#ifdef DEBUG_ADAPTIVE
    // assert(kmer_rank < 4096);
#endif
    // float level = read.get_drift_scaled_level(event_idx, strand);

    // float time =
    //    (events.event[event_idx].start - events.event[0].start) / sample_rate;
    float unscaledLevel = events.event[event_idx].mean;
    float scaledLevel = unscaledLevel;
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

// #define move_down(curr_band){ curr_band.event_idx + 1, curr_band.kmer_idx }
// #define move_right(curr_band){ curr_band.event_idx, curr_band.kmer_idx + 1 }

#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#define MAX(a, b) (((a) > (b)) ? (a) : (b))

#define BAND_ARRAY(r, c) (bands[((r) * (ALN_BANDWIDTH) + (c))])
#define TRACE_ARRAY(r, c) (trace[((r) * (ALN_BANDWIDTH) + (c))])

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

// __global inline EventKmerPair move_down(EventKmerPair curr_band)
// {
//     EventKmerPair ret = {curr_band.event_idx + 1, curr_band.kmer_idx};
//     return ret;
// }
// __global inline EventKmerPair move_right(EventKmerPair curr_band)
// {
//     EventKmerPair ret = {curr_band.event_idx, curr_band.kmer_idx + 1};
//     return ret;
// }

#define PROFILE 1

#define band_event_to_offset_shm(bi, ei) \
    band_lower_left_shm[bi].event_idx - (ei)
#define band_kmer_to_offset_shm(bi, ki) (ki) - band_lower_left_shm[bi].kmer_idx

#define event_at_offset_shm(bi, offset) \
    band_lower_left_shm[(bi)].event_idx - (offset)
#define kmer_at_offset_shm(bi, offset) \
    band_lower_left_shm[(bi)].kmer_idx + (offset)

#define BAND_ARRAY_SHM(r, c) (bands_shm[(r)][(c)])