#define CACHED_LOG 1 //if the log values of scalings and the model k-mers are cached

struct twoArrays
{
    float a[10];
    float b[10];
    float result[10];
};

//from nanopolish
typedef struct
{
    int ref_pos;
    int read_pos;
} AlignedPair;

// a single event : adapted from taken from scrappie
typedef struct
{
    ulong start;
    float length; //todo : cant be made int?
    float mean;
    float stdv;
    //int32_t pos;   //todo : always -1 can be removed
    //int32_t state; //todo : always -1 can be removed
} event1_t;

// // event table : adapted from scrappie
typedef struct
{
    size_t n;     //todo : int32_t not enough?
    size_t start; //todo : always 0?
    size_t end;   //todo : always equal to n?
    //event1_t *event;
} event_table;


// //k-mer model
typedef struct
{
    float level_mean;
    float level_stdv;

#ifdef CACHED_LOG
    //calculated for efficiency
    float level_log_stdv;
#endif

#ifdef LOAD_SD_MEANSSTDV
    //float sd_mean;
    //float sd_stdv;
    //float weight;
#endif
} model_t;

// //scaling parameters for the signal : taken from nanopolish
typedef struct
{
    // direct parameters that must be set
    float scale;
    float shift;
    //float drift; = 0 always?
    float var; // set later when calibrating
    //float scale_sd;
    //float var_sd;

    // derived parameters that are cached for efficiency
#ifdef CACHED_LOG
    float log_var;
#endif
    //float scaled_var;
    //float log_scaled_var;
} scalings_t;