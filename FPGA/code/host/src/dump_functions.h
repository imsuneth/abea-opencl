int32_t load_no_of_batches(const char *dump_dir)
{
    int32_t no_of_batches;
    char filename[100];
    FILE *fp;
    size_t read_count;
    // no_of_batches
    snprintf(filename, sizeof(filename), "%s/%s", dump_dir,
             "no_of_batches.dat");
    fp = fopen(filename, "r");
    if (fp == NULL)
    {
        printf("Can not open %s\n", filename);
        exit(1);
    }
    read_count = fread(&(no_of_batches), sizeof(int32_t), 1, fp);
    assert(read_count == 1);
    fclose(fp);

    return no_of_batches;
}
void load_n_bam_rec(db_t *db, const char *batch_dir)
{

    char filename[100];
    FILE *fp;
    size_t read_count;

    // db->n_bam_rec
    snprintf(filename, sizeof(filename), "%s/%s", batch_dir, "n_bam_rec.dat");
    fp = fopen(filename, "r");
    if (fp == NULL)
        printf("Can not open %s\n", filename);
    read_count = fread(&(db->n_bam_rec), sizeof(int32_t), 1, fp);
    assert(read_count == 1);
    fclose(fp);
}

void load_core(core_t *core, const char *batch_dir)
{

    char filename[100];
    FILE *fp;
    size_t read_count;

    // core->model - models
    snprintf(filename, sizeof(filename), "%s/%s", batch_dir, "model.dat");
    fp = fopen(filename, "r");
    if (fp == NULL)
        printf("Can not open %s\n", filename);
    read_count = fread(core->model, sizeof(model_t), NUM_KMER, fp);
    assert(read_count == NUM_KMER);
    fclose(fp);
}

void load_read_inputs(db_t *db, int32_t i, const char *batch_dir)
{

    char read_dir[150];
    snprintf(read_dir, sizeof(read_dir), "%s/%ld", batch_dir, i);

    char filename[200];
    FILE *fp;
    size_t read_count;
    int32_t read_count2;

    // db->read_len[i] - sequence_len
    snprintf(filename, sizeof(filename), "%s/%s", read_dir, "read_len[i].dat");
    fp = fopen(filename, "r");
    if (fp == NULL)
        printf("Can not open %s\n", filename);
    read_count = fread(&(db->read_len[i]), sizeof(int32_t), 1, fp);
    assert(read_count == 1);
    fclose(fp);

    // db->read[i] - sequence
    snprintf(filename, sizeof(filename), "%s/%s", read_dir, "read[i].dat");
    fp = fopen(filename, "r");
    if (fp == NULL)
        printf("Can not open %s\n", filename);
    db->read[i] = (char *)malloc(sizeof(char) * (db->read_len[i] + 1));  //with null term
    read_count2 = fread(db->read[i], sizeof(char), db->read_len[i], fp); //read without null term
    db->read[i][db->read_len[i]] = '\0';
    assert(read_count2 == db->read_len[i]);
    fclose(fp);

    // db->et[i] - events
    snprintf(filename, sizeof(filename), "%s/%s", read_dir, "et[i].dat");
    fp = fopen(filename, "r");
    if (fp == NULL)
        printf("Can not open %s\n", filename);
    read_count = fread(&(db->et[i]), sizeof(event_table), 1, fp);
    size_t n_events = db->et[i].n;
    db->et[i].event = (event_t *)malloc(sizeof(event_t) * n_events);
    read_count = fread(db->et[i].event, sizeof(event_t), n_events, fp);
    assert(read_count == n_events);
    fclose(fp);

    // db->scalings[i] - scaling
    snprintf(filename, sizeof(filename), "%s/%s", read_dir, "scalings[i].dat");
    fp = fopen(filename, "r");
    if (fp == NULL)
        printf("Can not open %s\n", filename);
    read_count = fread(&(db->scalings[i]), sizeof(scalings_t), 1, fp);
    assert(read_count == 1);
    fclose(fp);

    // db->f5[i]->sample_rate - sample_rate
    snprintf(filename, sizeof(filename), "%s/%s", read_dir, "f5[i].sample_rate.dat");
    fp = fopen(filename, "r");
    if (fp == NULL)
        printf("Can not open %s\n", filename);
    db->f5[i] = (fast5_t *)malloc(sizeof(fast5_t));
    read_count = fread(&(db->f5[i]->sample_rate), sizeof(float), 1, fp);
    assert(read_count == 1);
    fclose(fp);
}

void load_read_outputs(db_t *db, int32_t i, const char *batch_dir)
{

    char read_dir[150];
    snprintf(read_dir, sizeof(read_dir), "%s/%ld", batch_dir, i);

    char filename[200];
    FILE *fp;
    size_t read_count;
    int32_t read_count2;

    // db->n_event_align_pairs[i]
    snprintf(filename, sizeof(filename), "%s/%s", read_dir, "n_event_align_pairs[i].dat");
    fp = fopen(filename, "r");
    if (fp == NULL)
        printf("Can not open %s\n", filename);
    read_count = fread(&(db->n_event_align_pairs[i]), sizeof(int32_t), 1, fp);
    assert(read_count == 1);
    fclose(fp);

    // db->event_align_pairs - out_2
    snprintf(filename, sizeof(filename), "%s/%s", read_dir, "event_align_pairs.dat");
    fp = fopen(filename, "r");
    if (fp == NULL)
        printf("Can not open %s\n", filename);
    int32_t pairs = db->n_event_align_pairs[i];
    db->event_align_pairs[i] = (AlignedPair *)malloc(sizeof(AlignedPair) * pairs);
    read_count2 = fread(db->event_align_pairs[i], sizeof(AlignedPair), pairs, fp);
    assert(read_count2 == pairs);
    fclose(fp);
}

int check_event_align_pairs(AlignedPair *pair_1, AlignedPair *pair_2, int32_t size)
{
    int i;
    int passed = size;
    for (i = 0; i < size; i++)
    {

        if (pair_1[i].read_pos != pair_2[i].read_pos || pair_1[i].ref_pos != pair_2[i].ref_pos)
        {
            // fprintf(stderr, "read_pos:%d (%d), ref_pos:%d (%d)\tFailed\n", pair_1[i].read_pos, pair_2[i].read_pos, pair_1[i].ref_pos, pair_2[i].ref_pos);
            passed--;
            // return 0;
        }
        // fprintf(stderr, "read_pos:%d (%d), ref_pos:%d (%d)\tPasses\n", pair_1[i].read_pos, pair_2[i].read_pos, pair_1[i].ref_pos, pair_2[i].ref_pos);
    }

    if (size == 0)
    {
        fprintf(stderr, "INF%");
    }
    else
    {
        fprintf(stderr, "%d%", passed * 100 / size);
    }

    // return 1;
}