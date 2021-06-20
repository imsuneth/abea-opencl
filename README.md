# Accelerating Adaptive Banded Event Alignment Algorithm on FPGAs using OpenCL

## Aim of the proposed project:

- Effective utilization of OpenCL to map the Adaptive Banded Event Alignment(ABEA) algorithm to run efficiently on an FPGA.
- Evaluate the performance improvement of the ABEA with the FPGA implementation.

## Background

The process of DNA sequencing is a precise determination of the amount and distribution of nucleotides (adenine (A), guanine (G), cytosine (C), and thymine (T)) in DNA molecules. It has a very strong impact in various biological fields such as human genetics, agriculture, bioinformatics, etc.

DNA sequencing machines produce gene sequences much faster than the traditional molecular biology techniques and also these DNA sequencing data is much larger in size (terabytes of data, read lengths of 1000 to >1M bases). Analyzing these data still depends on high-performance or cloud computers.

Therefore, accelerating DNA sequencing methods by heterogeneous architectures ( i.e.FPGAs) and the capability of detecting entire genomes in short periods of time could revolutionize the world of medicine and technology.

Therefore, to overcome this immense computational load, reconfigurable computing, also known as FPGA is the field in which algorithms are mapped directly to configurable hardware resources with parallelism. The cost per computation and watts per computation is also quite favorable hence it is worth running the bioinformatics algorithms on FPGAs.

In the field of Nanopore sequencing, the ABEA algorithm can be used to align the raw signal (a time series of electric current that can be generated using the latest generation (third generation) of sequencing technologies) to a biological reference sequence.

ABEA is one of the most time-consuming steps when analyzing raw nanopore data. As of now it has been parallelized, optimized, and fine-tuned to exploit architectural features in general-purpose GPUs and CPUs, and in this project, we propose to take these techniques to FPGAs.

A custom hardware design of the ABEA algorithm done with hardware-software co-design principles has the potential to achieve superior performance.OpenCL can be used for writing programs at high-level languages on FPGA which are then converted by the underlying layers to run with the support of board support package (BSP) in order to accelerate the ABEA Algorithm on it.

## Usage

#### Create the Dataset

Clone repository https://github.com/imsuneth/f5c

```
git clone https://github.com/imsuneth/f5c
cd f5c
```

Run f5c according to the documentation

example: run f5c on ecoli_2kb_region dataset with maximum batch size of 1.3Mbases

```
./scripts/test.sh -c -B 1.3M
```

then copy dumped_test folder in to abea-on-fpga/FPGA/new/bins/

#### Compile kernels

```
aoc -report -board=<BOARD NAME> device/<KERNEL.cl> -o bins/bin/<KERNEL.aocx>
```
add -profile to enable profiling : may increase the execution time.

#### Example on de5-net

```
aoc -report -board=de5net_a7 device/align.cl -o bins/bin/align.aocx -profile
```

#### run the host program linking the dataset

```

make BIN=bins/bin CPP=host/<HOST.cpp>
./bins/bin/host /path_to_<dumped_dataset>
```

#### Example

```
make BIN=bins/bin CPP=host/align_3k.cpp
./bins/bin/host ../dumped_dataset
```

### Reports & Profiling

1. The kernel report will be created under bins/bin/align/report/

2. To open The Source Code tab in the Intel FPGA Dynamic Profiler for OpenCL GUI (should have compile kernel with -profile flag to use this)

```
aocl report bins/bin/align.aocx bins/bin/profile.mon device/align.cl
```
