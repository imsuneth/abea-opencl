aoc -report -board=de5net_a7 device/align_single_fpga.cl -o bins/bin_single_fpga_1/align.aocx

aoc -report -board=de5net_a7 device/align_single_fpga.cl -o bins/bin_single_fpga_1_prof/align.aocx -profile

aoc -report -board=de5net_a7 device/align_single_fpga2.cl -o bins/bin_single_fpga_2/align.aocx

aoc -report -board=de5net_a7 device/align_single_fpga2.cl -o bins/bin_single_fpga_2_prof/align.aocx -prof

make BIN=bins/bin_single_fpga_1 CPP=host/src/align_single_fpga.cpp
make BIN=bins/bin_single_fpga_1_prof CPP=host/src/align_single_fpga.cpp
make BIN=bins/bin_single_fpga_2 CPP=host/src/align_single_fpga.cpp
make BIN=bins/bin_single_fpga_2_prof CPP=host/src/align_single_fpga.cpp

./bins/bin_single_fpga_1/host ../dump_large_1.4M
./bins/bin_single_fpga_1_prof/host ../dump_large_1.4M
./bins/bin_single_fpga_2/host ../dump_large_1.4M
./bins/bin_single_fpga_2_prof/host ../dump_large_1.4M
