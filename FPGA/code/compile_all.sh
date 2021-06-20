aoc -report -board=de5net_a7 device/align_pre.cl -o bins/bin_separate/align_pre.aocx

aoc -report -board=de5net_a7 device/align_core.cl -o bins/bin_separate/align_core.aocx

aoc -report -board=de5net_a7 device/align_core_2.cl -o bins/bin_separate/align_core_2cus.aocx

aoc -report -board=de5net_a7 device/align_post.cl -o bins/bin_separate/align_post.aocx

