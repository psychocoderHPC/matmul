#!/bin/bash
TILESIZE="32"
NUMTHREADS="256"
cd `dirname $0`
src_dir=`pwd`
cd -

rm -rf paper_scaling_k20_float
mkdir paper_scaling_k20_float
cd paper_scaling_k20_float
$src_dir/make_paper_autotuning_k20_float.sh $NUMTHREADS $TILESIZE
./matmul_benchmark 4096 4096 1024 5 | tee log.txt
cd ..
