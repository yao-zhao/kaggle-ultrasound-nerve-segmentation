# 
for i in `seq 1 1`; do

# train model on unsupervised data, with some random labeling
echo "start  training-------------------------------------------------------------------------"
~/caffe-yao/build/tools/caffe train -gpu 1 --solver=/home/yz/uns/models/bpnet/solver.prototxt \
     	2>&1 | tee /home/yz/uns/models/bpnet/log.txt
 # --snapshot=data/models/bpnet_iter_2000.solverstate 2>&1 | tee -a log.txt

 echo " training done "



done
