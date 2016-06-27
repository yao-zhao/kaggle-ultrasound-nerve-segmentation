# 
for i in `seq 1 1`; do

# train model on unsupervised data, with some random labeling
echo "start  training-------------------------------------------------------------------------"
~/caffe-yao/build/tools/caffe train -gpu 1 --solver=/home/yz/uns/models/segnet/solver.prototxt \
     	2>&1 | tee /home/yz/uns/models/segnet/log.txt

# --snapshot=data/models/segnet_iter_2000.solverstate 2>&1 | tee -a /home/yz/uns/models/segnet/log.txt

# ~/caffe-yao/build/tools/caffe train --solver=/home/yz/uns/models/segnet/solver.prototxt \
 echo " training done "



done
