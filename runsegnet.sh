# 
for i in `seq 1 1`; do

# train model on unsupervised data, with some random labeling
echo "start  training-------------------------------------------------------------------------"
~/caffe-yao/build/tools/caffe train --solver=/home/yz/uns/models/segnet/solver.prototxt 2>&1 | tee log.txt
# ~/caffe-yao/build/tools/caffe train --solver=/home/yz/kaggle_sf/models/resnet/solver.prototxt \
#    --snapshot=data/models/resnet_iter_1500.solverstate 2>&1 | tee log.txt
 echo " training done "



done
