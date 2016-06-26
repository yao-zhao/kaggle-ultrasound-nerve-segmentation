# path parameters
DATA=~/uns/data
RAWDATA=~/uns/data/raw
TOOLS=~/caffe-yao/build/tools
RESIZE_HEIGHT=210
RESIZE_WIDTH=280

# create new lmdb database
rm -rf $DATA/val_lmdb
GLOG_logtostderr=1 $TOOLS/convert_imageset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    --shuffle \
    $RAWDATA/train/ \
    $DATA/val_BP_label.txt \
    $DATA/val_lmdb
echo "val lmdb created"

# create new lmdb database
rm -rf $DATA/train_lmdb 
GLOG_logtostderr=1 $TOOLS/convert_imageset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    --shuffle \
    $RAWDATA/train/ \
    $DATA/train_BP_label.txt \
    $DATA/train_lmdb
echo "train lmdb created"


# create new lmdb database
rm -rf $DATA/test_lmdb 
GLOG_logtostderr=1 $TOOLS/convert_imageset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    $RAWDATA/test/ \
    $DATA/test_BP_label.txt \
    $DATA/test_lmdb
echo "test lmdb created"
# echo "calculate training and validation image mean"
# sh make_mean.sh
