for i in `seq 1 1`; do

# # visualize result
# python visualize_result.py --MODEL_DEF 'models/segnet/deploy_val.prototxt'

# # predict BP detection
# python predict_bpnet.py --MODEL_WEIGHT 'data/models/bpnet_iter_1500.caffemodel' \
# 	--MODEL_DEF 'models/bpnet/deploy.prototxt' \
# 	--RESULT_NAME $i

# # predict segmentation
# python predict_segnet.py --MODEL_WEIGHT 'data/models/segnet_iter_2500.caffemodel' \
# 	--MODEL_DEF 'models/segnet/deploy.prototxt' \
# 	--RESULT_NAME $i

# # predict all
python predict_final.py --BP_PREDICTION 'results/result_'$i'.csv' \
--MASK_FOLDER 'data/result_mask/result_'$i'/' \
--RESULT_NAME $i

done