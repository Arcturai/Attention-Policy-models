This repository contains all of the onnx backend-compatible weights files for Attention Policy v4 and v5 that I have trained up to this point (`apv4_t16` is AKA. `apv5_t00`, which is the only v5 net so far). It also contains weights for the three 128x16b baseline nets, and lc0 network 744204 as an equivalent onnx conversion.

`apv4_t13` is missing because I accidentally wiped its SWA weights from the final training checkpoint-- a shame because it had the highest training accuracy out of out all 16 ap nets (although within error). It might be recoverable but I haven't bothered to try because it probably isn't worth it. `apv4_t07` uses the SWA weights from its 90k training chkpt, because it underwent a training spike at 100k and nosedived in all metrics. baselines 03 and 04 are relatively untested, because they recently finished training as of this release (Oct 21). `baseline01`, the first baseline, is not included because it was trained with a sub-optimal configuration and egregiously overfit the training data, so it is much weaker than the others.

`policy_attn_arch.txt` contains a brief but incomplete overview of the neural architecture of the policy head in attention policy v5

`lc0_apv4_docs.txt` groups each attention policy net into categories based on some high-level features of their architecture

The training configs for each network are also included to provide more details on the architecture of each net. All three baselines were trained using `128x16-se4_baseline02.yaml`. 

With exception of the baselines and apv4_t16, none of these nets can be retrained from their yaml configurations because the code in my training repository has gone through constant development and is not backwards-compatible through certain changes to network architecture. To retrain these nets, it would be necessary to roll back the repo to a previous version. However, there is no documentation for which nets were trained on which commit, and I can't even guarantee that certain nets were not trained with changes made in-between commits. However, retraining is probably not necessary because most of these nets are very similar with only one or two differences between them, and they all performed very similarly in training. Additional nets can be trained with the current v5 format, with either dynamic or static pawn promotion policy.

.

All nets were converted to be compatible with the onnx backend using the following workflow:

1. load the last checkpoint from training (or create one with `net_to_model.py` when training checkpoints are absent, like with net 744204 provided)
2. load the SWA weights using code from `tfprocess.py`
3. save the network in TensorFlow 'saved model' format using model.save(path)
4. load the tf model configuration using model.get_config()
5. with a python script, directly modify the model config to resolve net arch inconsistencies between lc0 backend and training code, specifically:
	* change input layer shape to [112x8x8] from [112x64]
	* remove reshape layer after input and replace it with a TFOpLambda layer to scale the rule 50 plane by 1/99, because the onnx backend expects this
	* resolve the necessary 'inbound_nodes' fields after insertion of new layer
	* add softmax activation to the value head output
6. create a new model from the modified config using `model.__class__.from_config(model_config, custom_objects)`, passing it the definitions of the custom object layers from tfprocess: ApplySqueezeExcitation and ApplyPolicyMap / ApplyAttentionPolicyMap 
7. transfer over the layer weights to the new model and save it
8. convert the saved model to onnx using tf2onnx (https://github.com/onnx/tensorflow-onnx/releases)
9. using another python script and code from `net.py`, load the onnx model, resolve inconsistencies in the naming of input and output layers, and save the weights as a proto file

I can provide the necessary scripts to anyone interested in doing it themselves.




