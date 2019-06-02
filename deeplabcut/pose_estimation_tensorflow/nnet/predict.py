'''
Adapted from original predict.py by Eldar Insafutdinov's implementation of [DeeperCut](https://github.com/eldar/pose-tensorflow)
To do faster inference on videos. See https://www.biorxiv.org/content/early/2018/10/30/457242

Source: DeeperCut by Eldar Insafutdinov
https://github.com/eldar/pose-tensorflow
'''

import numpy as np
import tensorflow as tf
from deeplabcut.pose_estimation_tensorflow.nnet.net_factory import pose_net

def setup_pose_prediction(cfg):
    tf.reset_default_graph()
    inputs = tf.placeholder(tf.float32, shape=[cfg.batch_size * cfg.num_views, None, None, 3])
    net_heads = pose_net(cfg).test(inputs)
    outputs = [net_heads['part_prob']]
    if cfg.location_refinement:
        outputs.append(net_heads['locref'])

    restorer = tf.train.Saver()
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    # Restore variables from disk.
    restorer.restore(sess, cfg.init_weights)

    return sess, inputs, outputs
    
def extract_cnn_output(outputs_np, cfg):
    ''' extract locref + scmap from network '''
    scmap = outputs_np[0]
    locref = None
    if cfg.location_refinement:
        locref = outputs_np[1]
        shape = locref.shape
        locref = np.reshape(locref, (cfg.num_views, shape[1], shape[2], -1, 2))
        locref *= cfg.locref_stdev
    return scmap, locref

def argmax_pose_predict(scmaps, offmats, stride):
    """Combine scoremat and offsets to the final pose."""
    N, H, W, C = scmaps.shape
    maxlocs = np.unravel_index(np.argmax(scmaps.reshape([N, H*W, C]), axis=1), [H,W])
    nn, cc = np.meshgrid(np.arange(N), np.arange(C), indexing='ij')
    offsets = offmats[nn, maxlocs[0], maxlocs[1], cc, ::-1] # i,j
    pos_f8s = np.stack(maxlocs, axis=2).astype(np.float)*stride + 0.5*stride + offsets
    return np.concatenate((pos_f8s[:,:,::-1], scmaps[nn, maxlocs[0], maxlocs[1], cc][:,:,None]), axis=2) # N x C x (x,y,confidence)

def getpose(image, cfg, sess, inputs, outputs, outall=False):
    ''' Extract pose '''
    im=np.expand_dims(image, axis=0).astype(float)
    outputs_np = sess.run(outputs, feed_dict={inputs: im})
    scmap, locref = extract_cnn_output(outputs_np, cfg)
    pose = argmax_pose_predict(scmap, locref, cfg.stride)
    if outall:
        return scmap, locref, pose
    else:
        return pose

## Functions below implement are for batch sizes > 1:
def extract_cnn_outputmulti(outputs_np, cfg):
    ''' extract locref + scmap from network 
    Dimensions: image batch x imagedim1 x imagedim2 x bodypart'''
    scmap = outputs_np[0]
    locref = None
    if cfg.location_refinement:
        locref =outputs_np[1]
        shape = locref.shape
        locref = np.reshape(locref, (shape[0], shape[1],shape[2], -1, 2))
        locref *= cfg.locref_stdev
    if len(scmap.shape)==2: #for single body part!
        scmap=np.expand_dims(scmap,axis=2)
    return scmap, locref


def getposeNP(image, cfg, sess, inputs, outputs, outall=False):
    ''' Adapted from DeeperCut, performs numpy-based faster inference on batches'''
    outputs_np = sess.run(outputs, feed_dict={inputs: image})
    
    scmap, locref = extract_cnn_outputmulti(outputs_np, cfg) #processes image batch.
    batchsize,ny,nx,num_joints = scmap.shape
    
    #Combine scoremat and offsets to the final pose.
    LOCREF=locref.reshape(batchsize,nx*ny,num_joints,2)
    MAXLOC=np.argmax(scmap.reshape(batchsize,nx*ny,num_joints),axis=1)
    Y,X=np.unravel_index(MAXLOC,dims=(ny,nx))
    DZ=np.zeros((batchsize,num_joints,3))
    for l in range(batchsize):
        for k in range(num_joints):
            DZ[l,k,:2]=LOCREF[l,MAXLOC[l,k],k,:]
            DZ[l,k,2]=scmap[l,Y[l,k],X[l,k],k]
            
    X=X.astype('float32')*cfg.stride+.5*cfg.stride+DZ[:,:,0]
    Y=Y.astype('float32')*cfg.stride+.5*cfg.stride+DZ[:,:,1]
    pose = np.empty((cfg['batch_size'], cfg['num_joints']*3), dtype=X.dtype) 
    pose[:,0::3] = X
    pose[:,1::3] = Y
    pose[:,2::3] = DZ[:,:,2] #P
    if outall:
        return scmap, locref, pose
    else:
        return pose

