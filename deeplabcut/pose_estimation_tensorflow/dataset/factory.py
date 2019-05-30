'''
Adopted from DeeperCut by Eldar Insafutdinov
https://github.com/eldar/pose-tensorflow
'''

from deeplabcut.pose_estimation_tensorflow.dataset.pose_dataset import SingleViewPoseDataset, MultiViewPoseDataset

def create(cfg):
	if cfg.num_views ==1 :
		data = SingleViewPoseDataset(cfg)
	else:
		data = MultiViewPoseDataset(cfg)
	return data
