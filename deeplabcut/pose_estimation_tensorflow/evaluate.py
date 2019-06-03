"""
DeepLabCut2.0 Toolbox
https://github.com/AlexEMG/DeepLabCut
A Mathis, alexander.mathis@bethgelab.org
T Nath, nath@rowland.harvard.edu
M Mathis, mackenzie@post.harvard.edu

"""

import os
import argparse

# Dependencies for anaysis
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path

def pairwisedistances(DataCombined,scorer1,scorer2,pcutoff=-1,bodyparts=None):
    ''' Calculates the pairwise Euclidean distance metric over body parts vs. images'''
    mask=DataCombined[scorer2].xs('likelihood',level=1,axis=1)>=pcutoff
    if bodyparts==None:
            Pointwisesquareddistance=((DataCombined[scorer1]-DataCombined[scorer2])**2).astype(np.float)
            RMSE=np.sqrt(Pointwisesquareddistance.xs('x',level=1,axis=1)+Pointwisesquareddistance.xs('y',level=1,axis=1)) #Euclidean distance (proportional to RMSE)
            return RMSE,RMSE[mask]
    else:
            Pointwisesquareddistance=((DataCombined[scorer1][bodyparts]-DataCombined[scorer2][bodyparts])**2).astype(np.float)
            RMSE=np.sqrt(Pointwisesquareddistance.xs('x',level=1,axis=1)+Pointwisesquareddistance.xs('y',level=1,axis=1)) #Euclidean distance (proportional to RMSE)
            return RMSE,RMSE[mask]

def evaluate_network(config,Shuffles=[1],plotting = None,show_errors = True,comparisonbodyparts="all",gputouse=None):
    """
    Evaluates the network based on the saved models at different stages of the training network.\n
    The evaluation results are stored in the .h5 and .csv file under the subdirectory 'evaluation_results'.
    Change the snapshotindex parameter in the config file to 'all' in order to evaluate all the saved models.

    Parameters
    ----------
    config : string
        Full path of the config.yaml file as a string.

    Shuffles: list, optional
        List of integers specifying the shuffle indices of the training dataset. The default is [1]

    plotting: bool, optional
        Plots the predictions on the train and test images. The default is ``False``; if provided it must be either ``True`` or ``False``

    show_errors: bool, optional
        Display train and test errors. The default is `True``

    comparisonbodyparts: list of bodyparts, Default is "all".
        The average error will be computed for those body parts only (Has to be a subset of the body parts).

    gputouse: int, optional. Natural number indicating the number of your GPU (see number in nvidia-smi). If you do not have a GPU put None.
    See: https://nvidia.custhelp.com/app/answers/detail/a_id/3751/~/useful-nvidia-smi-queries
    
    Examples
    --------
    If you do not want to plot
    >>> deeplabcut.evaluate_network('/analysis/project/reaching-task/config.yaml', shuffle=[1])
    --------

    If you want to plot
    >>> deeplabcut.evaluate_network('/analysis/project/reaching-task/config.yaml',shuffle=[1],True)
    """
    import os
    from skimage import io
    import skimage.color

    from deeplabcut.pose_estimation_tensorflow.nnet import predict as ptf_predict
    from deeplabcut.pose_estimation_tensorflow.config import load_config
    from deeplabcut.pose_estimation_tensorflow.dataset.pose_dataset import data_to_input
    from deeplabcut.utils import auxiliaryfunctions, visualization
    import tensorflow as tf
    
    if 'TF_CUDNN_USE_AUTOTUNE' in os.environ:
        del os.environ['TF_CUDNN_USE_AUTOTUNE'] #was potentially set during training
    

    tf.reset_default_graph()
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # 
#    tf.logging.set_verbosity(tf.logging.WARN)

    start_path=os.getcwd()
    # Read file path for pose_config file. >> pass it on
    cfg = auxiliaryfunctions.read_config(config)
    if gputouse is not None: #gpu selectinon
            os.environ['CUDA_VISIBLE_DEVICES'] = str(gputouse)
            
    # Loading human annotatated data
    trainingsetfolder=auxiliaryfunctions.GetTrainingSetFolder(cfg)
    Data=pd.read_hdf(os.path.join(cfg["project_path"],str(trainingsetfolder),'CollectedData_' + cfg["scorer"] + '.h5'),'df_with_missing')
    # Get list of body parts to evaluate network for
    comparisonbodyparts=auxiliaryfunctions.IntersectionofBodyPartsandOnesGivenbyUser(cfg,comparisonbodyparts)
    # Make folder for evaluation
    auxiliaryfunctions.attempttomakefolder(str(cfg["project_path"]+"/evaluation-results/"))
    for shuffle in Shuffles:
        for trainFraction in cfg["TrainingFraction"]:
            ##################################################
            # Load and setup CNN part detector
            ##################################################
            datafn,metadatafn=auxiliaryfunctions.GetDataandMetaDataFilenames(trainingsetfolder,trainFraction,shuffle,cfg)
            modelfolder=os.path.join(cfg["project_path"],str(auxiliaryfunctions.GetModelFolder(trainFraction,shuffle,cfg)))
            path_test_config = Path(modelfolder) / 'test' / 'pose_cfg.yaml'
            # Load meta data
            data, trainIndices, testIndices, trainFraction=auxiliaryfunctions.LoadMetadata(os.path.join(cfg["project_path"],metadatafn))

            try:
                dlc_cfg = load_config(str(path_test_config))
            except FileNotFoundError:
                raise FileNotFoundError("It seems the model for shuffle %s and trainFraction %s does not exist."%(shuffle,trainFraction))
            
            #change batch size, if it was edited during analysis!
            dlc_cfg['batch_size']=1 #in case this was edited for analysis.
            #Create folder structure to store results.
            evaluationfolder=os.path.join(cfg["project_path"],str(auxiliaryfunctions.GetEvaluationFolder(trainFraction,shuffle,cfg)))
            auxiliaryfunctions.attempttomakefolder(evaluationfolder,recursive=True)
            #path_train_config = modelfolder / 'train' / 'pose_cfg.yaml'
            
            # Check which snapshots are available and sort them by # iterations
            Snapshots = np.array([fn.split('.')[0]for fn in os.listdir(os.path.join(str(modelfolder), 'train'))if "index" in fn])
            try: #check if any where found?
              Snapshots[0]
            except IndexError:
              raise FileNotFoundError("Snapshots not found! It seems the dataset for shuffle %s and trainFraction %s is not trained.\nPlease train it before evaluating.\nUse the function 'train_network' to do so."%(shuffle,trainFraction))

            increasing_indices = np.argsort([int(m.split('-')[1]) for m in Snapshots])
            Snapshots = Snapshots[increasing_indices]

            if cfg["snapshotindex"] == -1:
                snapindices = [-1]
            elif cfg["snapshotindex"] == "all":
                snapindices = range(len(Snapshots))
            elif cfg["snapshotindex"]<len(Snapshots):
                snapindices=[cfg["snapshotindex"]]
            else:
                print("Invalid choice, only -1 (last), any integer up to last, or all (as string)!")

            final_result=[]
            ##################################################
            # Compute predictions over images
            ##################################################
            for snapindex in snapindices:
                dlc_cfg['init_weights'] = os.path.join(str(modelfolder),'train',Snapshots[snapindex]) #setting weights to corresponding snapshot.
                trainingsiterations = (dlc_cfg['init_weights'].split(os.sep)[-1]).split('-')[-1] #read how many training siterations that corresponds to.
                
                #name for deeplabcut net (based on its parameters)
                DLCscorer = auxiliaryfunctions.GetScorerName(cfg,shuffle,trainFraction,trainingsiterations)
                print("Running ", DLCscorer, " with # of trainingiterations:", trainingsiterations)
                resultsfilename=os.path.join(str(evaluationfolder),DLCscorer + '-' + Snapshots[snapindex]+  '.h5')
                try:
                    DataMachine = pd.read_hdf(resultsfilename,'df_with_missing')
                    print("This net has already been evaluated!")
                except FileNotFoundError:
                    # Specifying state of model (snapshot / training state)
                    sess, inputs, outputs = ptf_predict.setup_pose_prediction(dlc_cfg)

                    Numimages = len(Data.index)
                    PredicteData = np.zeros((Numimages,3 * len(dlc_cfg['all_joints_names'])))
                    print("Analyzing data...")
                    for imageindex, imagename in tqdm(enumerate(Data.index)):
                        image = io.imread(os.path.join(cfg['project_path'],imagename),mode='RGB')
                        image = skimage.color.gray2rgb(image)
                        image_batch = data_to_input(image)
                        
                        # Compute prediction with the CNN
                        outputs_np = sess.run(outputs, feed_dict={inputs: image_batch})
                        scmap, locref = ptf_predict.extract_cnn_output(outputs_np, dlc_cfg)

                        # Extract maximum scoring location from the heatmap, assume 1 person
                        pose = ptf_predict.argmax_pose_predict(scmap, locref, dlc_cfg.stride)
                        PredicteData[imageindex, :] = pose.flatten()  # NOTE: thereby     cfg_test['all_joints_names'] should be same order as bodyparts!

                    sess.close() #closes the current tf session

                    index = pd.MultiIndex.from_product(
                        [[DLCscorer], dlc_cfg['all_joints_names'], ['x', 'y', 'likelihood']],
                        names=['scorer', 'bodyparts', 'coords'])

                    # Saving results
                    DataMachine = pd.DataFrame(PredicteData, columns=index, index=Data.index.values)
                    DataMachine.to_hdf(resultsfilename,'df_with_missing',format='table',mode='w')

                    print("Done and results stored for snapshot: ", Snapshots[snapindex])
                    DataCombined = pd.concat([Data.T, DataMachine.T], axis=0).T
                    RMSE,RMSEpcutoff = pairwisedistances(DataCombined, cfg["scorer"], DLCscorer,cfg["pcutoff"],comparisonbodyparts)
                    testerror = np.nanmean(RMSE.iloc[testIndices].values.flatten())
                    trainerror = np.nanmean(RMSE.iloc[trainIndices].values.flatten())
                    testerrorpcutoff = np.nanmean(RMSEpcutoff.iloc[testIndices].values.flatten())
                    trainerrorpcutoff = np.nanmean(RMSEpcutoff.iloc[trainIndices].values.flatten())
                    results = [trainingsiterations,int(100 * trainFraction),shuffle,np.round(trainerror,2),np.round(testerror,2),cfg["pcutoff"],np.round(trainerrorpcutoff,2), np.round(testerrorpcutoff,2)]
                    final_result.append(results)

                    if show_errors == True:
                            print("Results for",trainingsiterations," training iterations:", int(100 * trainFraction), shuffle, "train error:",np.round(trainerror,2), "pixels. Test error:", np.round(testerror,2)," pixels.")
                            print("With pcutoff of", cfg["pcutoff"]," train error:",np.round(trainerrorpcutoff,2), "pixels. Test error:", np.round(testerrorpcutoff,2), "pixels")
                            print("Thereby, the errors are given by the average distances between the labels by DLC and the scorer.")


                    if plotting == True:
                        print("Plotting...")
                        colors = visualization.get_cmap(len(comparisonbodyparts),name=cfg['colormap'])

                        foldername=os.path.join(str(evaluationfolder),'LabeledImages_' + DLCscorer + '_' + Snapshots[snapindex])
                        auxiliaryfunctions.attempttomakefolder(foldername)
                        NumFrames=np.size(DataCombined.index)
                        for ind in np.arange(NumFrames):
                            visualization.PlottingandSaveLabeledFrame(DataCombined,ind,trainIndices,cfg,colors,comparisonbodyparts,DLCscorer,foldername)
                            
                    tf.reset_default_graph()
                    #print(final_result)
            make_results_file(final_result,evaluationfolder,DLCscorer)
            print("The network is evaluated and the results are stored in the subdirectory 'evaluation_results'.")
            print("If it generalizes well, choose the best model for prediction and update the config file with the appropriate index for the 'snapshotindex'.\nUse the function 'analyze_video' to make predictions on new videos.")
            print("Otherwise consider retraining the network (see DeepLabCut workflow Fig 2)")
    
    #returning to intial folder
    os.chdir(str(start_path))

def evaluate_multiview_network(config,videos,projection_matrices,multiview_step,snapshot_index=None,Shuffles=[1],plotting = None,show_errors = True,comparisonbodyparts="all",gputouse=None):
    """
    Evaluates the network based on the saved models at different stages of the training network.\n
    The evaluation results are stored in the .h5 and .csv file under the subdirectory 'evaluation_results'.
    Change the snapshotindex parameter in the config file to 'all' in order to evaluate all the saved models.

    Parameters
    ----------
    config : string
        Full path of the config.yaml file as a string.

    videos: list of strings
        Name of each video, one per viewpoint. Must be in the same order that it was in for training

    projection_matrices: list of arrays
        Projection matrix for each viewpoint. Each is a 3x4 array

    multiview_step:
        1 or 2. Indicates whether network was trained with train_multiview_network_step_1 or train_multiview_network_step_2

    Shuffles: list, optional
        List of integers specifying the shuffle indices of the training dataset. The default is [1]

    plotting: bool, optional
        Plots the predictions on the train and test images. The default is ``False``; if provided it must be either ``True`` or ``False``

    show_errors: bool, optional
        Display train and test errors. The default is `True``

    comparisonbodyparts: list of bodyparts, Default is "all".
        The average error will be computed for those body parts only (Has to be a subset of the body parts).

    gputouse: int, optional. Natural number indicating the number of your GPU (see number in nvidia-smi). If you do not have a GPU put None.
    See: https://nvidia.custhelp.com/app/answers/detail/a_id/3751/~/useful-nvidia-smi-queries
    
    Examples
    --------
    If you do not want to plot
    >>> deeplabcut.evaluate_network('/analysis/project/reaching-task/config.yaml', shuffle=[1])
    --------

    If you want to plot
    >>> deeplabcut.evaluate_network('/analysis/project/reaching-task/config.yaml',shuffle=[1],True)
    """
    import os
    from skimage import io
    import skimage.color

    from deeplabcut.pose_estimation_tensorflow.nnet import predict as ptf_predict
    from deeplabcut.pose_estimation_tensorflow.config import load_config
    from deeplabcut.pose_estimation_tensorflow.dataset.pose_dataset import data_to_input
    from deeplabcut.utils import auxiliaryfunctions, visualization
    import tensorflow as tf
    
    if 'TF_CUDNN_USE_AUTOTUNE' in os.environ:
        del os.environ['TF_CUDNN_USE_AUTOTUNE'] #was potentially set during training
    

    tf.reset_default_graph()
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # 
#    tf.logging.set_verbosity(tf.logging.WARN)

    start_path=os.getcwd()
    # Read file path for pose_config file. >> pass it on
    cfg = auxiliaryfunctions.read_config(config)
    if gputouse is not None: #gpu selectinon
            os.environ['CUDA_VISIBLE_DEVICES'] = str(gputouse)
            
    # Loading human annotatated data
    trainingsetfolder=auxiliaryfunctions.GetTrainingSetFolder(cfg)
    Datas = [pd.read_hdf(os.path.join(cfg['project_path'], 'labeled-data', video, 'CollectedData_'+cfg['scorer']+'.h5'), 'df_with_missing') for video in videos]
    # Get list of body parts to evaluate network for
    comparisonbodyparts=auxiliaryfunctions.IntersectionofBodyPartsandOnesGivenbyUser(cfg,comparisonbodyparts)
    # Make folder for evaluation
    auxiliaryfunctions.attempttomakefolder(str(cfg["project_path"]+"/evaluation-results/"))
    for shuffle in Shuffles:
        for trainFraction in cfg["TrainingFraction"]:
            ##################################################
            # Load and setup CNN part detector
            ##################################################
            datafn,metadatafn=auxiliaryfunctions.GetDataandMetaDataFilenames(trainingsetfolder,trainFraction,shuffle,cfg)
            modelfolder=os.path.join(cfg["project_path"],str(auxiliaryfunctions.GetModelFolder(trainFraction,shuffle,cfg)))
            path_test_config = Path(modelfolder) / 'test' / 'pose_cfg.yaml'
            # Load meta data
            metadatas = []
            for video in videos:
                m = ('-'+video).join(os.path.splitext(metadatafn))
                data, trainIndices, testIndices, trainFraction=auxiliaryfunctions.LoadMetadata(os.path.join(cfg["project_path"],m))
                metadatas.append(data)

            try:
                dlc_cfg = load_config(str(path_test_config))
            except FileNotFoundError:
                raise FileNotFoundError("It seems the model for shuffle %s and trainFraction %s does not exist."%(shuffle,trainFraction))
            
            #change batch size, if it was edited during analysis!
            dlc_cfg['batch_size']=1 #in case this was edited for analysis.
            #Create folder structure to store results.
            evaluationfolder=os.path.join(cfg["project_path"],str(auxiliaryfunctions.GetEvaluationFolder(trainFraction,shuffle,cfg)))
            auxiliaryfunctions.attempttomakefolder(evaluationfolder,recursive=True)
            #path_train_config = modelfolder / 'train' / 'pose_cfg.yaml'

            dlc_cfg.multiview_step = multiview_step
            dlc_cfg.projection_matrices = projection_matrices
            
            # Check which snapshots are available and sort them by # iterations
            Snapshots = np.array([fn.split('.')[0]for fn in os.listdir(os.path.join(str(modelfolder), 'train'))if "index" in fn])
            try: #check if any where found?
              Snapshots[0]
            except IndexError:
              raise FileNotFoundError("Snapshots not found! It seems the dataset for shuffle %s and trainFraction %s is not trained.\nPlease train it before evaluating.\nUse the function 'train_network' to do so."%(shuffle,trainFraction))

            increasing_indices = np.argsort([int(m.split('-')[1]) for m in Snapshots])
            Snapshots = Snapshots[increasing_indices]

            if snapshot_index is not None:
                snapindices = [i for i in range(len(Snapshots)) if int(Snapshots[i].split('-')[1].split('.')[0])==snapshot_index]
            elif cfg["snapshotindex"] == -1:
                snapindices = [-1]
            elif cfg["snapshotindex"] == "all":
                snapindices = range(len(Snapshots))
            elif cfg["snapshotindex"]<len(Snapshots):
                snapindices=[cfg["snapshotindex"]]
            else:
                print("Invalid choice, only -1 (last), any integer up to last, or all (as string)!")

            final_result=[]
            ##################################################
            # Compute predictions over images
            ##################################################
            for snapindex in snapindices:
                dlc_cfg['init_weights'] = os.path.join(str(modelfolder),'train',Snapshots[snapindex]) #setting weights to corresponding snapshot.
                trainingsiterations = (dlc_cfg['init_weights'].split(os.sep)[-1]).split('-')[-1] #read how many training siterations that corresponds to.
                
                #name for deeplabcut net (based on its parameters)
                DLCscorer = auxiliaryfunctions.GetScorerName(cfg,shuffle,trainFraction,trainingsiterations)
                print("Running ", DLCscorer, " with # of trainingiterations:", trainingsiterations)
                resultsfilename=os.path.join(str(evaluationfolder),DLCscorer + '-' + Snapshots[snapindex]+  '.h5')
                try:
                    DataMachine = pd.read_hdf(resultsfilename,'df_with_missing')
                    print("This net has already been evaluated!")
                except FileNotFoundError:
                    # Specifying state of model (snapshot / training state)
                    sess, inputs, outputs = ptf_predict.setup_pose_prediction(dlc_cfg)

                    Numimages = len(Datas[0].index)
                    PredicteDatas = np.zeros((Numimages,len(Datas), 3 * len(dlc_cfg['all_joints_names'])))
                    imagesizes = []
                    print("Analyzing data...")
                    if multiview_step == 1:
                        for imageindex in tqdm(range(len(Datas[0].index))):
                            imagenames = [Data.index[imageindex] for Data in Datas]
                            images = [io.imread(os.path.join(cfg['project_path'],imagename),mode='RGB') for imagename in imagenames]
                            images = [skimage.color.gray2rgb(image) for image in images]
                            image_batch = images
                            imagesizes.append([image.shape for image in images])
                            
                            # Compute prediction with the CNN
                            outputs_np = sess.run(outputs, feed_dict={inputs: image_batch})
                            scmap, locref = ptf_predict.extract_cnn_output(outputs_np, dlc_cfg)

                            # Extract maximum scoring location from the heatmap, assume 1 person
                            pose = ptf_predict.argmax_pose_predict(scmap, locref, dlc_cfg.stride)
                            PredicteDatas[imageindex] = pose.reshape([pose.shape[0], -1])  # NOTE: thereby     cfg_test['all_joints_names'] should be same order as bodyparts!

                        sess.close() #closes the current tf session

                        index = pd.MultiIndex.from_product(
                            [[DLCscorer], dlc_cfg['all_joints_names'], ['x', 'y', 'likelihood']],
                            names=['scorer', 'bodyparts', 'coords'])

                        # Saving results
                        for i, video in enumerate(videos):
                            print('Evaluating 2D predictions on video %s'%video)
                            Data = Datas[i]
                            DataMachine = pd.DataFrame(PredicteDatas[:,i], columns=index, index=Data.index.values)
                            r = ('-'+video).join(os.path.splitext(resultsfilename))
                            DataMachine.to_hdf(r,'df_with_missing',format='table',mode='w')

                            print("Done and results stored for snapshot: ", Snapshots[snapindex])
                            DataCombined = pd.concat([Data.T, DataMachine.T], axis=0).T
                            RMSE,RMSEpcutoff = pairwisedistances(DataCombined, cfg["scorer"], DLCscorer,cfg["pcutoff"],comparisonbodyparts)
                            testerror = np.nanmean(RMSE.iloc[testIndices].values.flatten())
                            trainerror = np.nanmean(RMSE.iloc[trainIndices].values.flatten())
                            testerrorpcutoff = np.nanmean(RMSEpcutoff.iloc[testIndices].values.flatten())
                            trainerrorpcutoff = np.nanmean(RMSEpcutoff.iloc[trainIndices].values.flatten())
                            results = [trainingsiterations,int(100 * trainFraction),shuffle,np.round(trainerror,2),np.round(testerror,2),cfg["pcutoff"],np.round(trainerrorpcutoff,2), np.round(testerrorpcutoff,2)]
                            final_result.append(results)

                            if show_errors == True:
                                    print("Results for",trainingsiterations," training iterations:", int(100 * trainFraction), shuffle, "train error:",np.round(trainerror,2), "pixels. Test error:", np.round(testerror,2)," pixels.")
                                    print("With pcutoff of", cfg["pcutoff"]," train error:",np.round(trainerrorpcutoff,2), "pixels. Test error:", np.round(testerrorpcutoff,2), "pixels")
                                    print("Thereby, the errors are given by the average distances between the labels by DLC and the scorer.")

                            if plotting == True:
                                print("Plotting...")
                                colors = visualization.get_cmap(len(comparisonbodyparts),name=cfg['colormap'])

                                foldername=os.path.join(str(evaluationfolder),'LabeledImages_' + DLCscorer + '_' + Snapshots[snapindex]+'_'+video)
                                auxiliaryfunctions.attempttomakefolder(foldername)
                                NumFrames=np.size(DataCombined.index)
                                for ind in np.arange(NumFrames):
                                    visualization.PlottingandSaveLabeledFrame(DataCombined,ind,trainIndices,cfg,colors,comparisonbodyparts,DLCscorer,foldername)
                        
                        # get predictions in homogeneous pixel coordinates
                        # pixel coordinates have (0,0) in the top-left, and the bottom-right coordinate is (h,w)
                        predictions = PredicteDatas.reshape(Numimages, len(Datas), len(dlc_cfg['all_joints_names']), 3)
                        scores = np.copy(predictions[:,:,:,2])
                        predictions[:,:,:,2] = 1.0 # homogeneous coordinates; (x,y,1). Top-left corner is (-width/2, -height/2, 1); Bottom-right corner is opposite. Shape is num_images x num_views x num_joints x 3
                        num_ims, num_views, num_joints, _ = predictions.shape

                        # get labels in homogeneous pixel coordinates
                        labels = np.array([Data.values.reshape(num_ims, num_joints, 2) for Data in Datas]) # num_views x num_ims x num_joints x (x,y)
                        labels = np.transpose(labels, [1, 2, 0, 3]) # num_ims x num_joints x num_views x (x,y)
                        labels = np.concatenate([labels, np.ones([num_ims, num_joints, num_views, 1])], axis=3)

                        # solve linear system to get labels in 3D
                        # helpful explanation of equation found on pg 5 here: https://hal.inria.fr/inria-00524401/PDF/Sturm-cvpr05.pdf
                        labs = labels.reshape([num_ims * num_joints, num_views, 3]).astype(np.float)
                        confidences = ~np.isnan(np.sum(labs, axis=2))
                        valid = np.sum(~np.isnan(np.sum(labs, axis=2)), axis=1) >= 2
                        labs[~confidences] = 0
                        labels3d = project_3d(projection_matrices, labs, confidences=confidences)
                        labels3d[~valid] = np.nan
                        labels3d = labels3d.reshape([num_ims, num_joints, 3]) 

                        # solve linear system to get 3D predictions
                        preds = np.transpose(predictions, [0, 2, 1, 3]) # num_ims x num_joints x num_views x 3
                        preds = preds.reshape([num_ims*num_joints, num_views, 3])
                        preds3d = project_3d(projection_matrices, preds)
                        preds3d = preds3d.reshape([num_ims, num_joints, 3])
                        
                        # try it with confidence weighting
                        scores = np.transpose(scores, [0, 2, 1]) # num_images x num_joints x num_views
                        scores = np.reshape(scores, [num_ims*num_joints, num_views])
                        preds3d_weighted = project_3d(projection_matrices, preds, confidences=scores)
                        preds3d_weighted = preds3d_weighted.reshape([num_ims, num_joints, 3])

                        # try it with the pcutoff
                        scores2 = np.copy(scores)
                        scores2[scores2 < cfg["pcutoff"]] = 0
                        preds3d_weighted_cutoff = project_3d(projection_matrices, preds, confidences=scores2)
                        preds3d_weighted_cutoff = preds3d_weighted_cutoff.reshape([num_ims, num_joints, 3])

                        print("\n\n3D errors:")
                        RMSE_train = np.nanmean(np.nansum((preds3d[trainIndices] - labels3d[trainIndices])**2, axis=2)**0.5)
                        RMSE_test = np.nanmean(np.nansum((preds3d[testIndices] - labels3d[testIndices])**2, axis=2)**0.5)
                        RMSE_train_weighted = np.nanmean(np.nansum((preds3d_weighted[trainIndices] - labels3d[trainIndices])**2, axis=2)**0.5)
                        RMSE_test_weighted = np.nanmean(np.nansum((preds3d_weighted[testIndices] - labels3d[testIndices])**2, axis=2)**0.5)
                        RMSE_train_weighted_cutoff = np.nanmean(np.nansum((preds3d_weighted_cutoff[trainIndices] - labels3d[trainIndices])**2, axis=2)**0.5)
                        RMSE_test_weighted_cutoff = np.nanmean(np.nansum((preds3d_weighted_cutoff[testIndices] - labels3d[testIndices])**2, axis=2)**0.5)

                        print("RMSE train: ", RMSE_train)
                        print("RMSE test: ", RMSE_test)
                        print("RMSE train weighted: ", RMSE_train_weighted)
                        print("RMSE test weighted: ", RMSE_test_weighted)
                        print("RMSE train weighted cutoff: ", RMSE_train_weighted_cutoff)
                        print("RMSE test weighted cutoff: ", RMSE_test_weighted_cutoff) 

                        tail = np.nansum((preds3d_weighted - labels3d)**2, axis=2)**0.5
                        tail = np.sort(tail[~np.isnan(tail)])
                        tail = tail[-10:]
                        print(tail)
                        print('tail error: ', np.nanmean(tail))

                        tf.reset_default_graph()
                    elif multiview_step==2:
                        preds3d = []
                        for imageindex in tqdm(range(len(Datas[0].index))):
                            imagenames = [Data.index[imageindex] for Data in Datas]
                            images = [io.imread(os.path.join(cfg['project_path'],imagename),mode='RGB') for imagename in imagenames]
                            images = [skimage.color.gray2rgb(image) for image in images]
                            image_batch = images
                            
                            # Compute prediction with the CNN
                            outputs_np = sess.run(outputs, feed_dict={inputs: image_batch})
                            pred_3d = outputs_np[2]
                            preds3d.append(pred_3d)

                        sess.close() #closes the current tf session
                        preds3d = np.array(preds3d) # num_ims x num_joints x (x,y,z)
                        num_ims, num_joints = preds3d.shape[:2]
                        num_views = dlc_cfg.num_views

                        # get labels in homogeneous pixel coordinates
                        labels = np.array([Data.values.reshape(num_ims, num_joints, 2) for Data in Datas]) # num_views x num_ims x num_joints x (x,y)
                        labels = np.transpose(labels, [1, 2, 0, 3]) # num_ims x num_joints x num_views x (x,y)
                        labels = np.concatenate([labels, np.ones([num_ims, num_joints, num_views, 1])], axis=3)

                        # solve linear system to get labels in 3D
                        # helpful explanation of equation found on pg 5 here: https://hal.inria.fr/inria-00524401/PDF/Sturm-cvpr05.pdf
                        labs = labels.reshape([num_ims * num_joints, num_views, 3]).astype(np.float)
                        confidences = ~np.isnan(np.sum(labs, axis=2))
                        valid = np.sum(~np.isnan(np.sum(labs, axis=2)), axis=1) >= 2
                        labs[~confidences] = 0
                        labels3d = project_3d(projection_matrices, labs, confidences=confidences)
                        labels3d[~valid] = np.nan
                        labels3d = labels3d.reshape([num_ims, num_joints, 3]) 

                        print("\n\n3D errors:")
                        RMSE_train = np.nanmean(np.nansum((preds3d[trainIndices] - labels3d[trainIndices])**2, axis=2)**0.5)
                        RMSE_test = np.nanmean(np.nansum((preds3d[testIndices] - labels3d[testIndices])**2, axis=2)**0.5)

                        print("RMSE train: ", RMSE_train)
                        print("RMSE test: ", RMSE_test)

                        tail = np.nansum((preds3d- labels3d)**2, axis=2)**0.5
                        tail = np.sort(tail[~np.isnan(tail)])
                        tail = tail[-10:]
                        print(tail)
                        print('tail error: ', np.nanmean(tail))

                        tf.reset_default_graph()
                    else:
                        print('invalid multiview_step given')
                        return
            make_results_file(final_result,evaluationfolder,DLCscorer)
            print("The network is evaluated and the results are stored in the subdirectory 'evaluation_results'.")
            print("If it generalizes well, choose the best model for prediction and update the config file with the appropriate index for the 'snapshotindex'.\nUse the function 'analyze_video' to make predictions on new videos.")
            print("Otherwise consider retraining the network (see DeepLabCut workflow Fig 2)")
    
    #returning to intial folder
    os.chdir(str(start_path))

# solve linear system to get 3D coordinates
# helpful explanation of equation found on pg 5 here: https://hal.inria.fr/inria-00524401/PDF/Sturm-cvpr05.pdf
# projection_matrices is a list, predictions is an array of shape n x num_views x 3
# confidences is n x num_views
def project_3d(projection_matrices, predictions, confidences=None):
    n, num_views = predictions.shape[:2]
    A1 = np.tile(np.vstack(projection_matrices)[None], [n, 1, 1])
    A2 = np.zeros([n, 3*num_views, num_views])
    A = np.concatenate([A1, A2], axis=2).astype(np.float)
    if confidences is not None:
        A[:,:,:4] *= np.repeat(confidences, 3, axis=1)[:,:,None]
        predictions = np.copy(predictions) * confidences[:,:,None]
    updates_rows = np.arange(3*num_views)
    updates_cols = np.repeat(np.arange(num_views)+4, 3)
    A[:,updates_rows, updates_cols] = -1*predictions.reshape([n, num_views*3])
    u, s, vh = np.linalg.svd(A)
    preds3d = vh[:,-1] # bottom row of V^T is eigenvector of smallest singular value
    preds3d = preds3d[:,:3] / preds3d[:,3,None]
    return preds3d

    
def make_results_file(final_result,evaluationfolder,DLCscorer):
    """
    Makes result file in .h5 and csv format and saves under evaluation_results directory
    """
    col_names = ["Training iterations:","%Training dataset","Shuffle number"," Train error(px)"," Test error(px)","p-cutoff used","Train error with p-cutoff","Test error with p-cutoff"]
    df = pd.DataFrame(final_result, columns = col_names)
    df.to_hdf(os.path.join(str(evaluationfolder),DLCscorer + '-results' + '.h5'),'df_with_missing',format='table',mode='w')
    df.to_csv(os.path.join(str(evaluationfolder),DLCscorer + '-results' + '.csv'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    cli_args = parser.parse_args()
