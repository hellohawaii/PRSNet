import tensorflow as tf
import numpy as np
from PRSNet import PRSNet
from utils.visualization import dump_mesh_with_planes_PCA
from criterion import compute_total_loss
# Why import criterion not working?
from pathlib import Path, PurePath
from shutil import copyfile
import os
from tqdm import tqdm
import datetime


# global variables
testing_ids = ["04379243"]  # table
# testing_data_path = r"E:\workfile\PRSNet\Preprocessed_ShapeNet\v2_using_new_distance_train"
testing_data_path = r"E:\workfile\PRSNet\Preprocessed_ShapeNet\v2_selected_test_preprocessed"
# testing_data_path = r"E:\workfile\PRSNet\Preprocessed_ShapeNet\single_model"
batch_size = 1


def load_testing_numpy_data(data_path):
    # tensorflow dataset contain WindowsPath on windows, convert it to str
    path_str = data_path.numpy()
    numpy_data = np.load(path_str)
    voxel = np.expand_dims(numpy_data["voxel"], axis=-1)  # convert from (32, 32, 32) to (32, 32, 32, 1)
    points = numpy_data["points"]
    nearest_pts = numpy_data["nearest_pts"]
    nearest_pts = numpy_data["nearest_pts"].reshape([32, 32, 32, 3])  # tmp add
    obj_path = path_str.decode('UTF-8')[0:-16] + "normed_model.obj"
    return voxel, points, nearest_pts, obj_path


@tf.function
def test_step(model, voxels, points, nearest_points):
    reflect_plane1, reflect_plane2, reflect_plane3, \
        rotation_quaternion1, rotation_quaternion2, rotation_quaternion3 = model(voxels)
    total_loss = compute_total_loss(points, reflect_plane1, reflect_plane2, reflect_plane3,
                                    rotation_quaternion1, rotation_quaternion2, rotation_quaternion3,
                                    nearest_points)
    planes = tf.stack([reflect_plane1, reflect_plane2, reflect_plane3], axis=1)
    quaternions = tf.stack([rotation_quaternion1, rotation_quaternion2, rotation_quaternion3], axis=1)
    return total_loss, planes, quaternions


def test(model, dataset, ckpt_manager):
    # check point
    if ckpt_manager.latest_checkpoint:
        print("Restored from {}".format(ckpt_manager.latest_checkpoint))
    else:
        print("No pretrained model found!")
    ckpt_manager.restore_or_initialize()
    for voxel_i, points_i, nearest_pts_i, obj_path in dataset:
        loss, planes, quaternions = test_step(model, voxel_i, points_i, nearest_pts_i)
        # visualization
        test_mesh_path = obj_path.numpy()[-1].decode('UTF-8')
        shape_id = PurePath(test_mesh_path).parts[-3]
        model_id = PurePath(test_mesh_path).parts[-2]
        rotation_name = PurePath(test_mesh_path).stem
        # TODO： copy from the original dataset
        # make dirs
        output_dir_path = 'result/' + trained_time + '/test/logs/result/' + shape_id + '/' + model_id
        os.makedirs(output_dir_path, exist_ok=True)
        # copy obj and binvox for compare
        src_obj_file = os.path.join(testing_data_path, shape_id, model_id, rotation_name + '.obj')
        dest_obj_file = os.path.join(output_dir_path, rotation_name + '.obj')
        copyfile(src_obj_file, dest_obj_file)
        src_binvox_file = os.path.join(testing_data_path, shape_id, model_id, rotation_name+'.binvox')
        dest_binvox_file = os.path.join(output_dir_path, rotation_name+'.binvox')
        copyfile(src_binvox_file, dest_binvox_file)
        # output mesh with plane
        output_path = 'result/' + trained_time + '/test/logs/result/' + shape_id + '/' + model_id + '/' + \
                      'loss' + str(int(loss)) + rotation_name + '_with_plane' + '.obj'
        dump_mesh_with_planes_PCA(test_mesh_path, output_path, planes[-1])



tf.config.experimental_run_functions_eagerly(True)

prs_model = PRSNet()
# dataset
all_testing_data = [Path(testing_data_path) / testing_id / '**/*.npz' for testing_id in testing_ids]
data_path_dataset = tf.data.Dataset.list_files(
    [str(all_testing_data_i) for all_testing_data_i in all_testing_data])
testing_dataset = data_path_dataset.map(lambda x: tf.py_function(load_testing_numpy_data, [x],
                                                                 [tf.float32, tf.float32, tf.float32, tf.string]))
testing_dataset = testing_dataset.batch(batch_size=batch_size)

trained_time = '20210304-120500'
ckpt = tf.train.Checkpoint(net=prs_model)
test_ckpt_dir = 'result/' + trained_time + '/train/model'
ckpt_manager = tf.train.CheckpointManager(ckpt, test_ckpt_dir, max_to_keep=3)
test(prs_model, testing_dataset, ckpt_manager)
