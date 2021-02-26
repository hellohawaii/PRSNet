import tensorflow as tf
import numpy as np
from PRSNet import PRSNet
from utils.visualization import dump_mesh_with_planes
from criterion import compute_total_loss
# Why import criterion not working?
from pathlib import Path
from tqdm import tqdm
import datetime


# global variables
testing_ids = ["04379243"]  # table
testing_data_path = r"E:\workfile\PRSNet\Preprocessed_ShapeNet\V2"
batch_size = 1


def load_testing_numpy_data(data_path):
    # tensorflow dataset contain WindowsPath on windows, convert it to str
    path_str = data_path.numpy()
    numpy_data = np.load(path_str)
    voxel = np.expand_dims(numpy_data["voxel"], axis=-1)  # convert from (32, 32, 32) to (32, 32, 32, 1)
    points = numpy_data["points"]
    nearest_pts = numpy_data["nearest_pts"]
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
        output_path = 'result/' + trained_time + '/test/logs/result' + str(optimizer.iterations.numpy()) + '.obj'
        dump_mesh_with_planes(test_mesh_path, output_path, planes[-1])


tf.config.experimental_run_functions_eagerly(True)

prs_model = PRSNet()
# dataset
all_testing_data = [Path(testing_data_path) / testing_id / '**/*.npz' for testing_id in testing_ids]
data_path_dataset = tf.data.Dataset.list_files(
    [str(all_testing_data_i) for all_testing_data_i in all_testing_data])
testing_dataset = data_path_dataset.map(lambda x: tf.py_function(load_testing_numpy_data, [x],
                                                                 [tf.float32, tf.float32, tf.float32, tf.string]))
testing_dataset = testing_dataset.batch(batch_size=batch_size)

trained_time = '20210226-183451'
ckpt = tf.train.Checkpoint(net=prs_model)
test_ckpt_dir = 'result/' + trained_time + '/train/model'
ckpt_manager = tf.train.CheckpointManager(ckpt, test_ckpt_dir, max_to_keep=3)
test(prs_model, testing_dataset, ckpt_manager)
