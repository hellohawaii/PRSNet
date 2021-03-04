import tensorflow as tf
import numpy as np
from PRSNet import PRSNet
from utils.visualization import dump_mesh_with_planes
from criterion import compute_total_loss
# Why import criterion not working?
from pathlib import Path
from tqdm import tqdm
import datetime
import trimesh
# import tensorflow_graphics as tfg
# from tensorboard.plugins.mesh import summary as mesh_summary

# import global_settings  # TODO: how to have global settings? Using a class？

# global variables
training_ids = ["04379243"]  # table
# training_data_path = r"E:\workfile\PRSNet\Preprocessed_ShapeNet\V2"
training_data_path = r"/content/Data/v2_using_new_distance_train"
# training_data_path = r"E:\workfile\PRSNet\Preprocessed_ShapeNet\single_model"
max_epochs = 50000
batch_size = 32


def load_training_numpy_data(data_path):
    # tensorflow dataset contain WindowsPath on windows, convert it to str
    path_str = data_path.numpy()
    # import pdb
    # pdb.set_trace()
    numpy_data = np.load(path_str)
    voxel = np.expand_dims(numpy_data["voxel"], axis=-1)  # convert from (32, 32, 32) to (32, 32, 32, 1)
    points = numpy_data["points"]
    nearest_pts = numpy_data["nearest_pts"]
    nearest_pts = numpy_data["nearest_pts"].reshape([32, 32, 32, 3])  # tmp add
    obj_path = path_str.decode('UTF-8')[0:-16] + "normed_model.obj"
    return voxel, points, nearest_pts, obj_path
    # return tf.io.read_file(data_path)


@tf.function
def train_step(model, optimizer, voxels, points, nearest_points):
    with tf.GradientTape() as tape:
        # import pdb
        # pdb.set_trace()
        reflect_plane1, reflect_plane2, reflect_plane3, \
            rotation_quaternion1, rotation_quaternion2, rotation_quaternion3 = model(voxels)
        # import pdb
        # pdb.set_trace()
        total_loss = compute_total_loss(points, reflect_plane1, reflect_plane2, reflect_plane3,
                                        rotation_quaternion1, rotation_quaternion2, rotation_quaternion3,
                                        nearest_points)
    gradients = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    # return planes and quaternions for visualization
    planes = tf.stack([reflect_plane1, reflect_plane2, reflect_plane3], axis=1)
    quaternions = tf.stack([rotation_quaternion1, rotation_quaternion2, rotation_quaternion3], axis=1)
    return total_loss, planes, quaternions


def train(model, optimizer, dataset, ckpt_manager, log_freq=10, save_freq_step=1000):
    # check point
    # import pdb
    # pdb.set_trace()
    ckpt_manager.restore_or_initialize()
    if ckpt_manager.latest_checkpoint:
        print("Restored from {}".format(ckpt_manager.latest_checkpoint))
    else:
        print("Initializing from scratch.")
    avg_loss = tf.keras.metrics.Mean(name='loss', dtype=tf.float32)
    epoch_avg_loss = tf.keras.metrics.Mean(name='epoch_mean_loss', dtype=tf.float32)
    for epoch in tqdm(range(max_epochs)):
        for voxel_i, points_i, nearest_pts_i, obj_path in dataset:
            loss, planes, quaternions = train_step(model, optimizer, voxel_i, points_i, nearest_pts_i)
            avg_loss.update_state(loss)  # add the new loss record to the log
            epoch_avg_loss.update_state(loss)
            # check point
            ckpt_manager.checkpoint.step.assign_add(1)
            if int(ckpt_manager.checkpoint.step) % save_freq_step == 0:
                save_path = ckpt_manager.save()
                print("Saved checkpoint for step {}: {}".format(int(ckpt.step), save_path))
                # print("loss {:1.2f}".format(loss.result().numpy()))
            # add loss to summary and then reset log
            if tf.equal(optimizer.iterations % log_freq, 0):
                tf.summary.scalar('loss', avg_loss.result(), step=optimizer.iterations)
                avg_loss.reset_states()
                # for debugging checkpoint
                # tf.summary.scalar('checkpoint step', ckpt_manager.checkpoint.step.numpy(), step=optimizer.iterations)
            # visualization
            if int(ckpt_manager.checkpoint.step) % log_freq == 0:
                # import pdb
                # pdb.set_trace()
                # mesh_to_be_visulized = trimesh.load(obj_path.numpy()[-1].decode('UTF-8'))
                # mesh_to_be_visulized = tfg.io.triangle_mesh.load(obj_path)
                # import pdb
                # pdb.set_trace()
                test_mesh_path = obj_path.numpy()[-1].decode('UTF-8')
                # output_path = test_mesh_path[0:-4]+"_with_result.obj"
                output_path = 'result/' + current_time + '/train/logs/result' + str(int(ckpt_manager.checkpoint.step)) + '.obj'
                try:
                    dump_mesh_with_planes(test_mesh_path, output_path, planes[-1])
                except:
                    pass
                # mesh_to_be_visulized = trimesh.load(output_path)
                # mesh_summary.op('mesh', vertices=np.expand_dims(mesh_to_be_visulized.vertices, axis=0),
                #                 faces=np.expand_dims(mesh_to_be_visulized.faces, axis=0))

        tf.summary.scalar('epoch_mean_loss', epoch_avg_loss.result(), step=epoch)
        epoch_avg_loss.reset_states()


tf.config.experimental_run_functions_eagerly(True)

prs_model = PRSNet()
adam_optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
# dataset
all_training_data = [Path(training_data_path) / training_id / '**/*.npz' for training_id in training_ids]
data_path_dataset = tf.data.Dataset.list_files(
    [str(all_training_data_i) for all_training_data_i in all_training_data])
training_dataset = data_path_dataset.map(lambda x: tf.py_function(load_training_numpy_data, [x],
                                                                  [tf.float32, tf.float32, tf.float32, tf.string]))
training_dataset = training_dataset.batch(batch_size=batch_size)

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# current_time = '20210226-183451'
train_log_dir = 'result/' + current_time + '/train/logs'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)

# checkpoint
# train_ckpt_dir = 'result/' + '20210226-152308' + '/train/model'
train_ckpt_dir = 'result/' + current_time + '/train/model'
# if need to continue training, replace the current time with certain dir
ckpt = tf.train.Checkpoint(optimizer=adam_optimizer, net=prs_model, step=tf.Variable(1))
ckpt_manager = tf.train.CheckpointManager(ckpt, train_ckpt_dir, max_to_keep=3)
with train_summary_writer.as_default():
    train(prs_model, adam_optimizer, training_dataset, ckpt_manager, log_freq=10, save_freq_step=200)
