import tensorflow as tf
from utils.quaternion import quaternion_to_axis_normed, quaternion_multiply, quaternion_inverse

resolution = 32
w_r = 25


def reflect_points(points, reflect_planes):
    # points Batch * point_num * 3
    # reflect_planes Batch * 4
    # import pdb
    # pdb.set_trace()
    plane_normal = reflect_planes[:, 0: 3]  # Batch * 3
    plane_d = tf.expand_dims(reflect_planes[:, 3], axis=1)  # Batch * 1
    normal_norm = tf.norm(plane_normal, ord=2, axis=1, keepdims=True)  # Batch * 1
    # import pdb
    # pdb.set_trace()
    points2plane = tf.math.add(tf.einsum('...j,...j->...', points, tf.expand_dims(plane_normal, axis=1)), plane_d)  # Batch * point_num
    points2plane = tf.math.multiply(
        tf.expand_dims(tf.divide(points2plane, tf.math.square(normal_norm)), axis=2),  # Batch * point_num * 1
        tf.expand_dims(plane_normal, axis=1)  # Batch * 1 * 3
    )   # Batch * point_num * 3
    points_reflected = points - 2 * points2plane
    return points_reflected


def rotate_points(points, rotation_quaternion):
    # points Batch * point_num * 3
    # rotation_quaternion Batch * 4
    points_quaternion = tf.concat([tf.zeros([points.shape[0], points.shape[1], 1]), points], axis=-1)
    # import pdb
    # pdb.set_trace()
    rotation_quaternion = tf.expand_dims(rotation_quaternion, axis=1)  # rotation_quaternion Batch * 1 * 4
    points_rotated_quaternion = quaternion_multiply(quaternion_multiply(rotation_quaternion, points_quaternion),
                                                    quaternion_inverse(rotation_quaternion))
    # import pdb
    # pdb.set_trace()
    return points_rotated_quaternion[:, :, 1:4]


def get_proxy_grid_center_index(points):
    # grid_min = -0.5 + (1/resolution) * index
    # grid_max = -0.5 + (1/resolution) * (index+1)
    # index start from 0
    # import pdb
    # pdb.set_trace()
    index = tf.floor(((points + tf.expand_dims(tf.constant([0.5, 0.5, 0.5]), axis=0)) * resolution))
    return tf.minimum(tf.math.maximum(tf.cast(index, tf.int32), 0), 31)


def distance_loss(points, points_prime, grid_center2points):
    # points Batch * point_num * 3
    # grid_center2points Batch * resolution * resolution * resolution * 3
    # USE TF.GATHER_ND
    proxy_grid_center_index = get_proxy_grid_center_index(points_prime)  # proxy_grid_center_index Batch * point_num * 3
    closet_points = tf.gather_nd(grid_center2points, proxy_grid_center_index, batch_dims=1)
    # USE TF.GATHER
    # grid_center2points = tf.reshape(grid_center2points, [-1, resolution*resolution*resolution, 3])
    # # grid_center2points Batch * resolution^3 * 3
    # proxy_grid_center_index = get_proxy_grid_center_index(points_prime)  # proxy_grid_center_index Batch * point_num * 3
    # flatten_weight = tf.expand_dims(tf.expand_dims(tf.constant([resolution*resolution,resolution,1]), axis=0), axis=0)
    # proxy_grid_center_index = tf.reduce_sum(tf.math.multiply(proxy_grid_center_index, flatten_weight), axis=-1)
    # # proxy_grid_center_index Batch * point_num
    # # closet_points = grid_center2points.numpy()[proxy_grid_center_index]  # want Batch * point_num * 3
    # closet_points = tf.gather(grid_center2points, proxy_grid_center_index, axis=1, batch_dims=1)
    return tf.math.reduce_sum(tf.norm(closet_points - points_prime, ord=2, axis=2), axis=1)


def regulation_reflect_loss(reflect_plane1, reflect_plane2, reflect_plane3):
    norm1 = reflect_plane1[:, 0:3]
    norm1 = tf.divide(norm1, tf.norm(norm1, ord=2, axis=1, keepdims=True))
    norm2 = reflect_plane2[:, 0:3]
    norm2 = tf.divide(norm2, tf.norm(norm2, ord=2, axis=1, keepdims=True))
    norm3 = reflect_plane3[:, 0:3]
    norm3 = tf.divide(norm3, tf.norm(norm3, ord=2, axis=1, keepdims=True))
    m1 = tf.stack([norm1, norm2, norm3], axis=1)
    a = tf.linalg.matmul(m1, m1, transpose_a=False, transpose_b=True) - tf.eye(num_rows=3, batch_shape=norm1.shape[0:1])
    return tf.square(tf.norm(a, ord='euclidean', axis=[1, 2]))


def regulation_rotation_loss(rotation_quaternion1, rotation_quaternion2, rotation_quaternion3):
    axis1 = quaternion_to_axis_normed(rotation_quaternion1)
    axis2 = quaternion_to_axis_normed(rotation_quaternion2)
    axis3 = quaternion_to_axis_normed(rotation_quaternion3)
    m2 = tf.stack([axis1, axis2, axis3], axis=1)
    b = tf.linalg.matmul(m2, m2, transpose_a=False, transpose_b=True) \
        - tf.eye(num_rows=3, batch_shape=rotation_quaternion1.shape[0:1])
    return tf.square(tf.norm(b, ord='euclidean', axis=[1, 2]))


def compute_total_loss(points,
                       reflect_plane1, reflect_plane2, reflect_plane3,
                       rotation_quaternion1, rotation_quaternion2, rotation_quaternion3,
                       nearest_pt_to_grids):
    reflected_points1 = reflect_points(points, reflect_plane1)
    reflected_points2 = reflect_points(points, reflect_plane2)
    reflected_points3 = reflect_points(points, reflect_plane3)
    rotated_points1 = rotate_points(points, rotation_quaternion1)
    rotated_points2 = rotate_points(points, rotation_quaternion2)
    rotated_points3 = rotate_points(points, rotation_quaternion3)
    # import pdb
    # pdb.set_trace()
    total_distance_loss = distance_loss(points, reflected_points1, nearest_pt_to_grids) + \
        distance_loss(points, reflected_points2, nearest_pt_to_grids) + \
        distance_loss(points, reflected_points3, nearest_pt_to_grids) + \
        distance_loss(points, rotated_points1, nearest_pt_to_grids) + \
        distance_loss(points, rotated_points2, nearest_pt_to_grids) + \
        distance_loss(points, rotated_points3, nearest_pt_to_grids)
    regulation_reflect_loss_computed = regulation_reflect_loss(reflect_plane1, reflect_plane2, reflect_plane3)
    regulation_rotation_loss_computed = regulation_rotation_loss(rotation_quaternion1, rotation_quaternion2,
                                                                 rotation_quaternion3)
    total_loss = total_distance_loss + w_r * (regulation_reflect_loss_computed + regulation_rotation_loss_computed)
    return total_loss
