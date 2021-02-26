import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten, Conv3D, MaxPool3D


class PRSNet(Model):
    def __init__(self):
        super(PRSNet, self).__init__()
        self.conv1 = Conv3D(filters=4, kernel_size=3, strides=1, padding='same', activation='relu')
        self.max_pooling1 = MaxPool3D(pool_size=(2, 2, 2), strides=(2, 2, 2))
        self.conv2 = Conv3D(filters=8, kernel_size=3, strides=1, padding='same', activation='relu')
        self.max_pooling2 = MaxPool3D(pool_size=(2, 2, 2), strides=(2, 2, 2))
        self.conv3 = Conv3D(filters=16, kernel_size=3, strides=1, padding='same', activation='relu')
        self.max_pooling3 = MaxPool3D(pool_size=(2, 2, 2), strides=(2, 2, 2))
        self.conv4 = Conv3D(filters=32, kernel_size=3, strides=1, padding='same', activation='relu')
        self.max_pooling4 = MaxPool3D(pool_size=(2, 2, 2), strides=(2, 2, 2))
        self.conv5 = Conv3D(filters=32, kernel_size=3, strides=1, padding='same', activation='relu')
        self.max_pooling5 = MaxPool3D(pool_size=(2, 2, 2), strides=(2, 2, 2))
        self.conv = tf.keras.Sequential([self.conv1, self.max_pooling1,
                                         self.conv2, self.max_pooling2,
                                         self.conv3, self.max_pooling3,
                                         self.conv4, self.max_pooling4,
                                         self.conv5, self.max_pooling5])
        self.linear1_reflect1 = Dense(32)
        self.linear2_reflect1 = Dense(16)
        self.linear3_reflect1 = Dense(4)
        self.linear_reflect1 = tf.keras.Sequential([self.linear1_reflect1, self.linear3_reflect1,
                                                    self.linear3_reflect1])
        self.linear1_reflect2 = Dense(32)
        self.linear2_reflect2 = Dense(16)
        self.linear3_reflect2 = Dense(4)
        self.linear_reflect2 = tf.keras.Sequential([self.linear1_reflect2, self.linear3_reflect2,
                                                    self.linear3_reflect2])
        self.linear1_reflect3 = Dense(32)
        self.linear2_reflect3 = Dense(16)
        self.linear3_reflect3 = Dense(4)
        self.linear_reflect3 = tf.keras.Sequential([self.linear1_reflect3, self.linear3_reflect3,
                                                    self.linear3_reflect3])
        self.linear1_rotation1 = Dense(32)
        self.linear2_rotation1 = Dense(16)
        self.linear3_rotation1 = Dense(4)
        self.linear_rotation1 = tf.keras.Sequential([self.linear1_rotation1, self.linear3_rotation1,
                                                     self.linear3_rotation1])
        self.linear1_rotation2 = Dense(32)
        self.linear2_rotation2 = Dense(16)
        self.linear3_rotation2 = Dense(4)
        self.linear_rotation2 = tf.keras.Sequential([self.linear1_rotation2, self.linear3_rotation2,
                                                     self.linear3_rotation2])
        self.linear1_rotation3 = Dense(32)
        self.linear2_rotation3 = Dense(16)
        self.linear3_rotation3 = Dense(4)
        self.linear_rotation3 = tf.keras.Sequential([self.linear1_rotation3, self.linear3_rotation3,
                                                     self.linear3_rotation3])

    def call(self, voxels):
        conv_feature = self.conv(voxels)
        # import pdb
        # pdb.set_trace()
        reflect_plane1 = tf.squeeze(self.linear_reflect1(conv_feature), axis=[1, 2, 3])
        reflect_plane2 = tf.squeeze(self.linear_reflect2(conv_feature), axis=[1, 2, 3])
        reflect_plane3 = tf.squeeze(self.linear_reflect3(conv_feature), axis=[1, 2, 3])
        rotation_quaternion1 = tf.squeeze(self.linear_rotation1(conv_feature), axis=[1, 2, 3])
        rotation_quaternion2 = tf.squeeze(self.linear_rotation2(conv_feature), axis=[1, 2, 3])
        rotation_quaternion3 = tf.squeeze(self.linear_rotation3(conv_feature), axis=[1, 2, 3])
        return reflect_plane1, reflect_plane2, reflect_plane3, \
            rotation_quaternion1, rotation_quaternion2, rotation_quaternion3
