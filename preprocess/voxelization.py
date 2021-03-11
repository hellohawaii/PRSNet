import trimesh
import os
import numpy as np
from utils.visualization import visualize_voxel
from tqdm import tqdm


shape_net_path = r"E:\workfile\PRSNet\Preprocessed_ShapeNet\v2_selected_test_before_preprocessing"
selected_shape_ids = ["04379243"]  # table
process_output_path = r"E:\workfile\PRSNet\Preprocessed_ShapeNet\v2_selected_test_preprocessed"
binvox_path = r"E:\workfile\PRSNet\binvox\binvox.exe"
resolution = 32
using_500_model = True

for selected_shape_id in selected_shape_ids:
    models_ids = os.listdir(os.path.join(shape_net_path, selected_shape_id))
    if using_500_model and len(models_ids) > 500:
        models_ids = models_ids[0:500]
    # referring to https://stackoverflow.com/questions/23113494/double-progress-bar-in-python
    for model_id in tqdm(models_ids, position=0, leave=True):
    # for model_id in models_ids:
        model_path = os.path.join(shape_net_path, selected_shape_id, model_id, "models", "model_normalized.obj")
        output_dir_path = os.path.join(process_output_path, selected_shape_id, model_id)
        if not os.path.isdir(output_dir_path):
            os.makedirs(output_dir_path)
        assert os.path.exists(model_path)
        mesh = trimesh.load(model_path)
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump(concatenate=True)
        # Remark, mesh loaded from ShapeNet have several parts, so here is mesh.scene, which store these parts
        # as mesh.geometry, a dict of trimesh. This make voxelization difficult, trimesh.exchange.binvox.voxelize_mesh
        # not working. use dump to convert it into a single mesh
        if using_500_model:
            rotation_times = 150
        else:
            rotation_times = 4000 // len(models_ids) + 1
        try:
            for rotate_i in tqdm(range(rotation_times), position=1, leave=True):
                mesh.apply_transform(trimesh.transformations.random_rotation_matrix())
                bounds = mesh.bounds  # 2*3, min and max for 3 dimensions
                # mesh.vertices()
                bounds_mid = np.mean(bounds, axis=0)
                T = trimesh.transformations.translation_matrix(-bounds_mid)
                bounds_swap = bounds[:, [1, 0]]
                max_scale = np.ndarray.max((bounds[[1, 0], :] - bounds)[0])
                S = trimesh.transformations.scale_matrix(1 / max_scale, [0, 0, 0])
                mesh.apply_transform(T)
                mesh.apply_transform(S)
                # mesh in [-0.5, 0.5]*[-0.5, 0.5]*[-0.5, 0.5]

                binvoxer = trimesh.exchange.binvox.Binvoxer(dimension=32, use_offscreen_pbuffer=False,
                                                            verbose=False, binvox_path=binvox_path)
                # output_mesh
                output_mesh_path = os.path.join(process_output_path, selected_shape_id, model_id,
                                                "rotated_"+str(rotate_i)+"_normed_model.obj")
                mesh.export(output_mesh_path)

                # dump using binvox
                binvox_output_file = binvoxer(output_mesh_path, overwrite=True)
                with open(binvox_output_file, "rb") as f:
                    # f.readline().strip()
                    # tt1 = trimesh.exchange.binvox.parse_binvox(f)
                    voxel_grid = trimesh.exchange.binvox.load_binvox(f)
                    f.close()

                # use trimesh.exchange.binvox.voxelize_mesh
                # this function also need to export mesh, but will clean it up
                # voxel_grid = trimesh.exchange.binvox.voxelize_mesh(mesh, binvoxer=binvoxer, export_type='obj')

                voxel_numpy = np.array(voxel_grid.matrix).astype(int)
                # sample on the mesh
                sample_pts, _ = trimesh.sample.sample_surface(mesh, count=1000)

                # # SOLUTION ONE
                # # try to use Structure of Array to generate coordinates
                # # in x/y/z_coors, x index change first, x majority
                # # x: 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2
                # # y: 0, 0, 0, 1, 1, 1, 2, 2, 2, 0, 0, 0, 1, 1, 1, 2, 2, 2, 0, 0, 0, 1, 1, 1, 2, 2, 2,
                # # z: 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                # coor_before_reshape = np.tile(np.arange(0, resolution), [resolution*resolution]).reshape([resolution, resolution, resolution])
                # x_coors = coor_before_reshape.transpose([0, 1, 2]).flatten()
                # y_coors = coor_before_reshape.transpose([0, 2, 1]).flatten()
                # z_coors = coor_before_reshape.transpose([2, 0, 1]).flatten()
                # # xyz_coors index like this: [x,y,z] = xyz_coors[z_index, y_index, x_index] after executing the line below
                # xyz_coors = np.stack([x_coors, y_coors, z_coors], axis=1).reshape([resolution, resolution, resolution, 3])
                # # xyz_coors index like this: [x,y,z] = xyz_coors[x_index, y_index, z_index] after executing the line below
                # xyz_coors = xyz_coors.transpose([2, 1, 0, 3])
                # # perhaps if we use z majority during constructing x/y/z_coors, we do not need transpose after reshaping
                # half_resolution = resolution / 2
                # offset = np.array([-half_resolution + 0.5, -half_resolution + 0.5, -half_resolution + 0.5])
                # xyz_coors = (xyz_coors + offset) / resolution

                # # SOLUTION TWO
                # # in x/y/z_coors, z index change first, z majority
                # # z: 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2
                # # y: 0, 0, 0, 1, 1, 1, 2, 2, 2, 0, 0, 0, 1, 1, 1, 2, 2, 2, 0, 0, 0, 1, 1, 1, 2, 2, 2,
                # # x: 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                # coor_before_reshape = np.tile(np.arange(0, resolution), [resolution * resolution]).reshape(
                #     [resolution, resolution, resolution])
                # z_coors = coor_before_reshape.transpose([0, 1, 2]).flatten()
                # y_coors = coor_before_reshape.transpose([0, 2, 1]).flatten()
                # x_coors = coor_before_reshape.transpose([2, 0, 1]).flatten()
                # # xyz_coors index like this: [x,y,z] = xyz_coors[x_index, y_index, z_index] after executing the line below
                # xyz_coors = np.stack([x_coors, y_coors, z_coors], axis=1).reshape([resolution, resolution, resolution, 3])
                # half_resolution = resolution / 2
                # offset = np.array([-half_resolution + 0.5, -half_resolution + 0.5, -half_resolution + 0.5])
                # xyz_coors = (xyz_coors + offset) / resolution

                # SOLUTION THREE
                # using meshgrid
                zero2resolution = np.arange(0, resolution)
                x_coors, y_coors, z_coors = np.meshgrid(zero2resolution, zero2resolution, zero2resolution, indexing='ij')
                xyz_coors = np.stack([x_coors, y_coors, z_coors], axis=-1).reshape([resolution, resolution, resolution, 3])
                # the following is the same as the code above
                half_resolution = resolution / 2
                offset = np.array([-half_resolution + 0.5, -half_resolution + 0.5, -half_resolution + 0.5])
                xyz_coors = (xyz_coors + offset) / resolution

                # reshape to list, find nearest pts, then reshape the result back to N*N*N*3
                xyz_coors = xyz_coors.reshape([-1, 3])
                nearest_pts, _, _ = mesh.nearest.on_surface(xyz_coors)
                nearest_pts = nearest_pts.reshape([resolution, resolution, resolution, 3])

                output_numpy_path = os.path.join(process_output_path, selected_shape_id, model_id,
                                                 "rotated_"+str(rotate_i)+"_preprocessed.npz")
                np.savez(output_numpy_path, voxel=voxel_numpy, points=sample_pts, nearest_pts=nearest_pts)
        except MemoryError:
            print(model_path)
            continue
