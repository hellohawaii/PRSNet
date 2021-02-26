import trimesh
import os
import numpy as np
from utils.visualization import visualize_voxel
from tqdm import tqdm


shape_net_path = r"E:\workfile\PRSNet\ShapeNetCore.v2"
selected_shape_ids = ["04379243"]  # table
process_output_path = r"E:\workfile\PRSNet\Preprocessed_ShapeNet\V2"
binvox_path = r"E:\workfile\PRSNet\binvox\binvox.exe"
resolution = 32

for selected_shape_id in selected_shape_ids:
    models_ids = os.listdir(os.path.join(shape_net_path, selected_shape_id))
    # referring to https://stackoverflow.com/questions/23113494/double-progress-bar-in-python
    for model_id in tqdm(models_ids, position=0, leave=True):
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
        rotation_times = 4000 // len(models_ids) + 1
        for rotate_i in range(rotation_times):
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
            # TODO: find a more efficient way
            nearest_pts = np.empty([resolution, resolution, resolution, 3])
            for x_coor in range(resolution):
                for y_coor in range(resolution):
                    for z_coor in range(resolution):
                        half_resolution = resolution / 2
                        offset = np.array([-half_resolution + 0.5, -half_resolution + 0.5, -half_resolution + 0.5])
                        xyz_coor = (np.array([x_coor, y_coor, z_coor], dtype=np.float32) + offset) / resolution
                        nearest_pt_id = np.argmin(np.sum(np.square((xyz_coor - sample_pts)), axis=1))
                        nearest_pt = sample_pts[nearest_pt_id]
                        nearest_pts[x_coor, y_coor, z_coor, :] = nearest_pt

            output_numpy_path = os.path.join(process_output_path, selected_shape_id, model_id,
                                             "rotated_"+str(rotate_i)+"_preprocessed.npz")
            np.savez(output_numpy_path, voxel=voxel_numpy, points=sample_pts, nearest_pts=nearest_pts)


            # ttt = np.load(output_numpy_path)
            # print(ttt)
