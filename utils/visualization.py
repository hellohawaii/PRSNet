import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
# import pyvista as pv
import trimesh
from sklearn.decomposition import PCA


def visualize_voxel(volume):
    # refer to https://stackoverflow.com/a/45971363/9758790
    # Create the x, y, and z coordinate arrays.  We use
    # numpy's broadcasting to do all the hard work for us.
    # We could shorten this even more by using np.meshgrid.
    x = np.arange(volume.shape[0])[:, None, None]
    y = np.arange(volume.shape[1])[None, :, None]
    z = np.arange(volume.shape[2])[None, None, :]
    x, y, z = np.broadcast_arrays(x, y, z)

    # Turn the volumetric data into an RGB array that's
    # just grayscale.  There might be better ways to make
    # ax.scatter happy.
    c = np.tile(volume.ravel()[:, None], [1, 3])

    # Do the plotting in a single call.
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.scatter(x.ravel(),
               y.ravel(),
               z.ravel(),
               c=c)
    fig.show()


def dump_mesh_with_planes(mesh_file_path, output_mesh_file, planes):
    mesh = trimesh.load(mesh_file_path)
    if isinstance(mesh, trimesh.scene.scene.Scene):
        pass
    else:
        mesh = trimesh.scene.scene.Scene(geometry=mesh)
    tolerance = 2.5
    for i in range(planes.shape[0]):
        a = planes[i, 0]
        b = planes[i, 1]
        c = planes[i, 2]
        d = planes[i, 3]
        z0 = (-d+0.5*a+0.5*b)/c
        z1 = (-d+0.5*a-0.5*b)/c
        z2 = (-d-0.5*a+0.5*b)/c
        z3 = (-d-0.5*a-0.5*b)/c
        if -0.5 <= z0/tolerance <= 0.5 and -0.5 <= z1/tolerance <= 0.5 and -0.5 <= z2/tolerance <= 0.5 and -0.5 <= z3/tolerance <= 0.5:
            # based on XY
            v0 = [-0.5, -0.5, z0]
            v1 = [-0.5, +0.5, z1]
            v2 = [+0.5, -0.5, z2]
            v3 = [+0.5, +0.5, z3]
        else:
            y0 = (-d+0.5*a+0.5*c)/b
            y1 = (-d+0.5*a-0.5*c)/b
            y2 = (-d-0.5*a+0.5*c)/b
            y3 = (-d-0.5*a-0.5*c)/b
            if -0.5 <= y0/tolerance <= 0.5 and -0.5 <= y1/tolerance <= 0.5 and -0.5 <= y2/tolerance <= 0.5 and -0.5 <= y3/tolerance <= 0.5:
                v0 = [-0.5, y0, -0.5]
                v1 = [-0.5, y1, +0.5]
                v2 = [+0.5, y2, -0.5]
                v3 = [+0.5, y3, +0.5]
            else:
                x0 = (-d + 0.5 * c + 0.5 * b) / a
                x1 = (-d - 0.5 * c + 0.5 * b) / a
                x2 = (-d + 0.5 * c - 0.5 * b) / a
                x3 = (-d - 0.5 * c - 0.5 * b) / a
                if -0.5 <= x0/tolerance <= 0.5 and -0.5 <= x1/tolerance <= 0.5 and -0.5 <= x2/tolerance <= 0.5 and -0.5 <= x3/tolerance <= 0.5:
                    v0 = [x0, -0.5, -0.5]
                    v1 = [x1, -0.5, +0.5]
                    v2 = [x2, +0.5, -0.5]
                    v3 = [x3, +0.5, +0.5]
                else:
                    # TODO: compare the area and find the smallest triangle
                    v0 = [x0, -0.5, -0.5]
                    v1 = [x1, -0.5, +0.5]
                    v2 = [x2, +0.5, -0.5]
                    v3 = [x3, +0.5, +0.5]
        try:
            plane_mesh = trimesh.Trimesh(vertices=[v0, v1, v2, v3], faces=[[0, 1, 2], [3, 2, 1]])
        except:
            import pdb
            pdb.set_trace()
        mesh.add_geometry(plane_mesh)
    # if isinstance(mesh, trimesh.Scene):
    #     mesh = mesh.dump(concatenate=True)
    mesh.export(output_mesh_file)


def dump_mesh_with_planes_PCA(mesh_file_path, output_mesh_file, planes):
    mesh = trimesh.load(mesh_file_path)
    if isinstance(mesh, trimesh.scene.scene.Scene):
        # print("scene")
        mesh = mesh.dump(concatenate=True)
    else:
        # print("mesh")
        mesh = trimesh.scene.scene.Scene(geometry=mesh)
    for i in range(planes.shape[0]):
        # projection
        geometry_list = list(mesh.geometry.items())
        # print(len(geometry_list))
        mesh_vertices = geometry_list[0][1].vertices
        # assuming line is represented in parametric equation, solving with the plane equation simultaneously
        plane_abc = planes[i, 0:3]
        plane_d = planes[i, 3]
        # t is the parameter of the line's parametric function
        t = np.divide(np.add(np.inner(mesh_vertices, plane_abc), plane_d), np.sum(np.square(plane_abc)))
        projected_mesh_vertices = mesh_vertices - \
            np.multiply(np.expand_dims(t, axis=1), np.expand_dims(plane_abc, axis=0))
        # do PCA
        pca = PCA(n_components=2)
        projected_mesh_vertices_pca_space = pca.fit_transform(projected_mesh_vertices)
        pca_space_lower_bound = np.amin(projected_mesh_vertices_pca_space, axis=0)
        pca_space_upper_bound = np.amax(projected_mesh_vertices_pca_space, axis=0)
        # compute plane in pca space
        area_factor = 1.2  # change the size of visualized plane related to mesh
        area_factor = area_factor/2.0 + 0.5
        pca_space_span = (pca_space_upper_bound - pca_space_lower_bound) * area_factor
        plane_pca_space_lower_bound = pca_space_upper_bound - pca_space_span  # shape (3,)
        plane_pca_space_upper_bound = pca_space_lower_bound + pca_space_span  # shape (3,)
        plane_axis_range = np.stack([plane_pca_space_lower_bound, plane_pca_space_upper_bound]).T  # plane_axis_range[i] is the
        # range of the ith pca axis.
        pca_x1, pca_y1 = np.meshgrid(plane_axis_range[0,:], plane_axis_range[1, :], indexing='ij')
        # pca_x1, pca_y1 = np.meshgrid(plane_pca_space_lower_bound, plane_pca_space_upper_bound)
        plane_vertices_pca = np.stack([pca_x1.flatten(), pca_y1.flatten()], axis=1)
        # map back to laboratory coordinate system
        v0 = np.matmul(plane_vertices_pca[0, :], pca.components_) + pca.mean_  # shape(3,)
        v1 = np.matmul(plane_vertices_pca[1, :], pca.components_) + pca.mean_  # shape(3,)
        v2 = np.matmul(plane_vertices_pca[2, :], pca.components_) + pca.mean_  # shape(3,)
        v3 = np.matmul(plane_vertices_pca[3, :], pca.components_) + pca.mean_  # shape(3,)

        plane_mesh = trimesh.Trimesh(vertices=[v0, v1, v2, v3], faces=[[0, 1, 2], [3, 2, 1]])
        plane_mesh.visual.vertex_colors = [0, 255, 0, 50]
        # plane_mesh.visual.face_colors = [0, 255, 0, 50]
        # plane_mesh.show()
        # trimesh.visual.color.ColorVisuals(mesh=plane_mesh, face_colors=None, vertex_colors=None)
        mesh.add_geometry(plane_mesh)
    # all_geometry_list = list(mesh.geometry.items())
    # mesh.delete_geometry(all_geometry_list[0][0])
    # mesh.show()
    mesh.export(output_mesh_file, include_normals=False, include_color=True)


def dump_mesh_with_axis(mesh_file_path, axis):
    pass

# def visualize_mesh_with_plane(mesh_file_path, plane_a, plane_b, plane_c, plane_d):
#     mesh = pv.read("mesh_file_path")
#     plotter = pv.Plotter(off_screen=True)
#     plotter.add_mesh(mesh, color='tan')
#
#
# def visualize_mesh_with_axis(mesh_file, axis_vector):
#     pass
