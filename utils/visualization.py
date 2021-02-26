import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
# import pyvista as pv
import trimesh


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
    for i in range(planes.shape[0]):
        a = planes[i, 0]
        b = planes[i, 1]
        c = planes[i, 2]
        d = planes[i, 3]
        v0 = [-0.5, -0.5, (-d+0.5*a+0.5*b)/c]
        v1 = [-0.5, +0.5, (-d+0.5*a-0.5*b)/c]
        v2 = [+0.5, -0.5, (-d-0.5*a+0.5*b)/c]
        v3 = [+0.5, +0.5, (-d-0.5*a-0.5*b)/c]
        try:
            plane_mesh = trimesh.Trimesh(vertices=[v0, v1, v2, v3], faces=[[0, 1, 2], [3, 2, 1]])
        except:
            import pdb
            pdb.set_trace()
        mesh.add_geometry(plane_mesh)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    mesh.export(output_mesh_file)


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
