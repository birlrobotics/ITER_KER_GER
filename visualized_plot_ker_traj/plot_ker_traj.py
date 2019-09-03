import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import rotations
import math
from ipdb import set_trace
# Sorting the directory
def sorting_dir(load_data_folder):
    sorting_folder_list = os.listdir(load_data_folder)
    sorted_folder_list = []
    for i in range(len(sorting_folder_list)+1):
        for sorting_folder in sorting_folder_list:
            if i == int(sorting_folder[-5:-4]):
                sorted_folder_list.append(sorting_folder)
    return sorted_folder_list

# Set up the table 
cube_x =[1,2,3,4,5,6,7,8,9,10]
cube_y =[5,6,2,3,13,4,1,2,4,8]
cube_z =[2,3,3,3,5,7,9,11,9,10]
def get_cube(cube_x,cube_y,cube_z):   
    phi = np.arange(1,10,2)*np.pi/4
    Phi, Theta = np.meshgrid(phi, phi)

    cube_x = np.cos(Phi)*np.sin(Theta)
    cube_y = np.sin(Phi)*np.sin(Theta)
    cube_z = np.cos(Theta)/np.sqrt(2)
    return cube_x,cube_y,cube_z
x_length = 0.5
y_length = 0.7
z_length = 0.4
cube_x,cube_y,cube_z = get_cube(cube_x,cube_y,cube_z)

# Extract the directories.
load_traj_folder = '/home/bourne/data_plot/visualized_plot_ker_traj/all_n_rsym_trajs'
# Sorting the files in the list: 0,1,2,3..., for plotting the label legends in right orders.
traj_sorted_folder_list = sorting_dir(load_traj_folder)

ds = []
for file in traj_sorted_folder_list:
    file = os.path.join(load_traj_folder,file)
    d = np.load(file)
    ds.append(d)




# Begin===================Extract the trajectories from observation from transition.
all_xs =[]
all_ys =[]
all_zs =[]
for d in ds:
    xs =[]
    ys =[]
    zs = []
    for transitions in d:
        x =[]
        y =[]
        z = []
        for ob in transitions[0]:
            x.append(ob[0][0])
            y.append(ob[0][1])
            z.append(ob[0][2])
        x = np.array(x)
        y = np.array(y)
        z = np.array(z)
        xs.append(x)
        ys.append(y)
        zs.append(z)
    all_xs.append(xs)
    all_ys.append(ys)
    all_zs.append(zs)
# End===================Extract the trajectories from observation from transition.


# Begin===================Extract the thetas.
load_theta_folder = '/home/bourne/data_plot/visualized_plot_ker_traj/all_n_rsym_thetas'
# Sorting the files in the list: 0,1,2,3..., for plotting the label legends in right orders.
thetas_sorted_folder_list = sorting_dir(load_theta_folder)

all_thetas = []
all_thetas.append('None')
for file in thetas_sorted_folder_list:
    file = os.path.join(load_theta_folder,file)
    thetas_in_each_file = np.load(file)
    all_thetas.append(thetas_in_each_file)

# End===================Extract the thetas.

# Prepare the color for plot
prepared_color = ['c','m','gold','g','b']
plane_colors = prepared_color.copy()
original_colors = []
# The original traj is red
original_colors.append('r')
for i in range(4):
    n = 2**i
    pop_color = prepared_color.pop()
    for _ in range(n):
        original_colors.append(pop_color)
original_colors.reverse()


    



# plot each n_rsym trajectories
for (xs,ys,zs,n_rsym_thetas) in zip(all_xs,all_ys,all_zs,all_thetas):
    
    colour = original_colors.copy()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    n_rsym_thetas_list = list(n_rsym_thetas)
    n_rsym_thetas_list.reverse()
    plane_colours = plane_colors.copy()
    for i,(x,y,z) in enumerate(zip(xs,ys,zs)):
        
        ax.plot(x, y, z, c=colour.pop())
        if i == 1 or i == 3 or i == 7 or i == 15:
            #Begin===================== show the kaleidoscope plane
            # Given the normal vector and the point in that plane
            point  = np.array([0, 0,0 ]) + np.array([0.695, 0.75, 0.4])
            normal = np.array([0, 1, 0])  
            # for i in range(360):
            theta = n_rsym_thetas_list.pop()
            rot_vec = theta
            
            rot_mat = rotations.euler2mat(rot_vec)
            normal = np.dot(rot_mat,normal)
            # a plane is a*x+b*y+c*z+d=0
            # [a,b,c] is the normal. Thus, we have to calculate
            # d and we're set
            d = -point.dot(normal)
            # create x,z , the plane range
            xx, zz = np.meshgrid(range(695,1800), range(0,8))
            xx = xx * 0.001
            zz = zz * 0.1
            
            # calculate corresponding z
            y = (-normal[0] * xx - normal[2] * zz - d) * 1. /normal[1]
            # plot the surface
            plane_color = plane_colours.pop()
            ax.plot_surface(xx, y, zz, alpha=0.2, color = plane_color)
            #End===================== show the kaleidoscope plane
    # robot base poiont
    ax.scatter(0.695, 0.75, 0.4, c='r')

    # plot the table
    ax.plot_surface(cube_x*x_length+1.3, cube_y*y_length+0.75, cube_z*z_length+0.2,alpha = 0.2, color = 'white')




    # setup the axis inf
    ax.set_xlim(0.6,1.5)
    ax.set_ylim(0.4,1)
    ax.set_zlim(0,0.8)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
plt.show()

set_trace()

