import numpy as np
from gym.envs.robotics import rotations as r_tool
from numpy.linalg import inv
from ipdb import set_trace
import math

BOOL_SYM = True
Z_ZHETA = math.pi/5
SYM_PLANE_Y = 0.48 * 2
class mirror_learning:
    def __init__(self):
        pass

    def y_mirror(self,param):
        return self.sym_plane_compute(param,'y_axis','y_mirror')

    def x_mirror():
        self.sym_plane_compute(param,'x_axis','y_mirror')


    def kaleidoscope_robot(self, param, sym_axis = 'y_axis', sym_method = 'y_mirror'):
        if sym_axis == 'y_axis':
            # in linear variable, plus i; in angular variable, minus i
            i = 0
        elif sym_axis == 'x_axis':
            i = -1

        if sym_method == 'y_mirror':
            SYM_PLANE = SYM_PLANE_Y
        elif sym_method == 'x_mirror':
            SYM_PLANE = SYM_PLANE_X

        # compute the rotation transformation & its inverse.
        theta = np.array([0,0,Z_ZHETA])
        rot_z_theta = r_tool.euler2mat(theta)
        inv_rot_z_theta = inv(rot_z_theta)

        # Determine the input is which element
        param_len = len(param[0])
        #goal & achieved goal or action
        if param_len == 3 or param_len == 4:    
            v_l_a = param[0][0:3]
            s_v_l_a = self.linear_vector_symmetric_with_rot_plane(True, v_l_a, rot_z_theta, inv_rot_z_theta, i, SYM_PLANE)
            param[0][0:3] =  s_v_l_a

        # elif param_len == 4:  #action

        elif param_len == 10:     # observation without object
            # grip pos
            v_l_a = param[0][0:3]
            s_v_l_a = self.linear_vector_symmetric_with_rot_plane(True, v_l_a, rot_z_theta, inv_rot_z_theta, i, SYM_PLANE)
            param[0][0:3] =  s_v_l_a
            # grip vel
            v_l_a = param[0][3:6]
            s_v_l_a = self.linear_vector_symmetric_with_rot_plane(False, v_l_a, rot_z_theta, inv_rot_z_theta, i, SYM_PLANE)
            param[0][3:6] =  s_v_l_a
            
        elif param_len == 25:     # observation with object
            # sym_grip_pos
            v_l_a = param[0][0:3]
            s_v_l_a = self.linear_vector_symmetric_with_rot_plane(True, v_l_a, rot_z_theta, inv_rot_z_theta, i, SYM_PLANE)
            param[0][0:3] =  s_v_l_a

            # sym_obj_pos
            v_l_a = param[0][3:6]
            s_v_l_a = self.linear_vector_symmetric_with_rot_plane(True, v_l_a, rot_z_theta, inv_rot_z_theta, i, SYM_PLANE)
            param[0][3:6] =  s_v_l_a
            

            # sym_obj_rel_pos
            param[0][6] = param[0][3]-param[0][0]
            param[0][7] = param[0][4]-param[0][1]
            param[0][8] = param[0][5]-param[0][2]

            # sym_obj_rot_euler
            theta_a = param[0][11:14]
            param[0][11:14] = self.orientation_mat_symmetric_with_rot_plane(theta_a, rot_z_theta, inv_rot_z_theta, i)
            
            # get the obj real velp 
            v_l_a = param[0][14:17]+param[0][20:23]
            # get the sym obj real velp
            s_v_l_a = self.linear_vector_symmetric_with_rot_plane(False, v_l_a, rot_z_theta, inv_rot_z_theta, i, SYM_PLANE)
            param[0][14:17] =  s_v_l_a
            # get the sym grip real velp
            v_l_a = param[0][20:23]
            s_v_l_a = self.linear_vector_symmetric_with_rot_plane(False, v_l_a, rot_z_theta, inv_rot_z_theta, i, SYM_PLANE)
            param[0][20:23] =  s_v_l_a
            # get the sym obj relative velp 
            param[0][14:17] -= param[0][20:23]

            # sym_obj_velr
            theta_a = param[0][17:20]
            param[0][17:20] = self.orientation_mat_symmetric_with_rot_plane(theta_a, rot_z_theta, inv_rot_z_theta, i)
        

        return param.copy()

    def linear_vector_symmetric_with_rot_plane(self,if_pos, v_l_a, rot_z_theta, inv_rot_z_theta, i, SYM_PLANE):
        # Point 'a' position = v_l_a

        v_l_a_hat = np.matmul(v_l_a,rot_z_theta)

        if if_pos == True:
            v_l_a_hat[1+i] = SYM_PLANE-v_l_a_hat[1+i]
        else:
            v_l_a_hat[1+i] = -v_l_a_hat[1+i]

        s_v_l_a =  np.matmul(v_l_a_hat,inv_rot_z_theta)
        return s_v_l_a.copy()

    def orientation_mat_symmetric_with_rot_plane(self, theta_a, rot_z_theta, inv_rot_z_theta, i):
        # Point 'a' orientation euler angle = theta_a
        # set_trace()
        # euler to rot matrix for transform
        v_r_a = r_tool.euler2mat(theta_a)
        # transform to the O cordinate from S cordinate
        v_r_a_hat = np.matmul(v_r_a,inv_rot_z_theta)

        # Rot matrix to euler for sym
        v_r_a_hat = r_tool.mat2euler(v_r_a_hat)

        # Sym on transformed S's xoz plane (y axis)
        v_r_a_hat[0-i] = -v_r_a_hat[0-i]
        v_r_a_hat[2] = -v_r_a_hat[2]

        # euler to rot matrix for transform 
        v_r_a_hat = r_tool.euler2mat(v_r_a_hat)
        s_v_r_a = np.matmul(v_r_a_hat,rot_z_theta)

        # Rot matrix to euler for return
        s_v_r_a = r_tool.mat2euler(s_v_r_a)
        return s_v_r_a.copy()

    def kaleidoscope_obj():
        pass
    def mirage():
        pass


    def sym_plane_compute(self,param,sym_axis,sym_method):
        # This function is for the vanilla mirror. (x&y mirror)
        if sym_axis == 'y_axis':
            # in linear variable, plus i; in angular variable, minus i
            i = 0
        elif sym_axis == 'x_axis':
            i = -1


        if sym_method == 'y_mirror':
            SYM_PLANE = SYM_PLANE_Y
        elif sym_method == 'x_mirror':
            SYM_PLANE = SYM_PLANE_X
        # elif sym_method == 'kaleidoscope_robot':
        #     SYM_PLANE = SYM_PLANE_Y


        param_len = len(param[0])
        if param_len == 3 or param_len == 4:    #goal & achieved goal or action
            param[0][1+i] = SYM_PLANE - param[0][1+i]

        # elif param_len == 4:  #action
        #     param[0][1+i] = SYM_PLANE - param[0][1+i]

        elif param_len == 10:     # observation without object
            param[0][1+i] = SYM_PLANE - param[0][1+i]
            # vel do not need SYM_PLANE
            param[0][4+i] = -param[0][4+i]
            
        elif param_len == 25:     # observation with object

            # sym_grip_pos
            param[0][1+i] = SYM_PLANE - param[0][1+i]
            # sym_obj_pos
            param[0][4+i] = SYM_PLANE - param[0][4+i]

            # sym_obj_rel_pos
            param[0][6] = param[0][3]-param[0][0]
            param[0][7] = param[0][4]-param[0][1]
            param[0][8] = param[0][5]-param[0][2]

            # sym_obj_rot_euler
            param[0][11-i] = -param[0][11-i]
            param[0][13] = -param[0][13]

            # get the sym_obj_rel_velp & sym_grip_velp
            # no need to transform back to original pose, it can be directly compute the relative pose.
            param[0][15+i] = -param[0][15+i]
            param[0][21+i] = -param[0][21+i]
            
            # sym_obj_velr
            param[0][17-i] = -param[0][17-i]
            param[0][19] = -param[0][19]
        return param.copy()

