import numpy as np
from gym.envs.robotics import rotations as r_tool
from numpy.linalg import inv
from ipdb import set_trace
BOOL_SYM = True
SYM_PLANE_Y = 0.48 * 2
class mirror_learning:
    def __init__(self):
        pass

    def y_mirror(self,param):
        return self.sym_plane_compute(param,'y_axis','y_mirror')

    def x_mirror():
        self.sym_plane_compute(param,x_axis)


    def kaleidoscope_robot():
        self.sym_plane_compute(param,kaleidoscope_robot)

    def kaleidoscope_obj():
        pass
    def mirage():
        pass


    def sym_plane_compute(self,param,sym_axis,sym_method):
        if sym_axis == 'y_axis':
            # in linear variable, plus i; in angular variable, minus i
            i = 0
        elif sym_axis == 'x_axis':
            i = -1


        if sym_method == 'y_mirror':
            SYM_PLANE = SYM_PLANE_Y
        elif sym_method == 'x_mirror':
            SYM_PLANE = SYM_PLANE_X
        elif sym_method == 'kaleidoscope_robot':
            SYM_PLANE = SYM_PLANE_Y


        param_len = len(param[0])
        if param_len == 3:    #goal & achieved goal
            param[0][1+i] = SYM_PLANE - param[0][1+i]

        elif param_len == 4:  #action
            param[0][1+i] = SYM_PLANE - param[0][1+i]

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