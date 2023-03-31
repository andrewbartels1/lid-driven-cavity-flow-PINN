# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 22:33:14 2018

@author: Gaurav
"""

import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation


#### Simulation inputs
# length, breadth = 1,1
# length, breadth = 2,2
# length, breadth = 3,3
# length, breadth = 4,4
# length, breadth = 5,5
# length, breadth = 6,6
# length, breadth = 7,7
# length, breadth = 8,8
# length, breadth = 9,9
# length, breadth = 10, 10
# length, breadth = 11,11
# length, breadth = 12,12
# length, breadth = 13,13
# length, breadth = 14,14
# length, breadth = 19,19
starting_Reynolds = 3 # this inviscid code doesn't converge well at low Re
ending_Reynolds = 25
# wrapped in loop
# CAUTION: This takes 20-30 minutes to run in it's current configuration
# you can put the resolution down, but this can cause divergence issues!
for i in range(starting_Reynolds,ending_Reynolds):
    length, breadth = i,i
    colpts = 151  # keep ODD MUST match whatever resolution `FlowPy_Input.py` was run with!
    rowpts = 151  # keep ODD MUST match whatever resolution `FlowPy_Input.py` was run with!
    time = 1500
    ###########################DEFINE PHYSICAL PARAMETERS##########################
    rho = 1
    mu = 0.01
    ##########################DEFINE INITIAL MOMENTUM PARAMETERS###################
    u_in = 1
    v_wall = 0
    p_out = 0
    Re = rho*u_in*length/mu

    ###########################SPECIFY WHERE DATA LIVES##########################
    # run from Lid-Driven_Cavity-Flow/src
    dir_path = os.path.join("../../../", "data", "Re" + str(int(Re)))
    os.chdir(dir_path)

    #Define function to read datafile


    def read_datafile(iteration):
        filename = "PUV{0}.txt".format(iteration)
        filepath = os.path.join(filename)
        arr = np.loadtxt(filepath, delimiter="\t")
        rows, cols = arr.shape[0], arr.shape[1]-1
        p_p = np.zeros((rowpts, colpts))
        u_p = np.zeros((rowpts, colpts))
        v_p = np.zeros((rowpts, colpts))
        p_arr = arr[:, 0]
        u_arr = arr[:, 1]
        v_arr = arr[:, 2]
        
        timestep = arr[:, 3]
        
        p_p = p_arr.reshape((rowpts, colpts))
        u_p = u_arr.reshape((rowpts, colpts))
        v_p = v_arr.reshape((rowpts, colpts))
        

        return p_p, u_p, v_p 


    #Go to the Result directory

    #Go through files in the directory
    filenames = []
    iterations = []
    for root, dirs, files in os.walk("."):
        # print(root, dirs)
        for datafile in files:
            # print(datafile)
            if "PUV" in datafile:
                filenames.append(datafile)
                no_ext_file = datafile.replace(".txt", "").strip()
                iter_no = int(no_ext_file.split("V")[-1])
                iterations.append(iter_no)

    #Discern the final iteration and interval
    initial_iter = np.amin(iterations)
    final_iter = np.amax(iterations)
    inter = (final_iter - initial_iter)/(len(iterations)-1)
    number_of_frames = len(iterations)  # int(final_iter/inter)+1
    sorted_iterations = np.sort(iterations)

    #Create arrays and mesh
    x = np.linspace(0, length, colpts)
    y = np.linspace(0, breadth, rowpts)
    [X, Y] = np.meshgrid(x, y)

    #Determine indexing for streamplot
    index_cut_x = int(colpts/10)
    index_cut_y = int(rowpts/10)

    #Create blank figure
    fig = plt.figure(figsize=(16, 8))
    ax = plt.axes(xlim=(0, length), ylim=(0, breadth))

    #Create initial contour plot
    p_p, u_p, v_p = read_datafile(0)
    ax.set_xlim([0, length])
    ax.set_ylim([0, breadth])
    ax.set_xlabel("$x$", fontsize=12)
    ax.set_ylabel("$y$", fontsize=12)
    ax.set_title("Frame No: 0")
    cont = ax.contourf(X, Y, p_p)
    stream = ax.streamplot(X[::index_cut_y, ::index_cut_x], Y[::index_cut_y, ::index_cut_x],
                        u_p[::index_cut_y, ::index_cut_x], v_p[::index_cut_y, ::index_cut_x], color="k")
    fig.colorbar(cont)
    fig.tight_layout()


    def animate(i):
        sys.stdout.write("\rFrames remaining: {0:03d}".format(
            len(sorted_iterations)-i))
        sys.stdout.flush()
        iteration = sorted_iterations[i]
        p_p, u_p, v_p = read_datafile(iteration)
        ax.clear()
        ax.set_xlim([0, length])
        ax.set_ylim([0, breadth])
        ax.set_xlabel("$x$", fontsize=12)
        ax.set_ylabel("$y$", fontsize=12)
        ax.set_title("Frame No: {0}".format(i))
        cont = ax.contourf(X, Y, p_p)
        stream = ax.streamplot(X[::index_cut_y, ::index_cut_x], Y[::index_cut_y, ::index_cut_x],
                            u_p[::index_cut_y, ::index_cut_x], v_p[::index_cut_y, ::index_cut_x], color="k")
        return cont, stream


    print("######## Making FlowPy Animation ########")
    print("#########################################")
    anim = animation.FuncAnimation(
        fig, animate, frames=number_of_frames, interval=50, blit=False)
    movie_path = os.path.join("FluidFlowAnimation_Re{}.mp4".format(str(int(Re))))
    anim.save(r"{0}".format(movie_path))
    print("\nAnimation saved as FluidFlowAnimation.mp4 in Result")
    os.chdir("../../src/lid_driven_cavity_flow_pinn/FlowPy/")
