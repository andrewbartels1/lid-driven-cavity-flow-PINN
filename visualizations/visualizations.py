
from lid_driven_cavity_flow_pinn.load_data import prepare_data, LiddedDataset
from lid_driven_cavity_flow_pinn.models import  BoxFlowNet
from lid_driven_cavity_flow_pinn.train import train_model
import torch
from torch import nn
from lid_driven_cavity_flow_pinn.utils import generate_csv_catalog, load_model, navier_calc, navier_mse, read_datafile
import pandas as pd
import sys
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np


catalog_path = "../data/catalog.csv"

# =============================================================================
# Plot some predictions (the money plot, or not?)
# =============================================================================
#Create initial contour plot
dataCatalog = pd.read_csv(catalog_path)

# get a lower Re... let's say 400?
lower_Re = dataCatalog.iloc[1256]

# get a higher Re... let's say 1500?
higher_Re = dataCatalog.iloc[4000]


# =============================================================================
# Manually for the lower Re
# =============================================================================
#Create arrays and mesh
length = lower_Re['Re']/100
breadth = lower_Re['Re']/100
colpts = lower_Re['xsize']
rowpts = lower_Re['ysize']
x = np.linspace(0, length, colpts)
y = np.linspace(0, breadth, rowpts)
[X, Y] = np.meshgrid(x, y)

#Determine indexing for streamplot
index_cut_x = int(colpts/10)
index_cut_y = int(rowpts/10)

# read in data file
p_p, u_p, v_p, time, Re = read_datafile(lower_Re['filepath'])  # w.r.t. src/

#Create blank figure
fig = plt.figure(figsize=(8, 8))
ax = plt.axes()
ax.set_xlim([0, length])
ax.set_ylim([0, breadth])
ax.set_xlabel("$x$", fontsize=20)
ax.set_ylabel("$y$", fontsize=20)
# ax.set_title(f"Pressure and steamline ($\psi$) at Re={lower_Re['Re']}", fontsize=25)
cont = ax.contourf(X, Y, p_p, 50, cmap=plt.get_cmap('jet'))
stream = ax.streamplot(X[::index_cut_y, ::index_cut_x], Y[::index_cut_y, ::index_cut_x],
                       u_p[::index_cut_y, ::index_cut_x], v_p[::index_cut_y, ::index_cut_x], density=2, color="k")
fig.colorbar(cont)
fig.tight_layout()

# ax.grid()
# ax.axis('equal')
ax.legend()

fig.savefig('../report/images/Re400FlowPy_pres_stream.pdf',bbox_inches='tight')

#Create blank figure
fig = plt.figure(figsize=(8, 8))
ax = plt.axes()
ax.set_xlim([0, length])
ax.set_ylim([0, breadth])
ax.set_xlabel("$x$", fontsize=20)
ax.set_ylabel("$y$", fontsize=20)
# ax.set_title(f"U (horizontal) velocity at Re={lower_Re['Re']}", fontsize=25)
cont = ax.contourf(X, Y, u_p, 50, cmap=plt.get_cmap('jet'))
cont.cmap = 'jet'
fig.colorbar(cont)
fig.tight_layout()
ax.legend()

fig.savefig('../report/images/Re400FlowPy_u.pdf',bbox_inches='tight')

#Create blank figure
fig = plt.figure(figsize=(8, 8))
ax = plt.axes()
ax.set_xlim([0, length])
ax.set_ylim([0, breadth])
ax.set_xlabel("$x$", fontsize=20)
ax.set_ylabel("$y$", fontsize=20)
# ax.set_title(f"V (vertical) velocity at Re={lower_Re['Re']}", fontsize=25)
cont = ax.contourf(X, Y, v_p,50, cmap=plt.get_cmap('jet'))
fig.colorbar(cont)
fig.tight_layout()

# ax.grid()
# ax.axis('equal')
ax.legend()

fig.savefig('../report/images/Re400FlowPy_pres_v.pdf',bbox_inches='tight')

# =============================================================================
# BoxFlowNet visualization for lower Re
# =============================================================================
# use the gridded x and y so it's a flattened 151 x 151
# the _m denotes it's going into the model
x_m, y_m, time_m, Re_m = torch.Tensor(X.flatten()).requires_grad_(), \
                         torch.Tensor(Y.flatten()).requires_grad_(), \
                             torch.Tensor(time.flatten()).requires_grad_(), \
                         torch.Tensor([Re]*len(time.flatten())).requires_grad_()
                         
# put into model --> a LOT of this should be cleaned up!
# u_pred_lower, v_pred_lower, pressure_pred_lower, f_u_pred, f_v_pred = navier_calc(
#     x_m[None, :], y_m[None, :], time_m[None, :], BoxFlowNet_model_run.lambda1, BoxFlowNet_model_run.lambda2, device, Re_m[None, :], BoxFlowNet_model_run,
#     layers)


#Create blank figure
# fig = plt.figure(figsize=(8, 8))
# ax = plt.axes()
# ax.set_xlim([0, length])
# ax.set_ylim([0, breadth])
# ax.set_xlabel("$x$", fontsize=20)
# ax.set_ylabel("$y$", fontsize=20)
# # ax.set_title(f"V (vertical) velocity at Re={lower_Re['Re']}", fontsize=25)
# cont = ax.contourf(X, Y, u_pred_lower.detach().numpy().reshape(rowpts, colpts), 50, cmap=plt.get_cmap('jet'))
# fig.colorbar(cont)
# fig.tight_layout()

# ax.grid()
# ax.axis('equal')
# ax.legend()

# fig.savefig('../report/images/Re400BoxFlowNet_pred_u.pdf',bbox_inches='tight')

# #Create blank figure
# fig = plt.figure(figsize=(8, 8))
# ax = plt.axes()
# ax.set_xlim([0, length])
# ax.set_ylim([0, breadth])
# ax.set_xlabel("$x$", fontsize=20)
# ax.set_ylabel("$y$", fontsize=20)
# ax.set_title(f"u - u model prediction", fontsize=20)
# diff =u_p - u_pred_lower.detach().numpy().reshape(rowpts, colpts)
# cont = ax.contourf(X, Y, diff, 50, cmap=plt.get_cmap('jet'))
# fig.colorbar(cont)
# fig.tight_layout()

# # ax.grid()
# # ax.axis('equal')
# ax.legend()

# fig.savefig('../report/images/Re400BoxFlowNet_diff_u.pdf',bbox_inches='tight')

# fig = plt.figure(figsize=(8, 8))
# ax = plt.axes()
# ax.set_xlim([0, length])
# ax.set_ylim([0, breadth])
# ax.set_xlabel("$x$", fontsize=20)
# ax.set_ylabel("$y$", fontsize=20)
# ax.set_title(f"model produced pressure prediction", fontsize=20)
# cont = ax.contourf(X, Y, pressure_pred_lower.detach().numpy().reshape(rowpts, colpts), 50, cmap=plt.get_cmap('jet'))
# fig.colorbar(cont)
# fig.tight_layout()

# # ax.grid()
# # ax.axis('equal')
# ax.legend()

# fig.savefig('../report/images/Re400BoxFlowNet_pres.pdf',bbox_inches='tight')

# fig = plt.figure(figsize=(8, 8))
# ax = plt.axes()
# ax.set_xlim([0, length])
# ax.set_ylim([0, breadth])
# ax.set_xlabel("$x$", fontsize=20)
# ax.set_ylabel("$y$", fontsize=20)
# ax.set_title(f"model produced v velocity prediction", fontsize=20)
# cont = ax.contourf(X, Y, v_pred_lower.detach().numpy().reshape(rowpts, colpts), 50, cmap=plt.get_cmap('jet'))
# fig.colorbar(cont)
# fig.tight_layout()

# # ax.grid()
# # ax.axis('equal')
# ax.legend()

# fig.savefig('../report/images/Re400BoxFlowNet_v.pdf',bbox_inches='tight')



# =============================================================================
# Manually for the higher Re
# =============================================================================
# read in data file
#Create arrays and mesh
length = higher_Re['Re']/100
breadth = higher_Re['Re']/100
colpts = higher_Re['xsize']
rowpts = higher_Re['ysize']
x = np.linspace(0, length, colpts)
y = np.linspace(0, breadth, rowpts)
[X, Y] = np.meshgrid(x, y)

#Determine indexing for streamplot
index_cut_x = int(colpts/10)
index_cut_y = int(rowpts/10)
p_p, u_p, v_p, time, Re = read_datafile(higher_Re['filepath'])  # w.r.t. src/

#Create blank figure
fig = plt.figure(figsize=(8, 8))
ax = plt.axes()
ax.set_xlim([0, length])
ax.set_ylim([0, breadth])
ax.set_xlabel("$x$", fontsize=20)
ax.set_ylabel("$y$", fontsize=20)
# ax.set_title(f"Pressure and steamline ($\psi$) at Re={higher_Re['Re']}", fontsize=25)
cont = ax.contourf(X, Y, p_p, 50, cmap=plt.get_cmap('jet'))
stream = ax.streamplot(X[::index_cut_y, ::index_cut_x], Y[::index_cut_y, ::index_cut_x],
                       u_p[::index_cut_y, ::index_cut_x], v_p[::index_cut_y, ::index_cut_x], density=2, color="k")
fig.colorbar(cont)
fig.tight_layout()

# ax.grid()
# ax.axis('equal')
ax.legend()

fig.savefig('../report/images/Re1500FlowPy_pres_stream.pdf',bbox_inches='tight')

#Create blank figure
fig = plt.figure(figsize=(8, 8))
ax = plt.axes()
ax.set_xlim([0, length])
ax.set_ylim([0, breadth])
ax.set_xlabel("$x$", fontsize=20)
ax.set_ylabel("$y$", fontsize=20)
# ax.set_title(f"U (horizontal) velocity at Re={higher_Re['Re']}", fontsize=25)
cont = ax.contourf(X, Y, u_p, 50, cmap=plt.get_cmap('jet'))
cont.cmap = 'jet'
fig.colorbar(cont)
fig.tight_layout()
ax.legend()

fig.savefig('../report/images/Re1500FlowPy_u.pdf',bbox_inches='tight')

#Create blank figure
fig = plt.figure(figsize=(8, 8))
ax = plt.axes()
ax.set_xlim([0, length])
ax.set_ylim([0, breadth])
ax.set_xlabel("$x$", fontsize=20)
ax.set_ylabel("$y$", fontsize=20)
# ax.set_title(f"V (vertical) velocity at Re={higher_Re['Re']}", fontsize=25)
cont = ax.contourf(X, Y, v_p,50, cmap=plt.get_cmap('jet'))
fig.colorbar(cont)
fig.tight_layout()

# ax.grid()
# ax.axis('equal')
ax.legend()

fig.savefig('../report/images/Re1500FlowPy_pres_v.pdf',bbox_inches='tight')


# =============================================================================
# BoxFlowNet visualization for lower Re
# =============================================================================
# use the gridded x and y so it's a flattened 151 x 151
# the _m denotes it's going into the model
# x_m, y_m, time_m, Re_m = torch.Tensor(X.flatten()).requires_grad_(), \
#                          torch.Tensor(Y.flatten()).requires_grad_(), \
#                              torch.Tensor(time.flatten()).requires_grad_(), \
#                          torch.Tensor([Re]*len(time.flatten())).requires_grad_()
                         
# # put into model --> a LOT of this should be cleaned up!
# u_pred_lower, v_pred_lower, pressure_pred_lower, f_u_pred, f_v_pred = navier_calc(
#     x_m[None, :], y_m[None, :], time_m[None, :], BoxFlowNet_model_run.lambda1, BoxFlowNet_model_run.lambda2, device, Re_m[None, :], BoxFlowNet_model_run,
#     layers)


#Create blank figure
fig = plt.figure(figsize=(8, 8))
ax = plt.axes()
ax.set_xlim([0, length])
ax.set_ylim([0, breadth])
ax.set_xlabel("$x$", fontsize=20)
ax.set_ylabel("$y$", fontsize=20)
# ax.set_title(f"V (vertical) velocity at Re={lower_Re['Re']}", fontsize=25)
cont = ax.contourf(X, Y, u_pred_lower.detach().numpy().reshape(rowpts, colpts), 50, cmap=plt.get_cmap('jet'))
fig.colorbar(cont)
fig.tight_layout()

# ax.grid()
# ax.axis('equal')
ax.legend()

fig.savefig('../report/images/Re1500BoxFlowNet_pred_u.pdf',bbox_inches='tight')

#Create blank figure
fig = plt.figure(figsize=(8, 8))
ax = plt.axes()
ax.set_xlim([0, length])
ax.set_ylim([0, breadth])
ax.set_xlabel("$x$", fontsize=20)
ax.set_ylabel("$y$", fontsize=20)
ax.set_title(f"u - u model prediction", fontsize=20)
diff =u_p - u_pred_lower.detach().numpy().reshape(rowpts, colpts)
cont = ax.contourf(X, Y, diff, 50, cmap=plt.get_cmap('jet'))
fig.colorbar(cont)
fig.tight_layout()

# ax.grid()
# ax.axis('equal')
ax.legend()

fig.savefig('../report/images/Re1500BoxFlowNet_diff_u.pdf',bbox_inches='tight')

fig = plt.figure(figsize=(8, 8))
ax = plt.axes()
ax.set_xlim([0, length])
ax.set_ylim([0, breadth])
ax.set_xlabel("$x$", fontsize=20)
ax.set_ylabel("$y$", fontsize=20)
ax.set_title(f"model produced pressure prediction", fontsize=20)
cont = ax.contourf(X, Y, pressure_pred_lower.detach().numpy().reshape(rowpts, colpts), 50, cmap=plt.get_cmap('jet'))
fig.colorbar(cont)
fig.tight_layout()

# ax.grid()
# ax.axis('equal')
ax.legend()

fig.savefig('../report/images/Re1500BoxFlowNet_pres.pdf',bbox_inches='tight')

fig = plt.figure(figsize=(8, 8))
ax = plt.axes()
ax.set_xlim([0, length])
ax.set_ylim([0, breadth])
ax.set_xlabel("$x$", fontsize=20)
ax.set_ylabel("$y$", fontsize=20)
ax.set_title(f"model produced v velocity prediction", fontsize=20)
cont = ax.contourf(X, Y, v_pred_lower.detach().numpy().reshape(rowpts, colpts), 50, cmap=plt.get_cmap('jet'))
fig.colorbar(cont)
fig.tight_layout()

# ax.grid()
# ax.axis('equal')
ax.legend()

fig.savefig('../report/images/Re1500BoxFlowNet_v.pdf',bbox_inches='tight')











