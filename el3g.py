#!/usr/bin/env python2

import itertools
import numpy as np
import math
import h5py

Lx = 15.0
Ly = 10.0
Lz = 5.0
kappa = 1.0
rho = 1.0
G = 1.0

nx = 150
ny = 100
nz = 50
dx = Lx/nx
dy = Ly/ny
dz = Lz/nz
nt = 5
dt = min(dx,dy,dz)/math.sqrt(kappa/rho)/2.75

OVERLENGTH_X = 1
OVERLENGTH_Y = 1
OVERLENGTH_Z = 1
def full_bounds():
    return itertools.product(range(0, nx+OVERLENGTH_X),
                             range(0, ny+OVERLENGTH_Y),
                             range(0, nz+OVERLENGTH_Z))

# quantities stored on:
# centers: x, y, z, P, Txx, Tyy, Tzz
# x-faces (faces perpendicular to x axis): Vx
# x-edges (edges parallel to x axis): Tyz

# boundary conditions:
# Vx = 0 on xmin,xmax
# Vy = 0 on ymin,ymax
# Vz = 0 on zmin,zmax
# Txy = 0 on xmin,xmax,ymin,ymax
# Txz = 0 on xmin,xmax,zmin,zmax
# Tyz = 0 on ymin,ymax,zmin,zmax

x   = np.zeros((nx,  ny,  nz  ))
y   = np.zeros((nx,  ny,  nz  ))
z   = np.zeros((nx,  ny,  nz  ))
P   = np.zeros((nx,  ny,  nz  ))
Vx  = np.zeros((nx+1,ny,  nz  ))
Vy  = np.zeros((nx,  ny+1,nz  ))
Vz  = np.zeros((nx,  ny,  nz+1))
Txx = np.zeros((nx,  ny,  nz  ))
Tyy = np.zeros((nx,  ny,  nz  ))
Tzz = np.zeros((nx,  ny,  nz  ))
Txy = np.zeros((nx+1,ny+1,nz  ))
Txz = np.zeros((nx+1,ny,  nz+1))
Tyz = np.zeros((nx,  ny+1,nz+1))

for (i,j,k) in full_bounds():
    if 0 <= i and i < nx and 0 <= j and j < ny and 0 <= k and k < nz:
        x[i,j,k] = (-Lx+dx)/2 + i*dx
for (i,j,k) in full_bounds():
    if 0 <= i and i < nx and 0 <= j and j < ny and 0 <= k and k < nz:
        y[i,j,k] = (-Ly+dy)/2 + j*dy
for (i,j,k) in full_bounds():
    if 0 <= i and i < nx and 0 <= j and j < ny and 0 <= k and k < nz:
        z[i,j,k] = (-Lz+dz)/2 + k*dz
for (i,j,k) in full_bounds():
    if 0 <= i and i < nx and 0 <= j and j < ny and 0 <= k and k < nz:
        P[i,j,k] = math.exp(-(x[i,j,k]**2+y[i,j,k]**2+z[i,j,k]**2))
# stresses are already correct for the next iteration

t = -1
while True:
    t += 1
    hdf_out = h5py.File('%d.hdf' % t, 'w')
    hdf_out['pressure'] = P
    hdf_out.close()
    if t >= nt:
        break

    for (i,j,k) in full_bounds():
        if i >= nx or j >= ny or k >= nz:
            continue
        P_ijk = P[i,j,k]
        Txy_ijk = Txy[i,j,k]
        Txz_ijk = Txz[i,j,k]
        Tyz_ijk = Tyz[i,j,k]
        if 1 <= i:
            Vx[i,j,k] += dt/rho*( - (P_ijk-P[i-1,j,k])/dx + (Txx[i,j,k]-Txx[i-1,j,k])/dx + (Txy[i,j+1,k]-Txy_ijk)/dy + (Txz[i,j,k+1]-Txz_ijk)/dz )
        if 1 <= j:
            Vy[i,j,k] += dt/rho*( - (P_ijk-P[i,j-1,k])/dy + (Tyy[i,j,k]-Tyy[i,j-1,k])/dy + (Txy[i+1,j,k]-Txy_ijk)/dx + (Tyz[i,j,k+1]-Tyz_ijk)/dz )
        if 1 <= k:
            Vz[i,j,k] += dt/rho*( - (P_ijk-P[i,j,k-1])/dz + (Tzz[i,j,k]-Tzz[i,j,k-1])/dz + (Txz[i+1,j,k]-Txz_ijk)/dx + (Tyz[i,j+1,k]-Tyz_ijk)/dy )
    for (i,j,k) in full_bounds():
        if i >= nx or j >= ny or k >= nz:
            continue
        Vx_ijk = Vx[i,j,k]
        Vy_ijk = Vy[i,j,k]
        Vz_ijk = Vz[i,j,k]
        Dx_Vx_ijk = ( Vx[i+1,j,k] - Vx_ijk ) / dx
        Dy_Vy_ijk = ( Vy[i,j+1,k] - Vy_ijk ) / dy
        Dz_Vz_ijk = ( Vz[i,j,k+1] - Vz_ijk ) / dz
        P[i,j,k] -= dt*kappa*( Dx_Vx_ijk + Dy_Vy_ijk + Dz_Vz_ijk )
        Txx[i,j,k] += dt*2.0*G*( + 2.0/3.0*Dx_Vx_ijk - 1.0/3.0*Dy_Vy_ijk - 1.0/3.0*Dz_Vz_ijk )
        Tyy[i,j,k] += dt*2.0*G*( - 1.0/3.0*Dx_Vx_ijk + 2.0/3.0*Dy_Vy_ijk - 1.0/3.0*Dz_Vz_ijk )
        Tzz[i,j,k] += dt*2.0*G*( - 1.0/3.0*Dx_Vx_ijk - 1.0/3.0*Dy_Vy_ijk + 2.0/3.0*Dz_Vz_ijk )
        if 1 <= i and 1 <= j:
            Txy[i,j,k] += dt*G*( (Vx_ijk-Vx[i,j-1,k])/dy + (Vy_ijk-Vy[i-1,j,k])/dx )
        if 1 <= i and 1 <= k:
            Txz[i,j,k] += dt*G*( (Vx_ijk-Vx[i,j,k-1])/dz + (Vz_ijk-Vz[i-1,j,k])/dx )
        if 1 <= j and 1 <= k:
            Tyz[i,j,k] += dt*G*( (Vy_ijk-Vy[i,j,k-1])/dz + (Vz_ijk-Vz[i,j-1,k])/dy )
