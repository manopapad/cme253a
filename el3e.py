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
        if 1 <= i and i < nx and 0 <= j and j < ny and 0 <= k and k < nz:
            Vx[i,j,k] += dt/rho*( - (P[i,j,k]-P[i-1,j,k])/dx + (Txx[i,j,k]-Txx[i-1,j,k])/dx + (Txy[i,j+1,k]-Txy[i,j,k])/dy + (Txz[i,j,k+1]-Txz[i,j,k])/dz )
    for (i,j,k) in full_bounds():
        if 0 <= i and i < nx and 1 <= j and j < ny and 0 <= k and k < nz:
            Vy[i,j,k] += dt/rho*( - (P[i,j,k]-P[i,j-1,k])/dy + (Tyy[i,j,k]-Tyy[i,j-1,k])/dy + (Txy[i+1,j,k]-Txy[i,j,k])/dx + (Tyz[i,j,k+1]-Tyz[i,j,k])/dz )
    for (i,j,k) in full_bounds():
        if 0 <= i and i < nx and 0 <= j and j < ny and 1 <= k and k < nz:
            Vz[i,j,k] += dt/rho*( - (P[i,j,k]-P[i,j,k-1])/dz + (Tzz[i,j,k]-Tzz[i,j,k-1])/dz + (Txz[i+1,j,k]-Txz[i,j,k])/dx + (Tyz[i,j+1,k]-Tyz[i,j,k])/dy )
    for (i,j,k) in full_bounds():
        if 0 <= i and i < nx and 0 <= j and j < ny and 0 <= k and k < nz:
            P[i,j,k] -= dt*kappa*( (Vx[i+1,j,k]-Vx[i,j,k])/dx + (Vy[i,j+1,k]-Vy[i,j,k])/dy + (Vz[i,j,k+1]-Vz[i,j,k])/dz )
    for (i,j,k) in full_bounds():
        if 1 <= i and i < nx and 1 <= j and j < ny and 0 <= k and k < nz:
            Txy[i,j,k] += dt*G*( (Vx[i,j,k]-Vx[i,j-1,k])/dy + (Vy[i,j,k]-Vy[i-1,j,k])/dx )
    for (i,j,k) in full_bounds():
        if 1 <= i and i < nx and 0 <= j and j < ny and 1 <= k and k < nz:
            Txz[i,j,k] += dt*G*( (Vx[i,j,k]-Vx[i,j,k-1])/dz + (Vz[i,j,k]-Vz[i-1,j,k])/dx )
    for (i,j,k) in full_bounds():
        if 0 <= i and i < nx and 1 <= j and j < ny and 1 <= k and k < nz:
            Tyz[i,j,k] += dt*G*( (Vy[i,j,k]-Vy[i,j,k-1])/dz + (Vz[i,j,k]-Vz[i,j-1,k])/dy )
    for (i,j,k) in full_bounds():
        if 0 <= i and i < nx and 0 <= j and j < ny and 0 <= k and k < nz:
            Txx[i,j,k] += dt*2.0*G*( + 2.0/3.0*(Vx[i+1,j,k]-Vx[i,j,k])/dx - 1.0/3.0*(Vy[i,j+1,k]-Vy[i,j,k])/dy - 1.0/3.0*(Vz[i,j,k+1]-Vz[i,j,k])/dz )
    for (i,j,k) in full_bounds():
        if 0 <= i and i < nx and 0 <= j and j < ny and 0 <= k and k < nz:
            Tyy[i,j,k] += dt*2.0*G*( - 1.0/3.0*(Vx[i+1,j,k]-Vx[i,j,k])/dx + 2.0/3.0*(Vy[i,j+1,k]-Vy[i,j,k])/dy - 1.0/3.0*(Vz[i,j,k+1]-Vz[i,j,k])/dz )
    for (i,j,k) in full_bounds():
        if 0 <= i and i < nx and 0 <= j and j < ny and 0 <= k and k < nz:
            Tzz[i,j,k] += dt*2.0*G*( - 1.0/3.0*(Vx[i+1,j,k]-Vx[i,j,k])/dx - 1.0/3.0*(Vy[i,j+1,k]-Vy[i,j,k])/dy + 2.0/3.0*(Vz[i,j,k+1]-Vz[i,j,k])/dz )
