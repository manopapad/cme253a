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
def Dx(a, i, j, k):
    return (a[i+1,j,k] - a[i,j,k])/dx
def Dy(a, i, j, k):
    return (a[i,j+1,k] - a[i,j,k])/dy
def Dz(a, i, j, k):
    return (a[i,j,k+1] - a[i,j,k])/dz

OVERLENGTH_X = 1
OVERLENGTH_Y = 1
OVERLENGTH_Z = 1
def bounds(xmin, xlim, ymin, ylim, zmin, zlim):
    def p((i, j, k)):
        return (xmin <= i and i < xlim and
                ymin <= j and j < ylim and
                zmin <= k and k < zlim)
    prod = itertools.product(range(0, nx+OVERLENGTH_X),
                             range(0, ny+OVERLENGTH_Y),
                             range(0, nz+OVERLENGTH_Z))
    return itertools.ifilter(p, prod)

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

for (i,j,k) in bounds(0, nx, 0, ny, 0, nz):
    x[i,j,k] = (-Lx+dx)/2 + i*dx
for (i,j,k) in bounds(0, nx, 0, ny, 0, nz):
    y[i,j,k] = (-Ly+dy)/2 + j*dy
for (i,j,k) in bounds(0, nx, 0, ny, 0, nz):
    z[i,j,k] = (-Lz+dz)/2 + k*dz
for (i,j,k) in bounds(0, nx, 0, ny, 0, nz):
    P[i,j,k] = math.exp(-(x[i,j,k]**2+y[i,j,k]**2+z[i,j,k]**2))

t = -1
while True:
    t += 1
    hdf_out = h5py.File('%d.hdf' % t, 'w')
    hdf_out['pressure'] = P
    hdf_out.close()
    if t >= nt:
        break

    for (i,j,k) in bounds(1, nx, 1, ny, 0, nz):
        Txy[i,j,k] += dt*G*( Dy(Vx,i,j-1,k) + Dx(Vy,i-1,j,k) )
    for (i,j,k) in bounds(1, nx, 0, ny, 1, nz):
        Txz[i,j,k] += dt*G*( Dz(Vx,i,j,k-1) + Dx(Vz,i-1,j,k) )
    for (i,j,k) in bounds(0, nx, 1, ny, 1, nz):
        Tyz[i,j,k] += dt*G*( Dz(Vy,i,j,k-1) + Dy(Vz,i,j-1,k) )
    for (i,j,k) in bounds(0, nx, 0, ny, 0, nz):
        Txx[i,j,k] += dt*2.0*G*( + 2.0/3.0*Dx(Vx,i,j,k) - 1.0/3.0*Dy(Vy,i,j,k) - 1.0/3.0*Dz(Vz,i,j,k) )
    for (i,j,k) in bounds(0, nx, 0, ny, 0, nz):
        Tyy[i,j,k] += dt*2.0*G*( - 1.0/3.0*Dx(Vx,i,j,k) + 2.0/3.0*Dy(Vy,i,j,k) - 1.0/3.0*Dz(Vz,i,j,k) )
    for (i,j,k) in bounds(0, nx, 0, ny, 0, nz):
        Tzz[i,j,k] += dt*2.0*G*( - 1.0/3.0*Dx(Vx,i,j,k) - 1.0/3.0*Dy(Vy,i,j,k) + 2.0/3.0*Dz(Vz,i,j,k) )
    for (i,j,k) in bounds(1, nx, 0, ny, 0, nz):
        Vx[i,j,k] += dt/rho*( - Dx(P,i-1,j,k) + Dx(Txx,i-1,j,k) + Dy(Txy,i,j,k) + Dz(Txz,i,j,k) )
    for (i,j,k) in bounds(0, nx, 1, ny, 0, nz):
        Vy[i,j,k] += dt/rho*( - Dy(P,i,j-1,k) + Dy(Tyy,i,j-1,k) + Dx(Txy,i,j,k) + Dz(Tyz,i,j,k) )
    for (i,j,k) in bounds(0, nx, 0, ny, 1, nz):
        Vz[i,j,k] += dt/rho*( - Dz(P,i,j,k-1) + Dz(Tzz,i,j,k-1) + Dx(Txz,i,j,k) + Dy(Tyz,i,j,k) )
    for (i,j,k) in bounds(0, nx, 0, ny, 0, nz):
        P[i,j,k] -= dt*kappa*( Dx(Vx,i,j,k) + Dy(Vy,i,j,k) + Dz(Vz,i,j,k) )
