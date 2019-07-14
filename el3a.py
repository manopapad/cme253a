#!/usr/bin/env python2

import numpy as np
import math
import h5py

XMF_HEADER = """<?xml version="1.0" ?>
<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>
<Xdmf xmlns:xi="http://www.w3.org/2003/XInclude" Version="3.0">
  <Domain>
    <Grid GridType="Collection" CollectionType="Temporal">
"""
XMF_BODY = """
      <Grid GridType="Uniform">
        <Time Value="{t}"/>
        <!-- Topology: orthonormal 3D grid -->
        <Topology TopologyType="3DCoRectMesh" Dimensions="{px} {py} {pz}"></Topology>
        <!-- Geometry: Node positions derived implicitly, based on grid origin and cell size -->
        <Geometry GeometryType="Origin_DxDyDz">
          <DataItem Name="Origin" Dimensions="3" NumberType="Float" Precision="8" Format="XML">{ox} {oy} {oz}</DataItem>
          <DataItem Name="Spacing" Dimensions="3" NumberType="Float" Precision="8" Format="XML">{dx} {dy} {dz}</DataItem>
        </Geometry>
        <Attribute Name="pressure" AttributeType="Scalar" Center="Cell">
          <DataItem Dimensions="{nx} {ny} {nz}" NumberType="Float" Precision="8" Format="HDF">{t}.hdf:/pressure</DataItem>
        </Attribute>
      </Grid>
"""
XMF_FOOTER = """
    </Grid>
  </Domain>
</Xdmf>
"""

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
nt = 200
dt = min(dx,dy,dz)/math.sqrt(kappa/rho)/2.75
def Dx(a):
    return np.diff(a, axis=0)/dx
def Dy(a):
    return np.diff(a, axis=1)/dy
def Dz(a):
    return np.diff(a, axis=2)/dz

x = np.linspace((-Lx+dx)/2, (Lx-dx)/2, nx)
y = np.linspace((-Ly+dy)/2, (Ly-dy)/2, ny)
z = np.linspace((-Lz+dz)/2, (Lz-dz)/2, nz)
x, y, z = np.meshgrid(x, y, z, indexing='ij')

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

P   = np.exp(-(x**2+y**2+z**2))
Vx  = np.zeros((nx+1,ny,  nz  ))
Vy  = np.zeros((nx,  ny+1,nz  ))
Vz  = np.zeros((nx,  ny,  nz+1))
Txx = np.zeros((nx,  ny,  nz  ))
Tyy = np.zeros((nx,  ny,  nz  ))
Tzz = np.zeros((nx,  ny,  nz  ))
Txy = np.zeros((nx+1,ny+1,nz  ))
Txz = np.zeros((nx+1,ny,  nz+1))
Tyz = np.zeros((nx,  ny+1,nz+1))

with open('out.xmf', 'w') as xmf_out:
    xmf_out.write(XMF_HEADER)
    t = 0
    while True:
        hdf_out = h5py.File('%d.hdf' % t, 'w')
        hdf_out['pressure'] = P
        hdf_out.close()
        xmf_out.write(XMF_BODY.format(t = t,
                                      nx = nx,    ny = ny,    nz = nz,
                                      px = nx+1,  py = ny+1,  pz = nz+1,
                                      ox = -Lx/2, oy = -Ly/2, oz = -Lz/2,
                                      dx = dx,    dy = dy,    dz = dz))
        if t >= nt:
            break

        Txy[1:-1,1:-1,:] += dt*G*( Dy(Vx[1:-1,:,:]) + Dx(Vy[:,1:-1,:]) )
        Txz[1:-1,:,1:-1] += dt*G*( Dz(Vx[1:-1,:,:]) + Dx(Vz[:,:,1:-1]) )
        Tyz[:,1:-1,1:-1] += dt*G*( Dz(Vy[:,1:-1,:]) + Dy(Vz[:,:,1:-1]) )
        Txx += dt*2.0*G*( + 2.0/3.0*Dx(Vx) - 1.0/3.0*Dy(Vy) - 1.0/3.0*Dz(Vz) )
        Tyy += dt*2.0*G*( - 1.0/3.0*Dx(Vx) + 2.0/3.0*Dy(Vy) - 1.0/3.0*Dz(Vz) )
        Tzz += dt*2.0*G*( - 1.0/3.0*Dx(Vx) - 1.0/3.0*Dy(Vy) + 2.0/3.0*Dz(Vz) )
        Vx[1:-1,:,:] += dt/rho*( - Dx(P) + Dx(Txx) + Dy(Txy[1:-1,:,:]) + Dz(Txz[1:-1,:,:]) )
        Vy[:,1:-1,:] += dt/rho*( - Dy(P) + Dy(Tyy) + Dx(Txy[:,1:-1,:]) + Dz(Tyz[:,1:-1,:]) )
        Vz[:,:,1:-1] += dt/rho*( - Dz(P) + Dz(Tzz) + Dx(Txz[:,:,1:-1]) + Dy(Tyz[:,:,1:-1]) )
        P -= dt*kappa*( Dx(Vx) + Dy(Vy) + Dz(Vz) )

        t += 1
    xmf_out.write(XMF_FOOTER)
