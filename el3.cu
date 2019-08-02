#include <chrono>
#include <fstream>
#include <iostream>
#include <math.h>
#include <sstream>
#include <stdlib.h>
#include "cuda.h"
#include "hdf5.h"

#define USE_DOUBLE
#define DO_CUDA_SYNC
#define DO_OUTPUT

#ifdef USE_DOUBLE
# define DAT double
# define H5_DAT H5T_IEEE_F64LE
#else
# define DAT float
# define H5_DAT H5T_IEEE_F32LE
#endif

#define CUDA_ERROR(call) do {                                         \
  cudaError_t code1 = (call);                                         \
  if (code1 != cudaSuccess) {                                         \
    std::cerr << "ERROR: " << __FILE__ << ":" << __LINE__ << " "      \
              << cudaGetErrorString(code1) << std::endl;              \
    exit(code1);                                                      \
  }                                                                   \
} while(0)
#ifdef DO_CUDA_SYNC
# define CUDA_DO(call) do {                     \
    cudaError_t code2 = (call);                 \
    if (code2 == cudaSuccess) {                 \
      code2 = cudaDeviceSynchronize();          \
    }                                           \
    CUDA_ERROR(code2);                          \
  } while(0)
#else
# define CUDA_DO(call) do {                     \
    cudaError_t code3 = (call);                 \
    CUDA_ERROR(code3);                          \
  } while(0)
#endif
#define CUDA_CHECK() CUDA_DO(cudaSuccess)

constexpr int GPU_ID = 0;
constexpr int BLOCK_X = 8;
constexpr int BLOCK_Y = 8;
constexpr int BLOCK_Z = 8;

constexpr DAT Lx = 15.0;
constexpr DAT Ly = 10.0;
constexpr DAT Lz = 5.0;
constexpr DAT kappa = 1.0;
constexpr DAT rho = 1.0;
constexpr DAT G = 1.0;

constexpr int nx = 150;
constexpr int ny = 100;
constexpr int nz = 50;
constexpr DAT dx = Lx/nx;
constexpr DAT dy = Ly/ny;
constexpr DAT dz = Lz/nz;
constexpr int nt = 5;

constexpr int PROBE_X = 75;
constexpr int PROBE_Y = 50;
constexpr int PROBE_Z = 25;

constexpr int OVERLENGTH_X = 1;
constexpr int OVERLENGTH_Y = 1;
constexpr int OVERLENGTH_Z = 1;

class MyArray {
public:
  MyArray(size_t ilim, size_t jlim, size_t klim)
    : extent_(make_cudaExtent(klim * sizeof(DAT), jlim, ilim)) {
    CUDA_DO(cudaMalloc3D(&ptr_, extent_));
    CUDA_DO(cudaMemset3D(ptr_, 0, extent_));
  }
  void dealloc() {
    CUDA_DO(cudaFree(ptr_.ptr));
    ptr_.ptr = NULL;
  }
public:
  size_t size() {
    return extent_.depth * extent_.height * extent_.width;
  }
  void write(const char* hdf_fname, const char* probe_fname, const char* field) {
    // copy data to host
    size_t ilim = extent_.depth;
    size_t jlim = extent_.height;
    size_t klim = extent_.width / sizeof(DAT);
    DAT* data = (DAT*) malloc(ilim*jlim*klim*sizeof(DAT));
    cudaMemcpy3DParms params = {0};
    params.srcPtr = ptr_;
    params.dstPtr = make_cudaPitchedPtr(data, klim*sizeof(DAT), klim, jlim);
    params.extent = extent_;
    params.kind = cudaMemcpyDeviceToHost;
    CUDA_DO(cudaMemcpy3D(&params));
    // HDF output
    hsize_t dims[3] = {ilim, jlim, klim};
    hid_t hdf_file = H5Fcreate(hdf_fname, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    hid_t dataspace = H5Screate_simple(3, dims, NULL);
    hid_t dataset = H5Dcreate(hdf_file, field, H5_DAT, dataspace,
                              H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(dataset, H5_DAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
    H5Dclose(dataset);
    H5Sclose(dataspace);
    H5Fclose(hdf_file);
    // probe output
    std::ofstream probe_file(probe_fname, std::ofstream::app);
    probe_file << data[PROBE_X*jlim*klim + PROBE_Y*klim + PROBE_Z] << std::endl;
    // cleanup
    free(data);
  }
public:
  __device__ DAT& operator()(size_t i, size_t j, size_t k) {
    return *( (DAT*)((unsigned char*)(ptr_.ptr) + (i*extent_.height+j)*ptr_.pitch) + k );
  }
private:
  cudaExtent extent_;
  cudaPitchedPtr ptr_;
};

__global__ void init(MyArray P) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  int j = blockIdx.y*blockDim.y + threadIdx.y;
  int k = blockIdx.z*blockDim.z + threadIdx.z;
  if (i >= nx || j >= ny || k >= nz) {
    return;
  }
  DAT x = (-Lx+dx)/2 + i*dx;
  DAT y = (-Ly+dy)/2 + j*dy;
  DAT z = (-Lz+dz)/2 + k*dz;
  P(i,j,k) = exp(-(x*x+y*y+z*z));
}

__global__ void compute_V(MyArray P,
                          MyArray Vx, MyArray Vy, MyArray Vz,
                          MyArray Txx, MyArray Tyy, MyArray Tzz,
                          MyArray Txy, MyArray Txz, MyArray Tyz,
                          DAT dt) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  int j = blockIdx.y*blockDim.y + threadIdx.y;
  int k = blockIdx.z*blockDim.z + threadIdx.z;
  if (i >= nx || j >= ny || k >= nz) {
    return;
  }
  DAT P_ijk = P(i,j,k);
  DAT Txy_ijk = Txy(i,j,k);
  DAT Txz_ijk = Txz(i,j,k);
  DAT Tyz_ijk = Tyz(i,j,k);
  if (1 <= i) {
    Vx(i,j,k) += dt/rho*( - (P_ijk-P(i-1,j,k))/dx + (Txx(i,j,k)-Txx(i-1,j,k))/dx
                          + (Txy(i,j+1,k)-Txy_ijk)/dy + (Txz(i,j,k+1)-Txz_ijk)/dz );
  }
  if (1 <= j) {
    Vy(i,j,k) += dt/rho*( - (P_ijk-P(i,j-1,k))/dy + (Tyy(i,j,k)-Tyy(i,j-1,k))/dy
                          + (Txy(i+1,j,k)-Txy_ijk)/dx + (Tyz(i,j,k+1)-Tyz_ijk)/dz );
  }
  if (1 <= k) {
    Vz(i,j,k) += dt/rho*( - (P_ijk-P(i,j,k-1))/dz + (Tzz(i,j,k)-Tzz(i,j,k-1))/dz
                          + (Txz(i+1,j,k)-Txz_ijk)/dx + (Tyz(i,j+1,k)-Tyz_ijk)/dy );
  }
}

__global__ void compute_P_T(MyArray P,
                            MyArray Vx, MyArray Vy, MyArray Vz,
                            MyArray Txx, MyArray Tyy, MyArray Tzz,
                            MyArray Txy, MyArray Txz, MyArray Tyz,
                            DAT dt) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  int j = blockIdx.y*blockDim.y + threadIdx.y;
  int k = blockIdx.z*blockDim.z + threadIdx.z;
  if (i >= nx || j >= ny || k >= nz) {
    return;
  }
  DAT Vx_ijk = Vx(i,j,k);
  DAT Vy_ijk = Vy(i,j,k);
  DAT Vz_ijk = Vz(i,j,k);
  DAT Dx_Vx_ijk = ( Vx(i+1,j,k) - Vx_ijk ) / dx;
  DAT Dy_Vy_ijk = ( Vy(i,j+1,k) - Vy_ijk ) / dy;
  DAT Dz_Vz_ijk = ( Vz(i,j,k+1) - Vz_ijk ) / dz;
  P(i,j,k) -= dt*kappa*( Dx_Vx_ijk + Dy_Vy_ijk + Dz_Vz_ijk );
  Txx(i,j,k) += dt*2.0*G*( + 2.0/3.0*Dx_Vx_ijk - 1.0/3.0*Dy_Vy_ijk - 1.0/3.0*Dz_Vz_ijk );
  Tyy(i,j,k) += dt*2.0*G*( - 1.0/3.0*Dx_Vx_ijk + 2.0/3.0*Dy_Vy_ijk - 1.0/3.0*Dz_Vz_ijk );
  Tzz(i,j,k) += dt*2.0*G*( - 1.0/3.0*Dx_Vx_ijk - 1.0/3.0*Dy_Vy_ijk + 2.0/3.0*Dz_Vz_ijk );
  if (1 <= i && 1 <= j) {
    Txy(i,j,k) += dt*G*( (Vx_ijk-Vx(i,j-1,k))/dy + (Vy_ijk-Vy(i-1,j,k))/dx );
  }
  if (1 <= i && 1 <= k) {
    Txz(i,j,k) += dt*G*( (Vx_ijk-Vx(i,j,k-1))/dz + (Vz_ijk-Vz(i-1,j,k))/dx );
  }
  if (1 <= j && 1 <= k) {
    Tyz(i,j,k) += dt*G*( (Vy_ijk-Vy(i,j,k-1))/dz + (Vz_ijk-Vz(i,j-1,k))/dy );
  }
}

int main() {
  // Complex constants
  DAT dt = min(min(dx,dy),dz)/sqrt(kappa/rho)/2.75;
  // Initialize CUDA
  dim3 grid, block;
  int reqd_x = nx + OVERLENGTH_X;
  block.x = BLOCK_X;
  grid.x = reqd_x/BLOCK_X + (reqd_x%BLOCK_X > 0 ? 1 : 0);
  int reqd_y = ny + OVERLENGTH_Y;
  block.y = BLOCK_Y;
  grid.y = reqd_y/BLOCK_Y + (reqd_y%BLOCK_Y > 0 ? 1 : 0);
  int reqd_z = nz + OVERLENGTH_Z;
  block.z = BLOCK_Z;
  grid.z = reqd_z/BLOCK_Z + (reqd_z%BLOCK_Z > 0 ? 1 : 0);
  CUDA_DO(cudaSetDevice(GPU_ID));
  CUDA_DO(cudaDeviceReset());
  // Allocate arrays
  MyArray P  (nx,  ny,  nz  );
  MyArray Vx (nx+1,ny,  nz  );
  MyArray Vy (nx,  ny+1,nz  );
  MyArray Vz (nx,  ny,  nz+1);
  MyArray Txx(nx,  ny,  nz  );
  MyArray Tyy(nx,  ny,  nz  );
  MyArray Tzz(nx,  ny,  nz  );
  MyArray Txy(nx+1,ny+1,nz  );
  MyArray Txz(nx+1,ny,  nz+1);
  MyArray Tyz(nx,  ny+1,nz+1);
  // Computation
  init<<<grid,block>>>(P); CUDA_CHECK();
  int t = -1;
  CUDA_DO(cudaDeviceSynchronize());
  std::chrono::time_point<std::chrono::system_clock> t_start = std::chrono::system_clock::now();
  while (true) {
    t += 1;
#   ifdef DO_OUTPUT
      std::stringstream hdf_fname;
      hdf_fname << t << ".hdf";
      P.write(hdf_fname.str().c_str(), "pressure.dat", "pressure");
#   endif
    if (t >= nt) {
      break;
    }
    compute_V<<<grid,block>>>(P, Vx, Vy, Vz, Txx, Tyy, Tzz, Txy, Txz, Tyz, dt); CUDA_CHECK();
    compute_P_T<<<grid,block>>>(P, Vx, Vy, Vz, Txx, Tyy, Tzz, Txy, Txz, Tyz, dt); CUDA_CHECK();
  }
  CUDA_DO(cudaDeviceSynchronize());
  std::chrono::time_point<std::chrono::system_clock> t_end = std::chrono::system_clock::now();
  // Report stats
  std::cout << "Domain: " << reqd_x << "x" << reqd_y << "x" << reqd_z << std::endl;
  size_t total_bytes = (P.size() +
                        Vx.size() + Vy.size() + Vz.size() +
                        Txx.size() + Tyy.size() + Tzz.size() +
                        Txy.size() + Txz.size() + Tyz.size());
  double total_gb = ((double) total_bytes) / 1024 / 1024 / 1024;
  std::cout << "Total memory used: " << total_gb << " GB" << std::endl;
  long ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t_end - t_start).count();
  double seconds = ((double) ns) / 1000 / 1000 / 1000;
  std::cout << nt << " iterations completed in " << seconds << " s" << std::endl;
  size_t eff_gb_touched = 2 * total_gb;
  double eff_throughput = eff_gb_touched*nt/seconds;
  std::cout << "Effective throughput: " << eff_throughput << " GB/s" << std::endl;
  // Free arrays
  Tyz.dealloc();
  Txz.dealloc();
  Txy.dealloc();
  Tzz.dealloc();
  Tyy.dealloc();
  Txx.dealloc();
  Vz.dealloc();
  Vy.dealloc();
  Vx.dealloc();
  P.dealloc();
  // Sanity-check CUDA errors
  CUDA_ERROR(cudaGetLastError());
}
