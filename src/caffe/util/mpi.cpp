#ifdef USE_MPI
#include <mpi.h>
#endif
#include <cstdlib>
#include <vector>
#include <utility>
#include <algorithm>

#include "caffe/common.hpp"
#include "caffe/util/mpi.hpp"

#ifdef USE_MPI
// MPI: various checks for different function calls.
#define MPI_CHECK(condition)                                            \
  do {                                                                  \
      int error = condition;                                            \
      CHECK_EQ(error, MPI_SUCCESS) << " MPI error "  << error << " in file '" \
      << __FILE__ << "' at line " << __LINE__;                          \
      } while (0)
#endif


namespace caffe {


static bool distributed = false;

static int get_free_deviceid(int local_rank);

void caffe_init_mpi(int* pargc, char*** pargv) {
  const char* local_rank_env = std::getenv("MV2_COMM_WORLD_LOCAL_RANK");
  shared_ptr<MPI> mpi;
  // We have launched with mpirun_rsh and will use MPI
  if (local_rank_env) {
#ifdef USE_MPI
    distributed = true;
    int local_rank = (std::atoi(local_rank_env));
    local_rank = get_free_deviceid(local_rank);
    Caffe::SetDevice(local_rank);
    int provided, requested = MPI_THREAD_MULTIPLE;
    MPI_CHECK(MPI_Init_thread(pargc, pargv, requested, &provided));
    CHECK_EQ(requested, provided) << "Thread level provided is too low";
    mpi.reset(new MPIDist());
    Caffe::set_device_state(Caffe::FIXED);
    LOG(INFO) << "Rank: " << mpi->rank() << " set device to: "
              << local_rank;
#endif
    } else {
    // Use the local version of MPI which acts as a dummy class
    LOG(INFO) << "Running locally";
    distributed = false;
    mpi.reset(new MPILocal());
  }
  // Turn off logging for all ranks except 0
  if (mpi->rank() > 0) {
    FLAGS_minloglevel = 4;
    // ostringstream rank_str;
    // rank_str << mpi->rank();
    // FLAGS_log_dir = FLAGS_log_dir + "/" + rank_str.str();
    // FLAGS_stderrthreshold = 4;
  }

  Caffe::set_mpi(mpi);
}


struct sort_pred {
  bool operator()(const std::pair<size_t, int>& left,
                  const std::pair<size_t, int>& right) {
    return left.first > right.first;
  }
};


// This is somewhat experimental because there is no guarantee all
// ranks will return the same info for free GPUs
static int get_free_deviceid(int local_rank) {
  int num_devices, orig_device;
  CUDA_CHECK(cudaGetDevice(&orig_device));
  CUDA_CHECK(cudaGetDeviceCount(&num_devices));
  std::vector<std::pair<size_t, int> > device_mems(num_devices);
  for (int devid = 0; devid < num_devices; ++devid) {
    Caffe::SetDevice(devid);
    size_t free_mem = Caffe::DeviceMemoryFree();
    device_mems[devid] = std::pair<int, size_t>(free_mem, devid);
  }
  std::sort(device_mems.begin(), device_mems.end(), sort_pred());
  Caffe::SetDevice(orig_device);
  std::ostringstream info;
  for (int devid = 0; devid < num_devices; ++devid) {
    info << "(" << device_mems[devid].second << ", "
         << device_mems[devid].first / (1024.*1024.) << ") ";
  }
  LOG(INFO) << "Rank "<< local_rank << " " << info.str();
  return device_mems[local_rank].second;
}


void caffe_finalize_mpi() {
#ifdef USE_MPI
  if (distributed)
    MPI_CHECK(MPI_Finalize());
#endif
}


MPIDist::MPIDist() {
#ifdef USE_MPI
  MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank_));
  MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &size_));
#endif
}


void MPIDist::Allreduce(const int count, float *sendrecv_buf) {
#ifdef USE_MPI
  MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, sendrecv_buf,
                          count, MPI_FLOAT,
                          MPI_SUM, MPI_COMM_WORLD));
#endif
}


void MPIDist::Allreduce(const int count, double *sendrecv_buf) {
#ifdef USE_MPI
  MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, sendrecv_buf,
                          count, MPI_DOUBLE,
                          MPI_SUM, MPI_COMM_WORLD));
#endif
}


void MPIDist::Allreduce(const int count, int *sendrecv_buf) {
#ifdef USE_MPI
  MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, sendrecv_buf,
                          count, MPI_INT,
                          MPI_SUM, MPI_COMM_WORLD));
#endif
}


void MPIDist::Allreduce(const int count, unsigned int *sendrecv_buf) {
#ifdef USE_MPI
  MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, sendrecv_buf,
                          count, MPI_UNSIGNED,
                          MPI_SUM, MPI_COMM_WORLD));
#endif
}


void MPIDist::Bcast(const int count, float *buffer, const int root) {
#ifdef USE_MPI
  MPI_CHECK(MPI_Bcast(buffer, count, MPI_FLOAT,
                      root, MPI_COMM_WORLD));
#endif
}


void MPIDist::Bcast(const int count, double *buffer, const int root) {
#ifdef USE_MPI
  MPI_CHECK(MPI_Bcast(buffer, count, MPI_DOUBLE,
                      root, MPI_COMM_WORLD));
#endif
}


void MPIDist::Bcast(const int count, int *buffer, const int root) {
#ifdef USE_MPI
  MPI_CHECK(MPI_Bcast(buffer, count, MPI_INT,
                      root, MPI_COMM_WORLD));
#endif
}


void MPIDist::Bcast(const int count, unsigned int *buffer, const int root) {
#ifdef USE_MPI
  MPI_CHECK(MPI_Bcast(buffer, count, MPI_UNSIGNED,
                      root, MPI_COMM_WORLD));
#endif
}


}  // namespace caffe
