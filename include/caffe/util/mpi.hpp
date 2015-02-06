#ifndef CAFFE_MPI_HPP_
#define CAFFE_MPI_HPP_

#include "caffe/common.hpp"

namespace caffe {


void caffe_init_mpi(int* pargc, char*** pargv);
void caffe_finalize_mpi();


class MPI {
public:
  MPI() : rank_(0), size_(1) {}
  virtual inline int rank() const { return rank_; }
  virtual inline int size() const { return size_; }
  // Can't make a virtual template function :(
  virtual void Allreduce(const int count, float *sendrecv_buf) = 0;
  virtual void Allreduce(const int count, double *sendrecv_buf) = 0;
  virtual void Allreduce(const int count, int *sendrecv_buf) = 0;
  virtual void Allreduce(const int count, unsigned int *sendrecv_buf) = 0;
  virtual void Bcast(const int count, float *buffer, const int root = 0) = 0;
  virtual void Bcast(const int count, double *buffer, const int root = 0) = 0;
  virtual void Bcast(const int count, int *buffer, const int root = 0) = 0;
  virtual void Bcast(const int count, unsigned int *buffer, const int root = 0) = 0;

 protected:
  int rank_, size_;

  // DISABLE_COPY_AND_ASSIGN(MPI);
 private:
  MPI(const MPI&);
  MPI& operator=(const MPI&);
};

class MPILocal : public MPI {
 public:
  MPILocal() : MPI() {}
  virtual void Allreduce(const int count, float *sendrecv_buf) {}
  virtual void Allreduce(const int count, double *sendrecv_buf) {}
  virtual void Allreduce(const int count, int *sendrecv_buf) {}
  virtual void Allreduce(const int count, unsigned int *sendrecv_buf) {}
  virtual void Bcast(const int count, float *buffer, const int root = 0) {}
  virtual void Bcast(const int count, double *buffer, const int root = 0) {}
  virtual void Bcast(const int count, int *buffer, const int root = 0) {}
  virtual void Bcast(const int count, unsigned int *buffer, const int root = 0) {}

  // DISABLE_COPY_AND_ASSIGN(MPILocal);
 private:
  MPILocal(const MPILocal&);
  MPILocal& operator=(const MPILocal&);
};

class MPIDist : public MPI {
 public:
  MPIDist();
  virtual void Allreduce(const int count, float *sendrecv_buf);
  virtual void Allreduce(const int count, double *sendrecv_buf);
  virtual void Allreduce(const int count, int *sendrecv_buf);
  virtual void Allreduce(const int count, unsigned int *sendrecv_buf);
  virtual void Bcast(const int count, float *buffer, const int root = 0);
  virtual void Bcast(const int count, double *buffer, const int root = 0);
  virtual void Bcast(const int count, int *buffer, const int root = 0);
  virtual void Bcast(const int count, unsigned int *buffer, const int root = 0);

  // DISABLE_COPY_AND_ASSIGN(MPIDist);
 private:
  MPIDist(const MPIDist&);
  MPIDist& operator=(const MPIDist&);
};


}  // namespace caffe


#endif  // CAFFE_MPI_HPP_
