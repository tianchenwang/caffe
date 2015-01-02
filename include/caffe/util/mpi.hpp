#ifndef CAFFE_UTIL_MPI_H_
#define CAFFE_UTIL_MPI_H_
#include <iostream>
#include <mpi.h>
#include <vector>
#include <string>

namespace caffe {

void mpiInit();

bool mpiFinalized();

int mpiQueryThread();

std::string mpiGetProcessorName();

int mpiRank(MPI_Comm comm);

int mpiSize(MPI_Comm comm);
void Send(void* buf, int count, MPI_Datatype type, int dest, int tag, MPI_Comm comm);

MPI_Request Isend(void* buf, int count, MPI_Datatype type, int dest, int tag, MPI_Comm comm);

void Recv(void* buf, int count, MPI_Datatype type, int source, int tag, MPI_Comm comm);

MPI_Request Irecv(void* buf, int count, MPI_Datatype type, int source, int tag, MPI_Comm comm);

void Bcast(void* buf, int count, MPI_Datatype type, int root, MPI_Comm comm);

void Allgather(void* sendbuf, int sendcount, MPI_Datatype sendtype,
	       void* recvbuf, int recvcount, MPI_Datatype recvtype,
	       MPI_Comm comm);

void Gather(void* sendbuf, int sendcount, MPI_Datatype sendtype,
	    void* recvbuf, int recvcount, MPI_Datatype recvtype,
	    int root, MPI_Comm comm);

MPI_Request Iallreduce(void* sendbuf, void* recvbuf, int count, MPI_Datatype datatype,
		       MPI_Op op, MPI_Comm comm);

void Allreduce(void* sendbuf, void* recvbuf, int count, MPI_Datatype datatype,
	       MPI_Op op, MPI_Comm comm);

MPI_Request Wait(MPI_Request req);

bool Test(MPI_Request req);

bool Request_get_status(MPI_Request req);

void Request_free(MPI_Request req);

/////////// Cartesian helpers ////////
void Dims_create(int nnodes, int dims, void* ptr);

MPI_Comm Cart_create(MPI_Comm comm_old, int ndims, void* dims, void* periods, int reorder);

void Cart_coords(MPI_Comm comm, int rank, int maxdims, void* coords);

//////////// Datatypes ///////////////
MPI_Datatype Type_contiguous(int count, MPI_Datatype oldtype);

MPI_Datatype Type_vector(int count, int blocklength, int stride, MPI_Datatype oldtype);

MPI_Datatype Type_hvector(int count, int blocklength, MPI_Aint stride, MPI_Datatype oldtype);

MPI_Datatype Type_commit(MPI_Datatype type);

MPI_Datatype Type_free(MPI_Datatype type);

int Type_size(MPI_Datatype type);

int Pack(void* inbuf, int incount, MPI_Datatype datatype, void* outbuf,
	 int outcount, int position, MPI_Comm comm);

int Unpack(void* inbuf, int insize, int position,
	   void* outbuf, int outcount, MPI_Datatype datatype, MPI_Comm comm);

void Buffer_attach(void* buf, int size);
}  // namespace caffe

#endif   // CAFFE_UTIL_MPI_H_
