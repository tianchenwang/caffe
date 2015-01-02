#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/mpi.hpp"


#define MPI_SAFE_CALL( call ) do {				\
    int err = call;						\
    if (err != MPI_SUCCESS) {					\
      fprintf(stderr, "MPI error %d in file '%s' at line %i",	\
	      err, __FILE__, __LINE__);                         \
      MPI_Abort(MPI_COMM_WORLD,1);				\
      exit(1);							\
    } } while(0)


namespace caffe {
void mpiInit()
{
  //int argc = 1;
  //const char* program = "python";
  //char ** argv = const_cast<char **>(&program);
  int requested = MPI_THREAD_MULTIPLE;
  int provided;
  MPI_SAFE_CALL( MPI_Init_thread(NULL, NULL, requested, &provided) );
  assert(requested == provided);
}

bool mpiFinalized() {
  int flag;
  MPI_SAFE_CALL( MPI_Finalized(&flag) );
  return bool(flag);
}

int mpiQueryThread() {
  int t;
  MPI_SAFE_CALL( MPI_Query_thread(&t) );
  return t;
}

std::string mpiGetProcessorName() {
  const int maxHostName = 80;
  char host[maxHostName];
  int length;
  MPI_SAFE_CALL( MPI_Get_processor_name(host, &length) );
  return std::string(host);
}

int mpiRank(MPI_Comm comm) {
  int r;
  MPI_SAFE_CALL( MPI_Comm_rank(comm, &r) );
  return r;
}

int mpiSize(MPI_Comm comm) {
  int s;
  MPI_SAFE_CALL( MPI_Comm_size(comm, &s) );
  return s;
}

void Send(void* buf, int count, MPI_Datatype type, int dest, int tag, MPI_Comm comm) {
  MPI_SAFE_CALL( MPI_Send((void *)buf, count, type, dest, tag, comm) );
}

MPI_Request Isend(void* buf, int count, MPI_Datatype type, int dest, int tag, MPI_Comm comm) {
  MPI_Request req;
  MPI_SAFE_CALL( MPI_Isend((void*)buf, count, type, dest, tag, comm, &req) );
  return req;
}

void Recv(void* buf, int count, MPI_Datatype type, int source, int tag, MPI_Comm comm) {
  MPI_SAFE_CALL( MPI_Recv((void*)buf, count, type, source, tag, comm, MPI_STATUS_IGNORE) );
}

MPI_Request Irecv(void* buf, int count, MPI_Datatype type, int source, int tag, MPI_Comm comm) {
  MPI_Request req;
  MPI_SAFE_CALL( MPI_Irecv((void*) buf, count, type, source, tag, comm, &req) );
  return req;
}

void Bcast(void* buf, int count, MPI_Datatype type, int root, MPI_Comm comm) {
  MPI_SAFE_CALL( MPI_Bcast((void*) buf, count, type, root, comm) );
}

void Allgather(void* sendbuf, int sendcount, MPI_Datatype sendtype,
	       void* recvbuf, int recvcount, MPI_Datatype recvtype,
	       MPI_Comm comm) {
  MPI_SAFE_CALL( MPI_Allgather((void*) sendbuf, sendcount, sendtype,
			       (void*) recvbuf, recvcount, recvtype,
			       comm) );
}

void Gather(void* sendbuf, int sendcount, MPI_Datatype sendtype,
	    void* recvbuf, int recvcount, MPI_Datatype recvtype,
	    int root, MPI_Comm comm) {
  MPI_SAFE_CALL( MPI_Gather((void*) sendbuf, sendcount, sendtype,
			    (void*) recvbuf, recvcount, recvtype,
			    root, comm) );
}

MPI_Request Iallreduce(void* sendbuf, void* recvbuf, int count, MPI_Datatype datatype,
		       MPI_Op op, MPI_Comm comm) {
  MPI_Request req;
  MPI_SAFE_CALL( MPI_Iallreduce((void*)sendbuf, (void*)recvbuf, count, datatype, op, comm, &req) );
  return req;
}

void Allreduce(void* sendbuf, void* recvbuf, int count, MPI_Datatype datatype,
	       MPI_Op op, MPI_Comm comm) {
  MPI_SAFE_CALL( MPI_Allreduce((void*)sendbuf, (void*)recvbuf, count, datatype, op, comm) );
}

MPI_Request Wait(MPI_Request req) {
  MPI_SAFE_CALL( MPI_Wait(&req, MPI_STATUS_IGNORE) );
  return req;
}

bool Test(MPI_Request req) {
  int flag;
  MPI_SAFE_CALL( MPI_Test(&req, &flag, MPI_STATUS_IGNORE) );
  return bool(flag);
}

bool Request_get_status(MPI_Request req) {
  int flag;
  MPI_SAFE_CALL( MPI_Request_get_status(req, &flag, MPI_STATUS_IGNORE) );
  return bool(flag);
}

void Request_free(MPI_Request req) {
  MPI_SAFE_CALL( MPI_Request_free(&req) );
}

/////////// Cartesian helpers ////////
void Dims_create(int nnodes, int dims, void* ptr) {
  MPI_SAFE_CALL( MPI_Dims_create(nnodes, dims, (int*)ptr) );
}

MPI_Comm Cart_create(MPI_Comm comm_old, int ndims, void* dims, void* periods, int reorder) {
  MPI_Comm comm_cart;
  MPI_SAFE_CALL( MPI_Cart_create(comm_old, ndims, (int*)dims, (int*)periods, reorder, &comm_cart) );
  return comm_cart;
}

void Cart_coords(MPI_Comm comm, int rank, int maxdims, void* coords) {
  MPI_SAFE_CALL( MPI_Cart_coords(comm, rank, maxdims, (int*) coords) );
}

//////////// Datatypes ///////////////
MPI_Datatype Type_contiguous(int count, MPI_Datatype oldtype) {
  MPI_Datatype newtype;
  MPI_SAFE_CALL( MPI_Type_contiguous(count, oldtype, &newtype) );
  return newtype;
}

MPI_Datatype Type_vector(int count, int blocklength, int stride, MPI_Datatype oldtype) {
  MPI_Datatype newtype;
  MPI_SAFE_CALL( MPI_Type_vector(count, blocklength, stride, oldtype, &newtype) );
  return newtype;
}

MPI_Datatype Type_hvector(int count, int blocklength, MPI_Aint stride, MPI_Datatype oldtype) {
  MPI_Datatype newtype;
  MPI_SAFE_CALL( MPI_Type_hvector(count, blocklength, stride, oldtype, &newtype) );
  return newtype;
}

MPI_Datatype Type_commit(MPI_Datatype type) {
  MPI_SAFE_CALL( MPI_Type_commit(&type) );
  return type;
}

MPI_Datatype Type_free(MPI_Datatype type) {
  MPI_SAFE_CALL( MPI_Type_free(&type) );
  return type;
}

int Type_size(MPI_Datatype type) {
  int size;
  MPI_SAFE_CALL( MPI_Type_size(type, &size) );
  return size;
}

int Pack(void* inbuf, int incount, MPI_Datatype datatype, void* outbuf,
	 int outcount, int position, MPI_Comm comm) {
  
  MPI_SAFE_CALL( MPI_Pack((void*) inbuf, incount, datatype, (void*) outbuf,
			  outcount, &position, comm) );
  return position;
}

int Unpack(void* inbuf, int insize, int position,
	   void* outbuf, int outcount, MPI_Datatype datatype, MPI_Comm comm) {
  
  MPI_SAFE_CALL( MPI_Unpack((void*) inbuf, insize, &position,
			    (void*) outbuf, outcount, datatype, comm) );
  return position;
}

void Buffer_attach(void* buf, int size) {
  MPI_SAFE_CALL( MPI_Buffer_attach((void*) buf, size) );
}

}  // namespace caffe
