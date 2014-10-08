#!/bin/bash

set -x

# per the installation instructions
sudo yum install atlas-devel leveldb-devel snappy-devel opencv-devel boost-devel hdf5-devel gflags-devel

# glog (already compiled)
# wget https://google-glog.googlecode.com/files/glog-0.3.3.tar.gz
# tar zxvf glog-0.3.3.tar.gz
# cd glog-0.3.3
# ./configure
# make && make install
cd /afs/cs.stanford.edu/u/brodyh/scr/src/glog-0.3.3
sudo make install

# lmdb (already compiled)
# git clone git://gitorious.org/mdb/mdb.git
# cd mdb/libraries/liblmdb
# make && make install
cd /afs/cs.stanford.edu/u/brodyh/scr/src/mdb/libraries/liblmdb
sudo make install

# latest protobuf
sudo yum remove protobuf-devel
cd /afs/cs.stanford.edu/u/brodyh/scr/src/protobuf-2.6.0
sudo make install

# for some reason this version of boost doesn't have this, it is typically a symlink anyway
sudo ln -s /usr/lib64/libboost_thread-mt.so /usr/lib64/libboost_thread.so
