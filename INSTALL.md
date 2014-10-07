# Installation

See http://caffe.berkeleyvision.org/installation.html for the latest
installation instructions.

Check the issue tracker in case you need help:
https://github.com/BVLC/caffe/issues


# Brody's install notes
- install python2.7 from source
  - ./configure --enable-unicode=ucs4 —enable-shared --prefix=/usr/local
- install boost 1.55 from source linking to new python
  - ./bootstrap.sh --with-libraries=all --with-python=/usr/local/bin/python
  - you may run into problems when you run "sudo ./b2 install" this is because
    when b2 runs "sudo python ..." to get version info it fails, but works
    when you run "python ..." (something to do with sudo not getting the
    right LD_LIBRARY_PATH). You just need to run "./b2 install --debug (or --debug-configuration)"
    then get where it fails to copy it without the right permisions, then just do this separately
