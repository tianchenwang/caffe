#!/bin/bash

find . -type f -name '*.py' -o -name '*.cpp' -o -name '*.cu' | xargs etags

