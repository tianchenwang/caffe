#!/bin/bash

find . -type f -name '*.py' -o -name '*.cpp' -o -name '*.hpp' -o -name '*.cu' -o -name '*.proto' | xargs etags
