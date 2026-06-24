#!/bin/bash

wget --no-check-certificate --show-progress https://share.phys.ethz.ch/~gseg/Predator/data.zip
unzip -q data.zip
rm data.zip
