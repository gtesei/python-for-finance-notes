#!/bin/bash

echo ">> creating conda env ... "

conda env create -f conda-osx.yml

echo "Environment pffn created successfully."

echo "Please, activate pffn environment: source activate pffn"

