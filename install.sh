#!/bin/bash
# rip out any old installs
bash uninstall.sh || echo "Could not uninstall any old lid_driven_cavity_flow_pin enviornments"
# use mamba because it's great (fall back to conda)
mamba env create -f lid_driven_cavity_flow_pin.yml  || conda env create -f lid_driven_cavity_flow_pin.yml

source activate lid_driven_cavity_flow_pin && \
  poetry install && \
  # pip3 install torch --index-url https://download.pytorch.org/whl/rocm5.4.2 && \
  python -m ipykernel install --user --name lid_driven_cavity_flow_pin --display-name "Python (lid_driven_cavity_flow_pin)"

echo  "\n\n To use the environment, simply call: source activate lid_driven_cavity_flow_pin"