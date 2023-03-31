#!/bin/bash
mamba env remove --name lid_driven_cavity_flow_pin || conda env remove --name lid_driven_cavity_flow_pin
yes | jupyter kernelspec uninstall lid_driven_cavity_flow_pin