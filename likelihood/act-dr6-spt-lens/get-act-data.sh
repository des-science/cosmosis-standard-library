#!/usr/bin/env bash

pushd "$(dirname "$0")"
if [ -d data/v1.2 ]
then
    echo ACT DR6 + SPT Lensing data already downloaded
else
    mkdir -p data
    pushd data
    wget https://lambda.gsfc.nasa.gov/data/suborbital/act_spt_joint/spt_act_likelihood-1.0.tar.gz
    tar -zxvf spt_act_likelihood-1.0.tar.gz
    cp -r spt_act_likelihood-1.0/act_dr6_spt_lenslike/data/v1.2/ .
    rm -rf spt_act_likelihood-1.0
    rm spt_act_likelihood-1.0.tar.gz
    popd
fi
popd
