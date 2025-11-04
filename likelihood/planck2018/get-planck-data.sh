#!/usr/bin/env bash

pushd "$(dirname "$0")"
if [ -d data/plc_3.0 ]
then
    echo ACT DR6 Lite data already downloaded
else
    wget -O COM_Likelihood_Data-baseline_R3.00.tar.gz "http://pla.esac.esa.int/pla/aio/product-action?COSMOLOGY.FILE_ID=COM_Likelihood_Data-baseline_R3.00.tar.gz"
    tar -zxvf COM_Likelihood_Data-baseline_R3.00.tar.gz
    #mkdir -p data/
    #mv baseline/* data/.
    rm COM_Likelihood_Data-baseline_R3.00.tar.gz
    #rmdir baseline
fi
popd