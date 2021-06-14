#!/bin/bash

cd ../stylegan2/nets/

dataset=$1 # One of FastMRIT1T2, CompMRIT1T2, FFHQ

if ! [[ -e README.md ]]; then
    wget https://databank.illinois.edu/datafiles/l7dre/download -O README.md
fi

if [[ $dataset = "FastMRIT1T2" ]]; then
    printf "\nDownloading StyleGAN2 weights trained on ${dataset} ...\n\n"
    wget https://databank.illinois.edu/datafiles/b4wji/download -O stylegan2-FastMRIT1T2-config-h.pkl

    printf "\nDownloading images ...\n\n"
    wget https://databank.illinois.edu/datafiles/1862l/download -O reals-FastMRIT1T2.png
    wget https://databank.illinois.edu/datafiles/sih64/download -O fakes-stylegan2-FastMRIT1T2-config-h.png
    printf "\nFinished downloading ${dataset} files to ../stylegan2/nets/.\n\n"
fi

if [[ $dataset = "CompMRIT1T2" ]]; then
    printf "\nDownloading StyleGAN2 weights trained on ${dataset} ...\n\n"
    wget https://databank.illinois.edu/datafiles/ln6ug/download -O stylegan2-CompMRIT1T2-config-f.pkl

    printf "\nDownloading images ...\n\n"
    wget https://databank.illinois.edu/datafiles/x5dn7/download -O reals-CompMRIT1T2.png
    wget https://databank.illinois.edu/datafiles/8ej1c/download -O fakes-stylegan2-CompMRIT1T2-config-f.png
    printf "\nFinished downloading ${dataset} files to ../stylegan2/nets/.\n\n"
fi

if [[ $dataset = "FFHQ" ]]; then
    printf "\nDownloading StyleGAN2 weights trained on ${dataset} ...\n\n"
    wget https://databank.illinois.edu/datafiles/vklzy/download -O stylegan2-FFHQ-config-f.pkl

    printf "\nDownloading images ...\n\n"
    wget https://databank.illinois.edu/datafiles/yyipx/download -O reals-FFHQ.png
    wget https://databank.illinois.edu/datafiles/v9d9m/download -O fakes-stylegan2-FFHQ-config-f.png
    printf "\nFinished downloading ${dataset} files to ../stylegan2/nets/.\n\n"
fi

