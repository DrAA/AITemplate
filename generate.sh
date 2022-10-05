#!/usr/bin/env bash
AIT_NAME=ait-built
AIT_OUTPUT=ait-output
set -ex
docker rm ${AIT_OUTPUT} || true
docker run --gpus=all --name ${AIT_OUTPUT} --volume $(pwd)/output:/output ${AIT_NAME} python3 \
    /AITemplate/examples/05_stable_diffusion/demo.py \
        --token "${ACCESS_TOKEN}" \
        --prompt "${PROMPT}"
#docker commit ${AIT_OUTPUT} ${AIT_OUTPUT}
#docker rm ${AIT_OUTPUT}
docker cp ${AIT_OUTPUT}:/*.png /tmp
ls -ltrh /tmp/*.png
