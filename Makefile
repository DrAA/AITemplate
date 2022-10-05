AIT_NAME=ait-built

build:
	./docker/build.sh cuda
	docker run --gpus=all --name ${AIT_NAME} ait python3 \
		/AITemplate/examples/05_stable_diffusion/compile.py \
		--token "${ACCESS_TOKEN}"
	docker commit ${AIT_NAME} ${AIT_NAME}

run:
	docker run --rm --gpus=all ${AIT_NAME} python3 \
		/AITemplate/examples/05_stable_diffusion/demo.py \
		--token "${ACCESS_TOKEN}" \
		--prompt "${PROMPT}"

clean:
	-docker rm ${AIT_NAME}
