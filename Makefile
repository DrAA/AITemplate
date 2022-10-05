AIT_NAME=ait-built
AIT_OUTPUT=ait-output

build:
ifndef ACCESS_TOKEN
	@echo "Please specify ACCESS_TOKEN=XXX"; exit 1
endif
	./docker/build.sh cuda
	-docker rm ${AIT_NAME}
	docker run --gpus=all --name ${AIT_NAME} ait python3 \
		/AITemplate/examples/05_stable_diffusion/compile.py \
		--token "${ACCESS_TOKEN}"
	docker commit ${AIT_NAME} ${AIT_NAME}
	docker rm ${AIT_NAME}

run:
ifndef PROMPT
	@echo "Please specify PROMPT=XXX"; exit 1
endif
ifndef ACCESS_TOKEN
	@echo "Please specify ACCESS_TOKEN=XXX"; exit 1
endif
	./generate.sh

clean:
	-docker rm ${AIT_NAME}
