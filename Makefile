
DOCKER_IMAGE_NAME=team-23-5i


### All these Values should be correctly preconfigured
REGISTRY=featurecloud.ai
DOCKER_IMAGE_VERSION=latest
DOCKER_IMAGE_TAGNAME=$(DOCKER_IMAGE_NAME):$(DOCKER_IMAGE_VERSION)

default: build ## default = build

build: ## build the image
	docker build -t $(DOCKER_IMAGE_TAGNAME) .
	docker tag $(DOCKER_IMAGE_TAGNAME) $(REGISTRY)/$(DOCKER_IMAGE_TAGNAME)

push: build ## push image to docker registry
	featurecloud app publish $(DOCKER_IMAGE_NAME)

test: build ## test the image
	docker run --rm $(DOCKER_IMAGE_TAGNAME) /bin/echo "Success."

rmi: ## remove the image
	docker rmi -f $(DOCKER_IMAGE_TAGNAME)

run: build config.yml ## run the image
	# to do a test run of your container with the following statement. In the logs you should see a server starting. When using Windows bases Systems we recognized mounting works better when triggering command directly in WSL System.
	docker run --rm -d -v ./config.yml:/mnt/input/config.yml -v ./data/output:/mnt/output -p 9000:9000 $(REGISTRY)/$(DOCKER_IMAGE_NAME):$(DOCKER_IMAGE_VERSION)

help: ## This help dialog
	@IFS=$$'\n' ; \
		help_lines=(`fgrep -h "##" $(MAKEFILE_LIST) | fgrep -v fgrep | sed -e 's/\\$$//'`); \
		for help_line in $${help_lines[@]}; do \
			IFS=$$'#' ; \
			help_split=($$help_line) ; \
			help_command=`echo $${help_split[0]} | sed -e 's/^ *//' -e 's/ *$$//'` ; \
			help_info=`echo $${help_split[2]} | sed -e 's/^ *//' -e 's/ *$$//'` ; \
			printf "%-10s %s\n" $$help_command $$help_info ; \
		done

.PHONY: help run list build test push rmi rebuild

config.yml:
	$(error Please create a config.yml file)
