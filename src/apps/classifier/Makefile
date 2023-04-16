SHELL := $(shell which bash)

DOCKER := $(shell command -v docker)
DOCKER_COMPOSE := $(shell command -v docker compose)

export DOCKER_BUILDKIT=1

target: up

deps:
ifndef DOCKER
	@echo "docker is not available. Please install docker"
	@exit 1
endif
ifndef DOCKER_COMPOSE
	@echo "docker compose plugin is not available. Please install docker compose plugin"
	@exit 1
endif


up:
	docker compose up --build api frontend