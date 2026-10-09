# Run from track_2a/, the project root. `make run` is what the judges run on a clean checkout: it builds
# the Docker image and decides the cases in data/examples/ into output/predictions.jsonl.
#
#     make run                                # needs API_KEY, in the environment or in .env
#     make run DATA=data/advanced             # another folder with a cases.jsonl, and its booklets/
#     make run ARGS="model=apertus-8b"        # Hydra overrides for src/entrypoint.py
#     make test                               # the tests, with uv and without Docker

IMAGE ?= apertushack-fact-checking
DATA ?= data/examples
OUTPUT ?= output
ARGS ?=
# The endpoint, its key and the model: those set in the environment win over .env
ENV_VARS = BASE_URL API_KEY LLM_BASE_URL LLM_API_KEY LLM_NAME

.PHONY: run build test

run: build
	mkdir -p $(OUTPUT)
	docker run --rm --platform linux/amd64 \
		$(if $(wildcard .env),--env-file .env) \
		$(foreach var,$(ENV_VARS),$(if $($(var)),-e $(var))) \
		-v "$(abspath $(DATA)):/data:ro" -v "$(abspath $(OUTPUT)):/output" \
		$(IMAGE) --input /data/cases.jsonl --output /output/predictions.jsonl $(ARGS)

build:
	docker build --platform linux/amd64 -t $(IMAGE) .

test:
	uv run pytest -q
