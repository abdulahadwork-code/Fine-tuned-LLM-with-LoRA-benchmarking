CONFIG ?= configs/qlora.yaml

.PHONY: install install-dev test lint data train benchmark merge serve smoke docker

install:
	pip install -e ".[all]"

install-dev:
	pip install -e ".[all,dev]"

test:
	pytest

lint:
	ruff check src tests

data:
	domain-llm prepare-data -c $(CONFIG)

train:
	domain-llm train -c $(CONFIG)

benchmark:
	domain-llm benchmark -c $(CONFIG)

merge:
	domain-llm merge -c $(CONFIG) --out-dir outputs/merged

serve:
	domain-llm serve -c $(CONFIG)

# Full pipeline on CPU with a tiny model (~10 min) to verify everything is wired up.
smoke:
	domain-llm prepare-data -c configs/smoke.yaml
	domain-llm train -c configs/smoke.yaml
	domain-llm benchmark -c configs/smoke.yaml

docker:
	docker build -t domain-llm .
