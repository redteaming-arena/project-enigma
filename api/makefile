clean:
	rm -fr .next node_modules .mypy_cache .ruff_cache
	find ./api -type d -name "__pycache__" -exec rm -fr {} +
	find ./api -type f -name "*.py[co]" -delete

test:
	pytest -v api/test/test_auth.py -W ignore::DeprecationWarning -v

backend:
	uvicorn api.main:app --reload --reload-dir api

frontend:
	[ -d "node_modules" ] || npm install --legacy-peer-deps
	npm run next-dev

build:
	npm run build

container:
	docker build -f Dockerfile -t redarena-fullstack:test .
