

dep: ## Install dependencies with pip and conda! (for gudhi)
	pip install -r requirements.txt
	conda install -c conda-forge gudhi # use -p to target local conda environ.

help: ## Display this help screen
	@grep -h -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'
