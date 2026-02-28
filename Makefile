.PHONY: contracts test ci lint clean poll

# Run observatory medium contracts
contracts:
	python verify_observatory_contracts.py
	python verify_forma_contracts.py --strict

# Run unit tests
test:
	python -m unittest -v

# Full CI check (contracts + tests)
ci: contracts test

# Run a single poll cycle
poll:
	python model_observatory.py --once

# Launch the TUI dashboard
tui:
	python observatory_tui.py

# Remove __pycache__ and .pyc files
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name '*.pyc' -delete 2>/dev/null || true
