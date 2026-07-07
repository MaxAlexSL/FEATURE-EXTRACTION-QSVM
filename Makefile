.PHONY: install run test test-quick clean help

install:
	pip install -r requirements_local.txt

run:
	python hybrid_quantum_classical_local.py

test:
	pip install pytest -q && pytest -v tests/

test-quick:
	pytest -v tests/ -k "not backward"

clean:
	rm -rf hybrid_qml_results/ data/ __pycache__/
	rm -rf tests/__pycache__/

help:
	@echo "Comandos disponibles:"
	@echo "  make install     Instalar dependencias"
	@echo "  make run         Ejecutar pipeline completo"
	@echo "  make test        Ejecutar tests (instala pytest si falta)"
	@echo "  make test-quick  Tests rápidos (sin backward)"
	@echo "  make clean       Limpiar resultados y cache"
