# Hybrid Quantum-Classical Feature Extraction + QSVM

Reproducción experimental del paper *"Hybrid Quantum-Classical Feature Extraction approach for Image Classification using Autoencoders and Quantum SVMs"* (2024).

**Pipeline:** Autoencoder Conv. (64d) + PCA (4d) + QSVM (4 qubits)

## Requisitos

### Opción 1: Google Colab (recomendado)

Abrir el notebook directamente en Colab y ejecutar todo. Las dependencias se instalan automáticamente al inicio:

```python
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
!pip install pennylane
!pip install qiskit qiskit-machine-learning qiskit-aer
!pip install scikit-learn matplotlib seaborn pandas numpy scipy
!pip install umap-learn
```

Requiere **GPU** (T4 en adelante) para el entrenamiento del autoencoder.

### Opción 2: Local (CPU/GPU)

**Python:** 3.10+

**Dependencias principales:**

| Paquete | Versión |
|---------|---------|
| torch | >= 2.0 |
| torchvision | >= 0.15 |
| pennylane | >= 0.35 |
| qiskit | >= 1.0 |
| qiskit-machine-learning | >= 0.7 |
| qiskit-aer | >= 0.14 |
| scikit-learn | >= 1.3 |
| matplotlib | >= 3.7 |
| seaborn | >= 0.12 |
| pandas | >= 2.0 |
| numpy | >= 1.24 |
| scipy | >= 1.11 |
| umap-learn | >= 0.5 |

Instalación:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install pennylane qiskit qiskit-machine-learning qiskit-aer
pip install scikit-learn matplotlib seaborn pandas numpy scipy umap-learn
```

**Nota:** Para ejecución local sin GPU, cambiar `device` a `"cpu"` en la celda de configuración del notebook.
