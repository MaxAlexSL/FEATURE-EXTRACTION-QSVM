# Hybrid Quantum-Classical Feature Extraction + QSVM

Reproducción experimental del paper *"Hybrid Quantum-Classical Feature Extraction approach for Image Classification using Autoencoders and Quantum SVMs"* (Slabbert, Petruccione, 2024).

**Pipeline:** Autoencoder Conv. (64d) + PCA (4d) + QSVM (4 qubits)

## Descripción

Este experimento implementa un pipeline híbrido clásico-cuántico que:
1. Entrena un **autoencoder convolucional** para extraer características de imágenes MNIST
2. Reduce dimensionalidad con **PCA** para adaptarse a recursos cuánticos limitados (NISQ)
3. Clasifica usando un **QSVM** (Quantum Support Vector Machine) con kernel cuántico simulado
4. Compara contra un **SVM clásico** con kernel RBF

## Resultados Obtenidos

| Métrica | SVM Clásico (RBF) | QSVM (Quantum Kernel) | Diferencia |
|---------|-------------------|----------------------|------------|
| Accuracy | 1.0000 | 0.8800 | -0.1200 |
| Precision | 1.0000 | 0.9600 | -0.0400 |
| Recall | 1.0000 | 0.8276 | -0.1724 |
| F1 Score | 1.0000 | 0.8889 | -0.1111 |
| ROC-AUC | 1.0000 | 0.9869 | -0.0131 |

### Pruebas Estadísticas
- **t-test**: p = 0.6666 (no significativo)
- **Wilcoxon**: p = 1.0000 (no significativo)
- **Mann-Whitney U**: p = 1.0000 (no significativo)

### Conclusiones Científicas

1. El SVM clásico (100.00%) supera al QSVM (88.00%) en accuracy. Es esperable en régimen NISQ: pocos qubits (4) y muestras limitadas (100).
2. El autoencoder convolucional comprime MNIST (784 px) a 64 dimensiones latentes con loss final de 0.008266.
3. El pipeline híbrido es completamente viable en Google Colab. El cálculo de la matriz kernel (100x100) requiere ~31.6s.
4. El análisis estadístico NO muestra diferencias significativas entre SVM clásico y QSVM.
5. Limitaciones NISQ: kernel O(n²), solo 4 qubits, PCA al 56.95% de varianza explicada.
6. Base sólida para tesis en QML aplicado a reconocimiento facial/biométrico.

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

## Configuración del Experimento

- **Dataset**: MNIST
- **Dimensión latente**: 64
- **Componentes PCA**: 4
- **Qubits del kernel**: 4
- **Muestras entrenamiento QSVM**: 100
- **Framework cuántico**: PennyLane 0.45.0
- **Framework clásico**: PyTorch 2.11.0+cu128
- **Seed**: 42
- **Tiempo estimado**: ~15-20 minutos

## Estructura del Repositorio

```
FEATURE EXTRACTION + QSVM/
├── hybrid_quantum_classical_qsvm_final.ipynb   # Notebook principal
├── README.md                                    # Este archivo
└── hybrid_qml_results/                          # Resultados del experimento
    ├── models/                                  # Modelos entrenados
    │   ├── autoencoder_final.pth
    │   ├── encoder_final.pth
    │   ├── svm_classic.pkl
    │   ├── qsvm.pkl
    │   ├── pca.pkl
    │   └── scaler_latent.pkl
    ├── metrics/                                 # Métricas y matrices kernel
    │   ├── metrics_summary.json
    │   ├── K_train.npy
    │   └── K_test.npy
    ├── plots/                                   # Gráficas generadas
    │   └── *.png
    ├── csv/                                     # Datos tabulares
    │   ├── comparison_table.csv
    │   ├── model_comparison.csv
    │   └── autoencoder_history.csv
    └── conclusiones.txt                         # Conclusiones detalladas
```

## Hardware Utilizado

- GPU: Tesla T4
- RAM: 13.61 GB
- Plataforma: Linux 6.6.122+ (Google Colab)
