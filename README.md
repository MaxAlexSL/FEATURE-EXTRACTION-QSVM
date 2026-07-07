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
3. El pipeline híbrido es completamente viable en CPU. El cálculo de la matriz kernel (100x100) requiere ~31.6s.
4. El análisis estadístico NO muestra diferencias significativas entre SVM clásico y QSVM.
5. Limitaciones NISQ: kernel O(n²), solo 4 qubits, PCA al 56.95% de varianza explicada.
6. Base sólida para tesis en QML aplicado a reconocimiento facial/biométrico.

## Requisitos e Instalación

**Python:** 3.10+

Instalación de dependencias:

```bash
pip install -r requirements_local.txt
```

Si no tienes GPU NVIDIA (solo CPU), instala PyTorch CPU primero:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements_local.txt
```

## Ejecución

### Opción 1: Notebook Jupyter

```bash
jupyter notebook hybrid_quantum_classical_qsvm_final.ipynb
```

### Opción 2: Script Python

```bash
python hybrid_quantum_classical_local.py
```

El script usa CPU automáticamente. Si hay una GPU NVIDIA disponible, usará CUDA sin cambios.

## Configuración del Experimento

- **Dataset**: MNIST
- **Dimensión latente**: 64
- **Componentes PCA**: 4
- **Qubits del kernel**: 4
- **Muestras entrenamiento QSVM**: 100
- **Framework cuántico**: PennyLane
- **Framework clásico**: PyTorch
- **Seed**: 42
- **Tiempo estimado en CPU**: ~30-60 minutos

## Estructura del Repositorio

```
FEATURE EXTRACTION + QSVM/
├── hybrid_quantum_classical_qsvm_final.ipynb   # Notebook principal (adaptado para CPU local)
├── hybrid_quantum_classical_local.py            # Script Python standalone
├── requirements_local.txt                       # Dependencias para Ubuntu/Linux
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

- CPU: AMD Ryzen 5 / Intel equivalente
- RAM: 8 GB+ recomendado
- Plataforma: Ubuntu/Linux (también funciona en Windows)

*Nota: No requiere GPU. El cálculo del kernel cuántico usa el simulador `default.qubit` de PennyLane en CPU.*
