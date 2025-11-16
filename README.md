# Food Classification 🍽️

Projeto de classificação de alimentos utilizando Deep Learning com PyTorch e ResNet-50.

## 📋 Descrição

Este projeto implementa um classificador de alimentos utilizando técnicas avançadas de Deep Learning. O modelo é baseado na arquitetura ResNet-50 pré-treinada no ImageNet, com fine-tuning e otimizações específicas para melhorar a acurácia na classificação de diferentes tipos de pratos.

## 🎯 Objetivo

Desenvolver um modelo de classificação de imagens capaz de identificar diferentes tipos de alimentos com alta precisão, aplicando técnicas de transfer learning e data augmentation para melhorar a generalização.

## 📊 Dataset

O projeto utiliza o **Food-41 Dataset** disponível no Kaggle:
- **Fonte:** [Food-41 Dataset (Kaggle)](https://www.kaggle.com/kmader/food41)
- **Número de classes:** 41 categorias diferentes de alimentos
- **Download:** Realizado automaticamente via `kagglehub`

### Divisão dos Dados
- **Treino:** 80% do dataset
- **Validação:** 20% do dataset
- **Opção de subset:** Possibilidade de treinar com um subset menor (ex: 5000 imagens) para testes mais rápidos

## 🏗️ Arquitetura do Modelo

### Modelo Base: ResNet-50

O projeto utiliza uma ResNet-50 pré-treinada com as seguintes características:

```
Modelo: ResNet-50 (pré-treinado no ImageNet)
├── Camadas Congeladas:
│   ├── conv1, bn1 (camadas iniciais)
│   ├── layer1 (bloco residual 1)
│   └── layer2 (bloco residual 2)
├── Camadas Treináveis:
│   ├── layer3 (bloco residual 3)
│   └── layer4 (bloco residual 4)
└── Classificador Personalizado:
    ├── Dropout(0.5)
    ├── Linear(2048 → 512)
    ├── ReLU
    ├── Dropout(0.3)
    └── Linear(512 → num_classes)
```

### Especificações Técnicas
- **Resolução de entrada:** 224x224 pixels
- **Normalização:** ImageNet mean/std ([0.485, 0.456, 0.406] / [0.229, 0.224, 0.225])
- **Batch size:** 32
- **Device:** CUDA (GPU) ou CPU

## 🔄 Data Augmentation

### Transformações de Treino
Para melhorar a generalização do modelo, são aplicadas as seguintes técnicas:

- **Resize:** 224x224 pixels
- **RandomHorizontalFlip:** p=0.5
- **RandomRotation:** ±20 graus
- **ColorJitter:** brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
- **RandomAffine:** translação de até 10%
- **RandomPerspective:** distortion_scale=0.2, p=0.5
- **Normalização:** ImageNet standards

### Transformações de Validação
- **Resize:** 224x224 pixels
- **Normalização:** ImageNet standards (sem augmentation)

## ⚙️ Configuração do Treinamento

### Otimizador e Hiperparâmetros

| Parâmetro | Valor |
|-----------|-------|
| **Otimizador** | Adam |
| **Learning Rate** | 0.0001 |
| **Weight Decay** | 1e-4 |
| **Scheduler** | ReduceLROnPlateau (patience=3, factor=0.5) |
| **Loss Function** | CrossEntropyLoss com Label Smoothing (0.1) |
| **Épocas Máximas** | 50 |
| **Early Stopping** | Patience de 10 épocas |

### Estratégias de Regularização
- **Dropout:** 0.5 na primeira camada FC, 0.3 na segunda
- **Weight Decay:** L2 regularization (1e-4)
- **Label Smoothing:** 0.1 para evitar overconfidence

## 📈 Resultados

### Melhorias Implementadas

| Componente | Melhoria |
|------------|----------|
| **Modelo** | ResNet-50 (mais profundo e robusto) |
| **Resolução** | 224x224 pixels (padrão ImageNet) |
| **Fine-tuning** | Layer3 e Layer4 treináveis |
| **Data Augmentation** | ColorJitter, RandomAffine, RandomPerspective |
| **Learning Rate** | 0.0001 com ReduceLROnPlateau |
| **Regularização** | Weight Decay (1e-4), Dropout (0.5 e 0.3) |
| **Loss Function** | CrossEntropyLoss com Label Smoothing (0.1) |
| **Early Stopping** | Patience de 10 épocas |
| **Épocas** | Até 50 (com early stopping) |

### Performance Esperada
- **Acurácia Anterior:** ~42%
- **Acurácia Esperada:** 65-75%
- **Observação:** Tempo de treinamento mais longo, mas com melhor generalização

## 🚀 Como Usar

### Pré-requisitos

```bash
pip install torch torchvision kagglehub matplotlib numpy
```

### Instalação

1. Clone o repositório:
```bash
git clone https://github.com/Andre-Quintela/Food-Classification.git
cd Food-Classification
```

2. Execute o notebook:
```bash
jupyter notebook Classificacao_de_Pratos.ipynb
```

### Executando o Projeto

O notebook está organizado nas seguintes seções:

1. **Download do Dataset:** Baixa automaticamente o Food-41 via kagglehub
2. **Preparação dos Dados:** Carrega e aplica transformações
3. **Definição do Modelo:** Cria a arquitetura ResNet-50 customizada
4. **Configuração do Treinamento:** Define otimizador e scheduler
5. **Treinamento:** Executa o loop de treinamento com early stopping
6. **Visualização dos Resultados:** Plots de loss e acurácia
7. **Predições:** Exemplos de classificação em imagens do conjunto de validação

### Ajustando o Tamanho do Dataset

Para testes mais rápidos, você pode reduzir o tamanho do dataset:

```python
subset_size = 5000  # Use None para dataset completo
```

Valores sugeridos: 5000, 10000, 20000, ou None para o dataset completo.

## 📁 Estrutura do Projeto

```
Food-Classification/
├── Classificacao_de_Pratos.ipynb  # Notebook principal com todo o pipeline
└── README.md                       # Este arquivo
```

## 🛠️ Tecnologias Utilizadas

- **Python 3.x**
- **PyTorch:** Framework de Deep Learning
- **TorchVision:** Modelos e transformações de imagens
- **KaggleHub:** Download automático de datasets
- **Matplotlib:** Visualização de resultados
- **NumPy:** Operações numéricas

## 📊 Visualizações

O projeto gera as seguintes visualizações:

1. **Curvas de Treinamento:**
   - Loss de treino vs validação
   - Acurácia de treino vs validação

2. **Exemplos de Predições:**
   - Imagens de validação com predições e labels verdadeiros
   - Visualização de acertos e erros do modelo

## 🤝 Contribuindo

Contribuições são bem-vindas! Sinta-se à vontade para:

1. Fazer um fork do projeto
2. Criar uma branch para sua feature (`git checkout -b feature/AmazingFeature`)
3. Commit suas mudanças (`git commit -m 'Add some AmazingFeature'`)
4. Push para a branch (`git push origin feature/AmazingFeature`)
5. Abrir um Pull Request

## 📝 Licença

Este projeto está sob uma licença aberta. Sinta-se livre para usar e modificar.

## 👤 Autor

**Andre Quintela**

- GitHub: [@Andre-Quintela](https://github.com/Andre-Quintela)

## 🙏 Agradecimentos

- Dataset Food-41 disponibilizado por kmader no Kaggle
- Comunidade PyTorch pelos excelentes recursos e documentação
- Arquitetura ResNet desenvolvida por Microsoft Research
