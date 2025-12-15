# 🍃 Projeto de Classificação de Espécies de Plantas - Visão Computacional

## 📖 Descrição do Projeto

Este projeto implementa um sistema de classificação de espécies de plantas a partir de imagens de folhas utilizando técnicas de Visão Computacional e Aprendizado de Máquina. O sistema realiza segmentação, extração de características, redução de dimensionalidade e classificação.

## 🎯 Objetivos

* Segmentar folhas de imagens usando técnicas de processamento de imagens

* Extrair características geométricas das folhas (circularidade, excentricidade, cantos, razão altura/largura)

* Aplicar PCA para redução de dimensionalidade

* Classificar espécies usando kNN e SVM

* Avaliar e comparar o desempenho dos classificadores

## 📁 Estrutura do Repositório

```
.
├── Leaves/                    # Pasta com imagens do dataset (não versionada)
├── fase1.py                   # Código principal com todas as fases
├── notebook.ipynb             # Notebook Jupyter com análise completa
├── relatorio.pdf              # Relatório técnico do projeto
├── requirements.txt           # Dependências do projeto
├── .gitignore                 # Arquivos ignorados pelo Git
└── README.md                  # Este arquivo
```

## 🛠️ Tecnologias Utilizadas

* **Python 3.8+**

* **OpenCV** - Processamento de imagens e segmentação

* **NumPy** - Operações numéricas

* **Pandas** - Manipulação de dados

* **Matplotlib** - Visualizações

* **Seaborn** - Visualizações estatísticas

* **Scikit-learn** - Machine Learning (PCA, kNN, SVM)

## 📦 Instalação

### 1\. Clone o repositório

```bash
git clone <url-do-repositorio>
cd <nome-do-repositorio>
```

### 2\. Crie um ambiente virtual (recomendado)

```bash
python -m venv venv
```

### 3\. Ative o ambiente virtual

**Windows:**

```bash
venv\Scripts\activate
```

**Linux/Mac:**

```bash
source venv/bin/activate
```

### 4\. Instale as dependências

```bash
pip install -r requirements.txt
```

## 🚀 Como Executar

### Executar o script principal

```bash
python fase1.py
```

### Executar o notebook Jupyter

```bash
jupyter notebook notebook.ipynb
```

## 📊 Pipeline do Projeto

### **Fase 1: Pré-processamento e Segmentação**

* Carregamento das imagens

* Conversão para escala de cinza

* Limiarização usando Otsu

* Operações morfológicas (abertura e fechamento)

* Extração do contorno principal da folha

### **Fase 2: Extração de Características**

* **Circularidade/Compacidade**: Mede o quão circular é a folha

* **Excentricidade**: Baseada no ajuste de elipse

* **Número de Cantos**: Detectados usando Shi-Tomasi

* **Razão Altura/Largura**: Proporção do bounding box

### **Fase 3: Redução de Dimensionalidade (PCA)**

* Normalização dos descritores usando StandardScaler

* Aplicação de PCA para reduzir dimensionalidade

* Análise da variância explicada

* Visualização 2D dos dados

### **Fase 4: Classificação**

* **k-Nearest Neighbors (kNN)**: Testado com k de 1 a 20

* **Support Vector Machine (SVM)**: Kernels linear e RBF

* Divisão treino/teste (70/30)

* Seleção do melhor modelo

### **Fase 5: Avaliação**

* Matriz de confusão

* Métricas: Acurácia, Precisão, Recall, F1-Score

* Análise de erros de classificação

### **Fase 6: Documentação**

* Código organizado e comentado

* Relatório técnico em PDF

* README com instruções completas

## 📈 Resultados Esperados

O projeto gera:

* ✅ Máscaras segmentadas das folhas

* ✅ Vetores de características extraídos

* ✅ Gráficos de variância explicada (PCA)

* ✅ Curvas de desempenho dos classificadores

* ✅ Matriz de confusão

* ✅ Relatório de métricas detalhadas

## 🗂️ Dataset

O dataset utilizado é o **Flavia Leaf Dataset**, contendo imagens de folhas de diferentes espécies de plantas. As imagens devem estar na pasta `Leaves/` na raiz do projeto.

### Formato esperado dos nomes de arquivo:

```
<classe>_<id>.jpg
Exemplo: 1001_1.jpg, 1001_2.jpg, 1002_1.jpg
```

Onde o prefixo numérico antes do underscore representa a classe/espécie.

## ⚠️ Observações Importantes

* Certifique-se de que a pasta `Leaves/` contém as imagens antes de executar

* O código filtra automaticamente classes com menos de 2 amostras

* Gráficos são exibidos durante a execução (use `plt.show()`)

* Para datasets grandes, o processamento pode demorar alguns minutos

## 📝 Licença

Este projeto foi desenvolvido para fins acadêmicos como parte da disciplina de Visão Computacional.

---

**Desenvolvido com 💚 para a disciplina de Visão Computacional**