import cv2
import numpy as np
import os
from glob import glob
import matplotlib.pyplot as plt

# --- Bibliotecas adicionais para a Fase 3, 4 e 5 ---
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd
import seaborn as sns
from collections import Counter
# --------------------------------------------------

# --- Funções da Fase 1 ---
def carregar_imagens(caminho_dataset):
    extensoes = ('*.jpg', '*.png')
    caminhos = []
    for ext in extensoes:
        caminhos.extend(glob(os.path.join(caminho_dataset, ext)))
    imagens = []
    nomes = []
    for caminho in caminhos:
        img = cv2.imread(caminho)
        if img is not None:
            imagens.append(img)
            nomes.append(os.path.basename(caminho))
    return imagens, nomes

def segmentar_folha(imagem):
    cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    _, limiar = cv2.threshold(cinza, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.sum(limiar == 255) > np.sum(limiar == 0):
        limiar = cv2.bitwise_not(limiar)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    limpo = cv2.morphologyEx(limiar, cv2.MORPH_CLOSE, kernel)
    limpo = cv2.morphologyEx(limpo, cv2.MORPH_OPEN, kernel)
    contornos, _ = cv2.findContours(limpo, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contornos:
        return np.zeros_like(cinza)
        
    maior_contorno = max(contornos, key=cv2.contourArea)
    mascara = np.zeros_like(limpo)
    cv2.drawContours(mascara, [maior_contorno], -1, 255, thickness=cv2.FILLED)
    return mascara

# --- Funções da Fase 2 (4 features apenas) ---
def calcular_descritores(mascara):
    contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contornos:
        return {
            'circularidade': 0,
            'excentricidade': 0,
            'num_cantos': 0,
            'razao_hw': 0
        }

    maior_contorno = max(contornos, key=cv2.contourArea)

    # Área e perímetro
    area = cv2.contourArea(maior_contorno)
    perimetro = cv2.arcLength(maior_contorno, True)

    # Circularidade/compacidade
    circularidade = 4 * np.pi * area / (perimetro ** 2) if perimetro != 0 else 0

    # Excentricidade (usando elipse ajustada)
    if len(maior_contorno) >= 5:
        elipse = cv2.fitEllipse(maior_contorno)
        (center, axes, orientation) = elipse
        maior_eixo = max(axes)
        menor_eixo = min(axes)
        excentricidade = np.sqrt(1 - (menor_eixo / maior_eixo) ** 2) if maior_eixo != 0 else 0
    else:
        excentricidade = 0

    # Detectar cantos usando Shi-Tomasi
    cantos = cv2.goodFeaturesToTrack(mascara, maxCorners=100, qualityLevel=0.01, minDistance=10)
    num_cantos = 0 if cantos is None else len(cantos)

    # Razão altura/largura
    x, y, w, h = cv2.boundingRect(maior_contorno)
    razao_hw = h / w if w != 0 else 0

    return {
        'circularidade': circularidade,
        'excentricidade': excentricidade,
        'num_cantos': num_cantos,
        'razao_hw': razao_hw
    }

# --- Processamento ---
caminho_dataset = 'Leaves'

imagens, nomes = carregar_imagens(caminho_dataset)

if len(imagens) == 0:
    print("Nenhuma imagem encontrada na pasta Leaves. Por favor, verifique o caminho e o conteúdo da pasta.")
else:
    descritores = []  # MUDANÇA: usar 'descritores' ao invés de 'todos_descritores'
    
    for i, imagem in enumerate(imagens):
        print(f"Processando imagem: {nomes[i]}")
        mascara = segmentar_folha(imagem)
        desc = calcular_descritores(mascara)
        
        # Converter dicionário para lista na ordem correta
        descritores.append([
            desc['circularidade'],
            desc['excentricidade'],
            desc['num_cantos'],
            desc['razao_hw']
        ])
        
        print(f"Descritores para {nomes[i]}: {desc}")

    # Opcional: Visualizar a máscara da primeira imagem processada
    if len(imagens) > 0:
        mascara_exemplo = segmentar_folha(imagens[0])
        plt.imshow(mascara_exemplo, cmap='gray')
        plt.title('Máscara da primeira folha processada')
        plt.show()

    print("\nExtração de descritores concluída para todas as imagens.")

    # ==========================================================================
    # === Fase 3: Redução de Dimensionalidade (PCA) ============================
    # ==========================================================================
    
    # Nomes das colunas (4 features)
    feature_names = ['Circularidade', 'Excentricidade', 'Num_Cantos', 'Razao_HW']
    
    # 1. Converter para DataFrame
    df_descritores = pd.DataFrame(descritores, columns=feature_names)
    print("\n[Fase 3] DataFrame de descritores criado:")
    print(df_descritores.head())

    # 2. Normalizar os vetores de descritores
    scaler = StandardScaler()
    descritores_normalizados = scaler.fit_transform(df_descritores)
    print("\n[Fase 3] Dados normalizados (primeiras 5 linhas):")
    print(descritores_normalizados[:5])

    # 3. Aplicar PCA
    pca_completo = PCA()
    pca_completo.fit(descritores_normalizados)
    variancia_explicada = pca_completo.explained_variance_ratio_
    variancia_acumulada = np.cumsum(variancia_explicada)

    print("\n[Fase 3] Variância explicada por componente:")
    for i, var in enumerate(variancia_explicada):
        print(f"  PC{i+1}: {var:.4f} ({var*100:.2f}%)")

    print("\n[Fase 3] Variância acumulada:")
    for i, var_ac in enumerate(variancia_acumulada):
        print(f"  Até PC{i+1}: {var_ac:.4f} ({var_ac*100:.2f}%)")

    # Escolher 2 componentes para visualização 2D
    n_componentes = 2
    pca = PCA(n_components=n_componentes)
    componentes_principais = pca.fit_transform(descritores_normalizados)

    print(f"\n[Fase 3] PCA aplicado com {n_componentes} componentes.")
    print("[Fase 3] Variância explicada por componente selecionado:")
    for i in range(n_componentes):
        print(f"  PC{i+1}: {pca.explained_variance_ratio_[i]:.4f} ({pca.explained_variance_ratio_[i]*100:.2f}%)")
    var_total = np.sum(pca.explained_variance_ratio_)
    print(f"[Fase 3] Variância explicada total com {n_componentes} componentes: {var_total:.4f} ({var_total*100:.2f}%)")

    # 4. Visualizar as duas primeiras componentes em gráfico 2D
    plt.figure(figsize=(10, 8))
    plt.scatter(componentes_principais[:, 0], componentes_principais[:, 1], alpha=0.7)
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.2f}% da variância)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.2f}% da variância)')
    plt.title('[Fase 3] Visualização 2D das Folhas após PCA')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Gráfico de variância explicada
    plt.figure(figsize=(8, 5))
    componentes_idx = np.arange(1, len(variancia_explicada) + 1)
    plt.bar(componentes_idx, variancia_explicada, alpha=0.7, label='Variância Individual')
    plt.step(componentes_idx, variancia_acumulada, where='mid', label='Variância Acumulada', color='red')
    plt.xlabel('Componentes Principais')
    plt.ylabel('Proporção de Variância Explicada')
    plt.title('[Fase 3] Variância Explicada pelos Componentes Principais')
    plt.legend()
    plt.grid(True)
    plt.xticks(componentes_idx)
    plt.tight_layout()
    plt.show()

    print("\n[Fase 3] Concluída. Os dados reduzidos estão em 'componentes_principais'.")
    
    # ==========================================================================
    # === Fase 4: Classificação ================================================
    # ==========================================================================

    # --- 1. Preparar os rótulos das classes ---
    def extrair_classe(nome_arquivo):
        """
        Agrupa 26 classes originais em 10 grupos balanceados.
        Ideal para 4 features geométricas.
        """
        nome_sem_ext = os.path.splitext(nome_arquivo)[0]
        import re
        numeros = re.findall(r'\d+', nome_sem_ext)
        
        if numeros:
            numero_completo = numeros[0]
            
            if len(numero_completo) >= 2:
                classe_original = int(numero_completo[:2])
            else:
                classe_original = int(numero_completo)
            
            # Mapear classes 10-35 para grupos 0-9
            classe_min = 10
            classe_max = 35
            num_grupos = 10
            
            classe_original = max(classe_min, min(classe_original, classe_max))
            grupo = int((classe_original - classe_min) * num_grupos / (classe_max - classe_min + 1))
            grupo = min(grupo, num_grupos - 1)
            
            return grupo
        else:
            print(f"Aviso: Não foi possível extrair classe de '{nome_arquivo}'.")
            return -1

    labels = [extrair_classe(nome) for nome in nomes]
    print("\n[Fase 4] Rótulos extraídos (todos):")
    print(labels)

    contagem = Counter(labels)
    print("\n[Fase 4] Contagem das classes:")
    for classe, qtd in contagem.items():
        print(f"Classe {classe}: {qtd} amostra(s)")

    # --- 2. Filtrar classes com pelo menos 2 amostras ---
    indices_validos = [i for i, label in enumerate(labels) if contagem[label] > 1]

    if len(indices_validos) == 0:
        print("\n[Fase 4] Nenhuma classe com 2 ou mais amostras. Removendo filtro e stratify.")
        X_train, X_test, y_train, y_test = train_test_split(
            componentes_principais, labels, test_size=0.3, random_state=42
        )
    else:
        componentes_principais_filtrados = componentes_principais[indices_validos]
        labels_filtrados = [labels[i] for i in indices_validos]
        print(f"\n[Fase 4] Dados filtrados para permitir stratify. Novo tamanho: {len(labels_filtrados)} amostras.")
        X_train, X_test, y_train, y_test = train_test_split(
            componentes_principais_filtrados, labels_filtrados, test_size=0.3, random_state=42, stratify=labels_filtrados
        )

    print(f"\n[Fase 4] Dados divididos:")
    print(f"  - Treino: {X_train.shape[0]} amostras")
    print(f"  - Teste:  {X_test.shape[0]} amostras")

    # --- 3. Treinar e testar classificador kNN ---
    print("\n[Fase 4] Treinando classificador kNN...")
    k_values = range(1, 21)
    accuracies_knn = []

    melhor_k = 1
    melhor_acc_knn = 0
    melhor_modelo_knn = None

    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        y_pred_knn = knn.predict(X_test)
        acc = accuracy_score(y_test, y_pred_knn)
        accuracies_knn.append(acc)
        if acc > melhor_acc_knn:
            melhor_acc_knn = acc
            melhor_k = k
            melhor_modelo_knn = knn

    print(f"[Fase 4] Melhor k para kNN: {melhor_k} com acurácia de {melhor_acc_knn:.4f}")

    # --- 4. Gerar curva desempenho x k para kNN ---
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, accuracies_knn, marker='o')
    plt.title('[Fase 4] Desempenho do kNN para diferentes valores de k')
    plt.xlabel('Valor de k')
    plt.ylabel('Acurácia')
    plt.grid(True)
    plt.axvline(x=melhor_k, color='r', linestyle='--', label=f'Melhor k={melhor_k}')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # --- 5. Treinar e testar classificador SVM ---
    print("\n[Fase 4] Treinando classificador SVM...")
    kernels = ['linear', 'rbf']
    modelos_svm = {}
    resultados_svm = {}

    for kernel in kernels:
        print(f"  - Treinando SVM com kernel '{kernel}'...")
        svm = SVC(kernel=kernel, random_state=42)
        svm.fit(X_train, y_train)
        y_pred_svm = svm.predict(X_test)
        acc_svm = accuracy_score(y_test, y_pred_svm)
        modelos_svm[kernel] = svm
        resultados_svm[kernel] = {
            'accuracy': acc_svm,
            'predictions': y_pred_svm
        }
        print(f"    Acurácia SVM ({kernel}): {acc_svm:.4f}")

    melhor_kernel = max(resultados_svm, key=lambda k: resultados_svm[k]['accuracy'])
    melhor_acc_svm = resultados_svm[melhor_kernel]['accuracy']
    melhor_modelo_svm = modelos_svm[melhor_kernel]
    print(f"[Fase 4] Melhor kernel para SVM: '{melhor_kernel}' com acurácia de {melhor_acc_svm:.4f}")

    # --- 6. Comparar resultados dos classificadores ---
    print("\n[Fase 4] Comparação Final de Classificadores:")
    print(f"  - kNN (k={melhor_k}):      Acurácia = {melhor_acc_knn:.4f}")
    print(f"  - SVM ({melhor_kernel}):   Acurácia = {melhor_acc_svm:.4f}")

    if melhor_acc_knn > melhor_acc_svm:
        print("  - O classificador vencedor é o kNN.")
        classificador_final = melhor_modelo_knn
        tipo_classificador = "kNN"
        y_pred_final = melhor_modelo_knn.predict(X_test)
    else:
        print("  - O classificador vencedor é o SVM.")
        classificador_final = melhor_modelo_svm
        tipo_classificador = "SVM"
        y_pred_final = resultados_svm[melhor_kernel]['predictions']

    print(f"\n[Fase 4] Classificação concluída. O melhor modelo ({tipo_classificador}) será usado na Fase 5.")

    # ==========================================================================
    # === Fase 5: Avaliação e Análise ==========================================
    # ==========================================================================

    print("\n[Fase 5] Gerando matriz de confusão...")

    if len(set(y_test)) == 0:
        print("Erro: Não há classes suficientes em y_test para gerar matriz de confusão.")
    else:
        cm = confusion_matrix(y_test, y_pred_final)

        plt.figure(figsize=(10, 8))
        try:
            cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                        xticklabels=sorted(set(y_test)), yticklabels=sorted(set(y_test)))
            plt.title(f'[Fase 5] Matriz de Confusão Normalizada - {tipo_classificador}')
        except Exception as e:
            print(f"Aviso: Não foi possível normalizar a matriz de confusão: {e}")
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=sorted(set(y_test)), yticklabels=sorted(set(y_test)))
            plt.title(f'[Fase 5] Matriz de Confusão (não normalizada) - {tipo_classificador}')

        plt.xlabel('Classe Predita')
        plt.ylabel('Classe Verdadeira')
        plt.tight_layout()
        plt.show()

    print("\n[Fase 5] Métricas detalhadas de classificação:")

    if len(set(y_test)) < 2:
        print("Aviso: Menos de 2 classes em y_test. Métricas não podem ser calculadas.")
    else:
        try:
            report = classification_report(y_test, y_pred_final, output_dict=True, zero_division=0)
            report_df = pd.DataFrame(report).transpose()
            print(report_df)

            acuracia_final = accuracy_score(y_test, y_pred_final)
            print(f"\n[Fase 5] Acurácia Final do Modelo ({tipo_classificador}): {acuracia_final:.4f}")
        except Exception as e:
            print(f"Erro ao calcular métricas: {e}")

    print("\n[Fase 5] Análise de Erros de Classificação:")

    y_test_array = np.array(y_test)
    erros_indices = np.where(y_test_array != y_pred_final)[0]
    print(f"  - Total de erros: {len(erros_indices)} de {len(y_test)} amostras de teste.")

    if len(erros_indices) > 0:
        print("  - Amostras com erro (verdadeira -> predita):")
        for i, idx in enumerate(erros_indices[:10]):
            verdadeira = y_test_array[idx]
            predita = y_pred_final[idx]
            print(f"    {i+1}. Classe {verdadeira} foi classificada como {predita}.")
            
        pares_erros = [(y_test_array[idx], y_pred_final[idx]) for idx in erros_indices]
        contagem_erros = Counter(pares_erros)
        print("\n  - Pares de classes mais confundidos (verdadeira -> predita):")
        for par, count in contagem_erros.most_common(5):
            print(f"    Classe {par[0]} -> Classe {par[1]}: {count} vezes")
    else:
        print("  - Nenhum erro de classificação foi encontrado!")

    print("\n[Fase 5] Avaliação concluída.")