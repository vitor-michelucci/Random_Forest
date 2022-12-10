# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#%% Carregando Pacotes
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import seaborn as sns

#%% Carregando o Dataset
#fonte: https://www.kaggle.com/datasets/dhanushnarayananr/credit-card-fraud?resource=download
df = pd.read_csv("card_transdata.csv")

#%% Visualizando o Dataset
print(df.info())

print(df.describe())

#%% Verificando a proporção de transações fraudulentas
fraude_count = df[df["fraud"] == 1]["fraud"].count()
n_fraude_count = df[df["fraud"] == 0]["fraud"].count()
print("Numero de transacoes fraudulentas:", fraude_count)
print("Numero de transacoes nao fraudulentas", n_fraude_count)
print("Percentual de fraudes:", fraude_count / (fraude_count + n_fraude_count) * 100)

#%% Visualização Gráfica
categories = ["Nao_Fraude", "Fraude"]
xpos = np.array([0, 1])
plt.xticks(xpos, categories)
plt.xlabel("Tipo Transacao")
plt.ylabel("Contagem Transacao")
plt.title("Transacoes por Tipo")
plt.bar(xpos[0], n_fraude_count, width= 0.7, color = "g")
plt.bar(xpos[1], fraude_count, width = 0.7, color="r")

#%%Matriz de Correlação

correlation_matrix = df.corr()
fig = plt.figure(figsize=(12,9))
sns.heatmap(correlation_matrix,annot=True,square=True, linewidths=.5,cmap=plt.cm.Reds)
plt.show()

#%% Preparando a Base  - Separando as variáveis X e Y

X = df.iloc[:,0:7]
y = df.iloc[:,7]

#X = df.drop("fraud", axis = 1).values
#y = df["fraud"].values

print(f"Tamanho de X: {X.shape}")
print(f"Tamanho de y: {y.shape}")

#%% Coletar os nomes das variáveis X

features = list(df.drop(columns=['fraud']).columns)

#%% Quebrando o dataset em treino e teste
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=123)

#%% Visualizando 
print('Tamanho de X_train: ', X_train.shape, '\n')
print('Tamanho de X_test: ', X_test.shape, '\n')
print('Tamanho de y_train: ', y_train.shape, '\n')
print('Tamanho de y_test: ', y_test.shape, '\n')

#%% Instanciando o modelo de regressão logística
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(n_jobs=-1, random_state=123)

#%% Fit do modelo 

clf.fit(X_train, y_train)

#%% Obtendo os coeficientes do modelo 
# Coeficientes do modelo
for feature, coef in zip(X.columns, clf.coef_[0].tolist()):
    print(f"{feature}: {round(coef, 3)}")

# Constante do modelo
print(f"Constante: {round(clf.intercept_[0], 3)}")

#%% Verificando a acurácia do modelo
from sklearn.metrics import accuracy_score

y_train_true = y_train
y_train_pred = clf.predict(X_train)
y_test_true = y_test
y_test_pred = clf.predict(X_test)


print(f"Acurácia de Treino: {round(accuracy_score(y_train_true, y_train_pred), 2)}")
print('\n ---------------------------\n')
print(f"Acurácia de Teste: {round(accuracy_score(y_test_true, y_test_pred), 2)}")

#%% Random Forest

from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.metrics import classification_report

rnd_clf = RandomForestClassifier(n_estimators=500, max_depth=4, n_jobs=-1, random_state=42)
rnd_clf.fit(X_train, y_train)

# Predict na base de teste
y_pred_rf = rnd_clf.predict(X_test)

# Matriz de classificação
print(classification_report(y_test, y_pred_rf))

# Importância das variáveis X
for name, score in zip(features, rnd_clf.feature_importances_):
    print(name, score)

#%% Matriz de Cassificação

from sklearn.metrics import confusion_matrix

LABELS = ['N-Fraude', 'Fraude']
conf_matrix = confusion_matrix(y_test, y_pred_rf)
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, cmap= "viridis", fmt="d")
plt.title("Matriz de Confusão")
plt.ylabel('Valores Reais')
plt.xlabel('Valores Preditos')
plt.show()

#%% Acurácia

print("Acurácia = ", (228123 + 18870)/ (228123 + 18870 + 3007))

#%% Fim