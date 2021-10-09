#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
arquivo = pd.read_csv('F:/User/Documents/DataSet/archive/wine_dataset.csv')


# In[4]:


arquivo.head()


# In[ ]:


arquivo['style'] = arquivo['style'].replace('red', 0)  #filtra a coluna e substitui o texto red por 0


# In[ ]:


arquivo['style'] = arquivo['style'].replace('white', 1) #filtra a coluna e substitui o texto white por 1


# In[6]:


# Separando as variaveis entre preditoras e variavel alvo

y = arquivo['style']
x = arquivo.drop('style', axis = 1) # excluindo a coluna style armazena o restante do arquivo em y


# In[8]:


from sklearn.model_selection import train_test_split

#criando conjunto de dados de treino e teste:
x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size = 0.3)


# In[9]:


from sklearn.ensemble import ExtraTreesClassifier

#criação do modelo
modelo = ExtraTreesClassifier()
modelo.fit (x_treino, y_treino)

# imprimindo resultados
resultado = modelo.score(x_teste, y_teste)
print("Acurácia:", resultado)


# In[10]:


y_teste[400:403]


# In[ ]:


x_teste[400:403]


# In[13]:


previsoes = modelo.predict(x_teste[400:403])


# In[14]:


previsoes


# In[ ]:




