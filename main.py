#!/usr/bin/env python
# coding: utf-8

# # Desafio 3
# 
# Neste desafio, iremos praticar nossos conhecimentos sobre distribuições de probabilidade. Para isso,
# dividiremos este desafio em duas partes:
#     
# 1. A primeira parte contará com 3 questões sobre um *data set* artificial com dados de uma amostra normal e
#     uma binomial.
# 2. A segunda parte será sobre a análise da distribuição de uma variável do _data set_ [Pulsar Star](https://archive.ics.uci.edu/ml/datasets/HTRU2), contendo 2 questões.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Setup_ geral

# In[ ]:





# In[6]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sct
import seaborn as sns
from statsmodels.distributions.empirical_distribution import ECDF


# In[4]:



from IPython.core.pylabtools import figsize
from IPython import get_ipython

# %matplotlib inline

figsize(12, 8)

sns.set()

    
#%matplotlib inline

#from IPython.core.pylabtools import figsize


#figsize(12, 8)

#sns.set()


# ## Parte 1

# ### _Setup_ da parte 1

# In[10]:


np.random.seed(42)
    
dataframe = pd.DataFrame({"normal": sct.norm.rvs(20, 4, size=10000),
                     "binomial": sct.binom.rvs(100, 0.2, size=10000)})


# ## Inicie sua análise a partir da parte 1 a partir daqui

# In[11]:


# Sua análise da parte 1 começa aqui.

dataframe.head()
sns.distplot(dataframe.normal)
plt.hist(dataframe.normal)


# In[12]:


norm = dataframe.normal.sort_values()
binom = dataframe.binomial.sort_values()


# Definindo quartil

# In[44]:



norm_q1 = (norm[2499] + norm[2500])/2
norm_q2 = (norm[4999] + norm[5000])/2
norm_q3 = (norm[7499] + norm[7500])/2

binom_q1 = (binom[2499] + binom[2500])/2
binom_q2 = (binom[4999] + binom[5000])/2
binom_q3 = (binom[7499] + binom[7500])/2


# In[26]:


a = norm.std()
b = norm.mean()
c = norm.var()


# ## Questão 1
# 
# Qual a diferença entre os quartis (Q1, Q2 e Q3) das variáveis `normal` e `binomial` de `dataframe`? Responda como uma tupla de três elementos arredondados para três casas decimais.
# 
# Em outra palavras, sejam `q1_norm`, `q2_norm` e `q3_norm` os quantis da variável `normal` e `q1_binom`, `q2_binom` e `q3_binom` os quantis da variável `binom`, qual a diferença `(q1_norm - q1 binom, q2_norm - q2_binom, q3_norm - q3_binom)`?

# In[127]:


def q1():
    q1 = round(norm.quantile(0.25)-binom.quantile(0.25),3)
    q2 = round(norm.quantile(0.5)-binom.quantile(0.5),3)
    q3 = round(norm.quantile(0.75)-binom.quantile(0.75),3)
    #return ("%.3f, %.3f, %.3f" %(norm_q1 - binom_q1, norm_q2 - binom_q2, norm_q3 - binom_q3))
    return (q1, q2, q3)
    # Retorne aqui o resultado da questão 1.
    pass


# Para refletir:
# 
# * Você esperava valores dessa magnitude?
# 
# * Você é capaz de explicar como distribuições aparentemente tão diferentes (discreta e contínua, por exemplo) conseguem dar esses valores?

# ## Questão 2
# 
# Considere o intervalo $[\bar{x} - s, \bar{x} + s]$, onde $\bar{x}$ é a média amostral e $s$ é o desvio padrão. Qual a probabilidade nesse intervalo, calculada pela função de distribuição acumulada empírica (CDF empírica) da variável `normal`? Responda como uma único escalar arredondado para três casas decimais.

# In[55]:


def q2():
    m = norm.mean()
    s = norm.std()
    out = ECDF(norm)(b+a) - ECDF(norm)(b-a)
    return  round(out,3)
    # Retorne aqui o resultado da questão 2.
    pass


# Para refletir:
# 
# * Esse valor se aproxima do esperado teórico?
# * Experimente também para os intervalos $[\bar{x} - 2s, \bar{x} + 2s]$ e $[\bar{x} - 3s, \bar{x} + 3s]$.

# ## Questão 3
# 
# Qual é a diferença entre as médias e as variâncias das variáveis `binomial` e `normal`? Responda como uma tupla de dois elementos arredondados para três casas decimais.
# 
# Em outras palavras, sejam `m_binom` e `v_binom` a média e a variância da variável `binomial`, e `m_norm` e `v_norm` a média e a variância da variável `normal`. Quais as diferenças `(m_binom - m_norm, v_binom - v_norm)`?

# In[135]:


def q3():
    mean = round(binom.mean() - norm.mean(),3)
    std = round(binom.var() - norm.var(),3)
    
    return (mean,std)
    # Retorne aqui o resultado da questão 3.
    pass


# Para refletir:
# 
# * Você esperava valore dessa magnitude?
# * Qual o efeito de aumentar ou diminuir $n$ (atualmente 100) na distribuição da variável `binomial`?

# ## Parte 2

# ### _Setup_ da parte 2

# In[56]:


stars = pd.read_csv("pulsar_stars.csv")

stars.rename({old_name: new_name
              for (old_name, new_name)
              in zip(stars.columns,
                     ["mean_profile", "sd_profile", "kurt_profile", "skew_profile", "mean_curve", "sd_curve", "kurt_curve", "skew_curve", "target"])
             },
             axis=1, inplace=True)

stars.loc[:, "target"] = stars.target.astype(bool)


# ## Inicie sua análise da parte 2 a partir daqui

# In[260]:


# Sua análise da parte 2 começa aqui.
stars_n = stars.mean_profile[stars.target == 0]
false_pulsar_mean_profile_standardized = (stars_n-stars_n.mean())/stars_n.std()
q_08 = sct.norm.ppf(0.8)
q_09 = sct.norm.ppf(0.9)
q_095 = sct.norm.ppf(0.95)
q_08_ecdf = ECDF(false_pulsar_mean_profile_standardized)(q_08)
q_09_ecdf = ECDF(false_pulsar_mean_profile_standardized)(q_09)
q_095_ecdf = ECDF(false_pulsar_mean_profile_standardized)(q_095)
false_pulsar_mean_profile_standardized.sort_values()


# ## Questão 4
# 
# Considerando a variável `mean_profile` de `stars`:
# 
# 1. Filtre apenas os valores de `mean_profile` onde `target == 0` (ou seja, onde a estrela não é um pulsar).
# 2. Padronize a variável `mean_profile` filtrada anteriormente para ter média 0 e variância 1.
# 
# Chamaremos a variável resultante de `false_pulsar_mean_profile_standardized`.
# 
# Encontre os quantis teóricos para uma distribuição normal de média 0 e variância 1 para 0.80, 0.90 e 0.95 através da função `norm.ppf()` disponível em `scipy.stats`.
# 
# Quais as probabilidade associadas a esses quantis utilizando a CDF empírica da variável `false_pulsar_mean_profile_standardized`? Responda como uma tupla de três elementos arredondados para três casas decimais.

# In[94]:


def q4():
    
    return (round(q_08_ecdf,3), round(q_09_ecdf,3), round(q_095_ecdf,3))
    # Retorne aqui o resultado da questão 4.
    pass


# Para refletir:
# 
# * Os valores encontrados fazem sentido?
# * O que isso pode dizer sobre a distribuição da variável `false_pulsar_mean_profile_standardized`?

# ## Questão 5
# 
# Qual a diferença entre os quantis Q1, Q2 e Q3 de `false_pulsar_mean_profile_standardized` e os mesmos quantis teóricos de uma distribuição normal de média 0 e variância 1? Responda como uma tupla de três elementos arredondados para três casas decimais.

# In[98]:


def q5():
    ppf_25 = sct.norm.ppf(0.25, loc=0, scale=1)
    ppf_5 = sct.norm.ppf(0.5, loc=0, scale=1)
    ppf_75 = sct.norm.ppf(0.75, loc=0, scale=1)
    
    r_25 = false_pulsar_mean_profile_standardized.quantile(0.25) - ppf_25
    r_5 = false_pulsar_mean_profile_standardized.quantile(0.5) - ppf_5
    r_75 = false_pulsar_mean_profile_standardized.quantile(0.75) - ppf_75
    
    return (round(r_25,3), round(r_5,3), round(r_75,3))
    # Retorne aqui o resultado da questão 5.
    pass


# Para refletir:
# 
# * Os valores encontrados fazem sentido?
# * O que isso pode dizer sobre a distribuição da variável `false_pulsar_mean_profile_standardized`?
# * Curiosidade: alguns testes de hipóteses sobre normalidade dos dados utilizam essa mesma abordagem.
