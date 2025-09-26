# Twitter Bot Detection com BERT 

Documento para acompanhar o notebook de detecção de bots no Twitter usando **BERT** com **TensorFlow/Keras** e **Hugging Face**. O foco é ter um passo a passo claro, com poucas citações de código para situar.



(Para mim, está dando erro ao tentar abrir o notebook pelo GitHub. Talvez seja interessante baixar e abrir no VSCode)
---

## Objetivo

Treinar um classificador que diferencia **bot** de **humano** a partir de texto (ex.: tweet/bio), usando um backbone BERT e uma cabeça densa binária.

---

## Estrutura geral do pipeline

1. **Setup** (instala libs, fixa versões e semente)  
2. **Dados** (carrega CSV, escolhe colunas de texto e rótulo)  
3. **Limpeza** (normaliza rótulo em `0/1` e remove nulos)  
4. **EDA** (contagem de classes e comprimento dos textos)  
5. **Split** (treino/val/test estratificado)  
6. **Tokenização** (BERT `bert-base-uncased`, `max_length=128`)  
7. **Modelo** (BERT + `Dropout` + `Dense(sigmoid)`)  
8. **Treino** (callbacks, early stopping, checkpoints)  
9. **Avaliação** (accuracy, F1, AUC, matriz de confusão, ROC/PR)  
10. **Persistência** (modelo, tokenizer e métricas)  
11. **Inferência** (função para pontuar novos textos)  

---

## 1) Setup

Dependências principais:
- `transformers` (Hugging Face)  
- `tensorflow`/`keras` (Treino)  
- `tokenizers`, `safetensors`  
- `scikit-learn` (apenas onde compatível; algumas métricas foram reimplementadas em NumPy)  
- `matplotlib`  

Para compatibilidade com **Python 3.12** no Colab:

```bash
pip install -U scikit-learn==1.5.1 transformers==4.41.2 safetensors==0.4.3
```

Semente e diretórios:

```python
SEED = 42
ART_DIR = "/content/artifacts"
```

---

## 2) Dados

O notebook trabalha com um CSV em `/content/data`. As colunas podem variar entre versões do dataset; por isso, há uma detecção automática de candidatos às colunas texto e rótulo. No caso usado, foi definido:

```python
TEXT_COL  = "Tweet"
LABEL_COL = "Bot Label"
```

O rótulo é mapeado para `0 = humano`, `1 = bot`. Linhas com texto vazio ou rótulo inválido são descartadas.

---

## 3) EDA

Checagens rápidas:
- Distribuição de classes (para saber se precisa de `class_weight`)  
- Comprimento dos textos (para escolher `max_length`)  

Gráficos simples ajudam a ver se há desbalanceamento ou textos muito curtos/longos. Exemplo de estatísticas: `value_counts()` por classe e `describe()` do comprimento (tokens por espaço).

---

## 4) Split

Split estratificado 80/10/10 (treino/val/test), garantindo que as proporções de classes se mantenham.  
Onde `scikit-learn` não foi estável, foi usado split estratificado em NumPy (mesma ideia do `train_test_split` com `stratify`).

---

## 5) Tokenização

Modelo e tokenizer:

```python
MODEL_NAME = "bert-base-uncased"
MAX_LEN    = 128
```

A tokenização usa `padding` e `truncation` no tamanho fixo (128). Os tensores são empacotados em `tf.data.Dataset` para treino/val/test.

---

## 6) Modelo

Backbone BERT em TensorFlow + cabeça densa binária. Camada Dropout de 0.2 antes da saída:

```python
x = Dropout(0.2)(cls_token)
logits = Dense(1, activation="sigmoid")(x)
```

Métricas durante o treino incluem `acc`, `auc` e uma F1 binária simples.

Observação: por questões de compatibilidade do Keras 3 com o Transformers, foi necessário um envolvimento do forward em camadas utilitárias.  
Em alguns cenários, isso pode manter o backbone congelado na prática.  
Se a ideia for ajustar os pesos do BERT, vale migrar para `TFBertForSequenceClassification` e treinar a cabeça nativa do modelo (serializa e carrega mais “liso”).

---

## 7) Treinamento

Callbacks usados:

```python
EarlyStopping(monitor="val_auc", patience=2, mode="max", restore_best_weights=True)
ModelCheckpoint(".../bert_botdet_best.keras", monitor="val_auc", mode="max", save_best_only=True)
```

Dicas práticas:
- Para validar rápido: limitar `steps` (`take()`), diminuir `EPOCHS` e ativar GPU.  
- Se houver desbalanceamento de classes: `class_weight` ajuda.  
- `mixed_precision` pode acelerar em GPUs modernas (opcional).  

---

## 8) Avaliação

Conjunto de teste:
- Accuracy, Precision, Recall, F1, AUC-ROC  
- Matriz de confusão  
- Curvas ROC e Precision-Recall  

Como algumas versões do scikit-learn deram conflito com Py3.12, parte das métricas foi feita em NumPy.  

Se o AUC ficar próximo de 0.5, significa que o modelo não está separando bem as classes.  
Isso pode acontecer se o backbone não estiver sendo ajustado, se o dataset for difícil/sintético, ou se o texto estiver muito curto para o BERT aprender.

---

## 9) Persistência

Modelo no formato Keras nativo (`.keras`):

```python
model.save("/content/artifacts/model/bert_botdet_best.keras")
```

Tokenizer em `artifacts/tokenizer/`:

```python
tokenizer.save_pretrained(TOK_DIR)
```

Métricas em `artifacts/metrics.json`.

SavedModel: com Keras 3 + Lambda, a serialização pode falhar.  
O caminho robusto é salvar pesos (`.weights.h5`) e reconstruir a arquitetura para `load_weights` em uma nova sessão, ou migrar para `TFBertForSequenceClassification` e usar `save_pretrained`.

---

## 10) Inferência

Duas formas:

a) No runtime, com o modelo carregado na memória:

```python
probs = model.predict({"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}).ravel()
preds = (probs >= 0.5).astype(int)
```

b) Em nova sessão, reconstruindo a arquitetura e usando `load_weights`  
Recria o backbone e a cabeça do mesmo jeito e chama `re_model.load_weights(...)`.  
Depois, usa a mesma função de tokenização para prever.

Exemplo de saída legível:

```
[BOT]    prob_bot=54.9% | Free iPhone!!! Click here http://spam
[HUMAN]  prob_bot=49.1% | Olá! Sou um estudante de engenharia e adoro nadar.
```

---

## 11) Ajuste de limiar (threshold)

O limiar padrão é `0.5`.  
Em alguns cenários, vale ajustar com base no conjunto de validação para maximizar F1:

- Mais conservador para “bot”: aumentar threshold (ex.: `0.6–0.7`)  
- Mais sensível para “bot”: diminuir threshold (ex.: `0.4`)  

---

## 12) Problemas comuns e soluções rápidas

- **scikit-learn quebrando no Py3.12**  
  Fixar `scikit-learn==1.5.x`.  

- **transformers + safetensors dando erro de safe_open**  
  Usar `transformers==4.41.2` e `safetensors==0.4.3`, e forçar pesos TF com `from_pt=False`.  

- **Keras 3 não carregando modelo com Lambda**  
  Preferir `SavedModel` (quando funcionar) ou salvar pesos e reconstruir a arquitetura.  
  Alternativa melhor: `TFBertForSequenceClassification + save_pretrained`.  
'''
