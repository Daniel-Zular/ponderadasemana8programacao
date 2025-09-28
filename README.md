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


## Ir além: threshold ótimo, atenção e erros

1. **Threshold ótimo (F1)** → encontra o melhor limiar de decisão no conjunto de validação.  
2. **Explicabilidade por atenção** → mostra quais tokens receberam mais atenção do BERT.  
3. **Análise de erros** → lista os *false positives* e *false negatives* mais confiantes.


```python
import numpy as np
import tensorflow as tf

# ---------- (1) Threshold ótimo ----------
val_y   = val_df["label"].values.astype(int)
val_prob = model.predict(val_ds, verbose=0).ravel().astype(float)

ths = np.linspace(0.2, 0.8, 121)
best_f1, best_thr = -1.0, 0.5
for thr in ths:
    pred = (val_prob >= thr).astype(int)
    tp = np.sum((val_y==1) & (pred==1))
    fp = np.sum((val_y==0) & (pred==1))
    fn = np.sum((val_y==1) & (pred==0))
    precision = tp/(tp+fp+1e-12); recall = tp/(tp+fn+1e-12)
    f1 = 2*precision*recall/(precision+recall+1e-12)
    if f1 > best_f1:
        best_f1, best_thr = f1, thr

print(f"Threshold ótimo: {best_thr:.3f} | F1≈{best_f1:.4f}")

# ---------- (2) Explicabilidade por atenção ----------
backbone.config.output_attentions = True

def explain_with_attention(text, k_top=8, max_len=128):
    enc = tokenizer(
        [text],
        max_length=max_len,
        padding="max_length",
        truncation=True,
        return_tensors="tf",
        add_special_tokens=True
    )
    outputs = backbone(
        input_ids=enc["input_ids"],
        attention_mask=enc["attention_mask"],
        output_attentions=True,
        training=False
    )
    att = outputs.attentions[-1]
    att_mean = tf.reduce_mean(att, axis=1)
    att_cls_to_tokens = att_mean[0, 0, :]
    scores = att_cls_to_tokens.numpy()

    ids = enc["input_ids"][0].numpy().tolist()
    toks = tokenizer.convert_ids_to_tokens(ids)
    mask = enc["attention_mask"][0].numpy().astype(bool)
    toks = np.array(toks)[mask]
    scores = scores[:len(toks)]

    keep = [i for i,t in enumerate(toks) if t not in ("[CLS]","[SEP]","[PAD]")]
    toks_vis = toks[keep]
    scores_vis = scores[keep]

    top_idx = np.argsort(-scores_vis)[:min(k_top, len(scores_vis))]
    return [(toks_vis[i], float(scores_vis[i])) for i in top_idx]

sample_text = "Free iPhone!!! Click here http://spam"
print("Tokens mais relevantes:", explain_with_attention(sample_text))

# ---------- (3) Análise de erros ----------
val_pred = (val_prob >= best_thr).astype(int)
fp_idx = np.where((val_y==0) & (val_pred==1))[0]
fn_idx = np.where((val_y==1) & (val_pred==0))[0]
val_df_reset = val_df.reset_index(drop=True)

fp_sorted = fp_idx[np.argsort(-val_prob[fp_idx])]
fn_sorted = fn_idx[np.argsort( val_prob[fn_idx])]

print("\nFalse Positives mais confiantes:")
for i in fp_sorted[:5]:
    print(f"[FP] prob_bot={val_prob[i]*100:5.1f}% | {val_df_reset.loc[i,'text'][:200]}")

print("\nFalse Negatives mais confiantes:")
for i in fn_sorted[:5]:
    print(f"[FN] prob_bot={val_prob[i]*100:5.1f}% | {val_df_reset.loc[i,'text'][:200]}")

```
