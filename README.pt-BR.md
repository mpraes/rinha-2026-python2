# Rinha 2026 Python2 - Notas de Otimizacao

Este branch contem ajustes de desempenho e resiliencia para melhorar score final no teste da Rinha 2026.

## O que foi ajustado

### 1) Balanceador (Nginx)

- Retry de upstream para erros transientes (`error`, `timeout`, `502`, `503`, `504`).
- Timeout de conexao reduzido para falhar rapido e tentar a outra API.
- Timeouts alinhados ao limite do teste (2001ms) para reduzir `http_errors`.
- Ajustes de keep-alive para conexoes mais estaveis sob carga.

Arquivos:
- `nginx.conf`

### 2) Busca vetorial (IVF)

- Busca com multiprobe (nao fica presa a um unico centroid).
- Pool de candidatos com limites configuraveis para controlar p99.
- Fallback de probes extras quando ha poucos candidatos para evitar vies de classificacao.
- Correcao no uso de `argpartition` para top-k.

Arquivos:
- `src/server.py`

### 3) Qualidade do indice offline

- Ordenacao de cada lista invertida pela distancia ao centroid durante o `pack`.
- Isso melhora a qualidade quando a lista e truncada em runtime.

Arquivos:
- `src/pack.py`

### 4) Hot path de request

- Menos alocacoes na leitura do body (`bytearray`).
- Reaproveitamento de respostas pre-serializadas para `fraud_score` discreto.

Arquivos:
- `src/server.py`

## Variaveis de tuning

Definidas no `docker-compose.yml`:

- `KNN_K=5`
- `IVF_NPROBE=3`
- `IVF_MAX_CANDIDATES=1200`
- `IVF_MAX_CANDIDATES_PER_LIST=400`

## Estrategia de calibracao recomendada

1. Rodar baseline com os valores padrao acima.
2. Se FN/FP estiverem altos, testar:
   - `IVF_NPROBE=4`
   - `IVF_MAX_CANDIDATES=1600`
   - `IVF_MAX_CANDIDATES_PER_LIST=500`
3. Se p99 subir demais, reduzir para:
   - `IVF_NPROBE=3`
   - `IVF_MAX_CANDIDATES=1000`

## Observacao importante

A mudanca em `src/pack.py` exige rebuild da imagem para regenerar o indice (`data/rinha.idx`). Sem rebuild, essa melhoria especifica nao entra em efeito.
