# rinha-2026-python2

Implementacao da minha solucao para a Rinha de Backend 2026 (detector de fraude com busca vetorial), com foco em equilibrio entre qualidade de deteccao e latencia sob limite de 1 CPU e 350 MB.

[Português](#portugues) · [English](#english)

## Portugues

### Repositorio desta solucao

- Codigo da solucao: branch `submission`
- Metadados da participacao: [info.json](info.json)
- Imagem publicada: `docker.io/rmoraes4/rinha-fraud-2026:latest`

### Arquitetura

- 2 instancias da API Python + 1 Nginx como load balancer
- API em ASGI/Uvicorn com vetorizacao de 14 dimensoes
- Indice IVF com vetores quantizados em int8, gerado offline (`src/pack.py` no branch `submission`)

### O que foi implementado/otimizado

- Busca IVF multiprobe (nao limitada a 1 centroid)
- Pool de candidatos configuravel para ajustar recall vs p99
- Fallback de probes extras quando candidatos < K
- Correcao de selecao top-k
- Menos alocacoes no hot path de request
- Respostas pre-serializadas para scores discretos
- Ordenacao das listas invertidas por distancia ao centroid durante o build do indice
- Hardening do Nginx com retry de upstream e timeouts ajustados ao timeout do teste

### Parametros de tuning usados

- `KNN_K=5`
- `IVF_NPROBE=3`
- `IVF_MAX_CANDIDATES=1200`
- `IVF_MAX_CANDIDATES_PER_LIST=400`

### Resultado de teste (producao tipo Docs)

- Total: 54.100 requests
- FP: 496
- FN: 571
- HTTP errors: 315
- Failure rate: 2,92%
- p99: 1586,36 ms
- Final score: -177,11

### Leitura do resultado

- HTTP errors tiveram impacto alto no score de deteccao
- FN ainda acima do ideal para o peso da formula
- p99 ficou em faixa que penaliza fortemente o componente de latencia

### Proximos passos

- Reduzir HTTP errors com failover mais agressivo e timeout budget estrito
- Subir recall sem estourar p99 (ajuste fino de `IVF_NPROBE` e budgets)
- Rodar bateria curta de benchmark para encontrar ponto de melhor score final

## English

### About this repository

- This is my implementation repository for Rinha 2026 fraud detection
- Solution runtime branch: `submission`
- Participation metadata: [info.json](info.json)
- Published image: `docker.io/rmoraes4/rinha-fraud-2026:latest`

### Architecture

- 2 Python API instances + 1 Nginx load balancer
- ASGI/Uvicorn API with 14-d vectorization
- Offline-built IVF index with int8 quantization (`src/pack.py` on `submission`)

### Implemented optimizations

- Multi-probe IVF search
- Configurable candidate budget (recall vs latency tuning)
- Extra-probe fallback when candidates < K
- Top-k selection fix
- Reduced hot-path allocations
- Pre-serialized responses for discrete fraud scores
- Inverted lists sorted by centroid distance at index build time
- Nginx upstream retry and timeout hardening

### Latest production-like run

- 54,100 requests
- FP: 496
- FN: 571
- HTTP errors: 315
- Failure rate: 2.92%
- p99: 1586.36 ms
- Final score: -177.11

### Current focus

- Cut HTTP errors first
- Improve FN/FP with tuned IVF probing budget
- Lower p99 tail without sacrificing detection quality
