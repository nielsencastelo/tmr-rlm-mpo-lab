# TMR-RLM-MPO Lab

Ambiente de pesquisa e experimentação para comparar estratégias de QA sobre corpora legislativos mutáveis.

## Objetivo
Comparar:
- Baseline de contexto direto
- RAG híbrido
- RLM simples
- TMR-RLM-MPO

## Principais ideias
- Recursive Language Models
- Memória transativa entre agentes
- Quadro compartilhado de evidências
- Meta-Prompt Orchestrator
- Rollback e checkpoints

## Estrutura
- `src/baselines`: métodos de comparação
- `src/tmr_rlm_mpo`: arquitetura principal
- `configs/`: configurações dos experimentos
- `experiments/`: execuções versionadas
- `results/`: métricas, logs e relatórios
- `docs/`: proposta, revisão e artigos

## Como rodar
```bash
python scripts/run_baseline.py
python scripts/run_tmr_rlm_mpo.py
python scripts/evaluate.py