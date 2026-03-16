from __future__ import annotations


def build_demo_docs_and_qa():
    """
    Mini base local sintética para testar:
    - baseline context stuffing
    - baseline RAG simples
    - pipeline sem RAG com leitura recursiva
    """

    docs = {
        "DOC_PHI4": {
            "title": "Phi-4 resumo técnico",
            "text": (
                "Phi-4 é um modelo de linguagem com 14 bilhões de parâmetros. "
                "Ele foi projetado com foco em qualidade de dados e uso extensivo de dados sintéticos.\n\n"
                "No catálogo local descrito para estes experimentos, Phi-4 aparece com janela de contexto de 16k tokens.\n\n"
                "O modelo também passa por refinamentos de instrução para melhorar aderência a comandos e respostas mais úteis."
            ),
        },
        "DOC_OLLAMA_API": {
            "title": "Ollama API local",
            "text": (
                "A API /api/chat do Ollama retorna informações úteis para medição, incluindo "
                "prompt_eval_count e eval_count, além de durations relacionadas à inferência.\n\n"
                "A API /api/generate também retorna campos como total_duration, load_duration, "
                "prompt_eval_duration e eval_duration, o que ajuda a decompor latência.\n\n"
                "A API /api/embed permite gerar embeddings locais e também retorna informações úteis sobre o processamento."
            ),
        },
        "DOC_TOT": {
            "title": "Tree of Thoughts explicação",
            "text": (
                "Tree of Thoughts generaliza Chain-of-Thought ao permitir explorar múltiplos caminhos de raciocínio "
                "antes de decidir a resposta final.\n\n"
                "A ideia central é usar busca, como BFS ou DFS, além de autoavaliação, para decidir quais caminhos "
                "devem ser expandidos e quais devem ser descartados.\n\n"
                "Esse processo pode melhorar qualidade em tarefas difíceis, mas também aumenta custo e latência."
            ),
        },
        "DOC_RLM_BUDGETMEM": {
            "title": "Recursive reading e memória seletiva",
            "text": (
                "Recursive reading pode ser entendido como um ciclo no qual o sistema lê um trecho, extrai evidências "
                "e decide se precisa abrir outro trecho do documento.\n\n"
                "A ideia associada a Recursive Language Models é tratar partes do contexto como um ambiente externo, "
                "em vez de tentar empurrar tudo de uma vez para o prompt.\n\n"
                "Uma memória seletiva estilo BudgetMem prioriza retenção do que tem mais valor para a pergunta atual, "
                "controlando custo de contexto e evitando carregar informação irrelevante."
            ),
        },
        "DOC_RAG": {
            "title": "RAG em uma frase",
            "text": (
                "RAG é uma abordagem em que primeiro se recuperam trechos relevantes de uma base externa "
                "e depois o modelo gera uma resposta condicionada a esses trechos.\n\n"
                "A principal vantagem do RAG é grounding com custo relativamente previsível, "
                "mas ele depende bastante da qualidade da recuperação."
            ),
        },
    }

    qa = [
        {
            "id": "q1",
            "question": "Qual é a janela de contexto do Phi-4 no catálogo descrito?",
            "answer": "16k tokens",
            "gold_citations": ["DOC_PHI4#p1"],
        },
        {
            "id": "q2",
            "question": "Quais campos de contagem de tokens a API /api/chat retorna segundo o documento?",
            "answer": "prompt_eval_count e eval_count",
            "gold_citations": ["DOC_OLLAMA_API#p0"],
        },
        {
            "id": "q3",
            "question": "Cite duas durations da API /api/generate que ajudam a decompor latência.",
            "answer": "total_duration e load_duration",
            "gold_citations": ["DOC_OLLAMA_API#p1"],
        },
        {
            "id": "q4",
            "question": "Em que sentido Tree of Thoughts generaliza Chain-of-Thought?",
            "answer": "Explora múltiplos caminhos de raciocínio antes da resposta final",
            "gold_citations": ["DOC_TOT#p0", "DOC_TOT#p1"],
        },
        {
            "id": "q5",
            "question": "Explique em uma frase o papel da memória seletiva estilo BudgetMem.",
            "answer": "Priorizar retenção do que mais importa para a pergunta atual, controlando custo de contexto",
            "gold_citations": ["DOC_RLM_BUDGETMEM#p2"],
        },
    ]

    return docs, qa