# Churn Prediction & Retention Copilot


> **Status do Projeto:** Em desenvolvimento — *Fase 1: Planejamento arquitetural e estruturação do ambiente.*

Sistema ponta a ponta de predição de cancelamento (*Churn*) e recomendação automatizada de retenção, combinando **engenharia de atributos em SQL**, **modelagem tabular calibrada (LightGBM)**, **explicabilidade local (SHAP)**, **API REST (FastAPI)** e um **Agente Copiloto Generativo** guiado por políticas comerciais.

---

## 1. Introdução e Contexto

Em operações digitais e de e-commerce, projetos tradicionais de Churn costumam parar na classificação binária (`0` ou `1`) otimizada apenas para acurácia ou F1-Score. Na prática, essa abordagem gera dois gargalos operacionais:

1. **Desperdício de Orçamento:** Acionar campanhas promocionais para clientes que possuem alta probabilidade de permanecer organicamente (ou clientes irrecuperáveis) prejudica a margem da operação.
2. **Falta de Acionabilidade no Atendimento:** Saber que um cliente tem "78% de risco de saída" não diz por que ele está saindo nem qual oferta faz sentido para o seu perfil.

### Objetivo do Projeto
Construir uma arquitetura que estime **probabilidades reais (calibradas)** de Churn, calcule o **Retorno sobre Investimento (ROI) esperado** da intervenção e acione um agente capaz de redigir abordagens personalizadas com base nos fatores individuais de risco de cada cliente.

---

## 2. Arquitetura Planejada

O fluxo de dados será dividido em quatro camadas principais:

```text
[Dados Brutos (Kaggle)]
       │
       ▼
[Camada de Dados: SQLite + SQL]
  └── Ingestão e extração de métricas comportamentais (CTEs e razões RFM)
       │
       ▼
[Camada de ML & Explicabilidade: Scikit-Learn + LightGBM + SHAP]
  ├── Modelo Calibrado (Sigmoid/Isotonic) ──► Probabilidade real de Churn
  └── TreeExplainer (SHAP) ──────────────────► Top-3 fatores locais de risco
       │
       ▼
[Camada de Serviço: FastAPI]
  └── Endpoint POST /predict-churn ──────────► Cálculo de ROI + Decisão de Ação
       │
       ▼
[Camada Conversacional: Agente de Retenção (LLM + Políticas)]
  └── Diagnóstico interno para CS + Mensagem personalizada para o cliente
```

---

## 3. Metodologia

* **Feature Engineering na Camada SQL:** As transformações comportamentais (frequência de pedidos por tempo de casa, razão de reclamações por compra, recência relativa e flags de atrito crítico) serão escritas em SQL antes da etapa de modelagem em Python, simulando um ambiente analítico de produção.
* **Calibração de Probabilidades:** Como a decisão de conceder ou não um benefício financeiro depende diretamente do valor esperado ($\text{ROI} = P(\text{Churn}) \times \text{LTV} \times \text{Uplift} - \text{Custo}$), o classificador passará por calibração (`CalibratedClassifierCV`). A avaliação utilizará **Brier Score**, **Log-Loss** e **PR-AUC** em vez de apenas acurácia.
* **Explicabilidade Local com SHAP:** Cada predição retornará os principais atributos que elevaram o risco daquele cliente específico, no sentido de evitar respostas genéricas.
* **Agente Orientado por Políticas (*Guardrails*):** O módulo generativo receberá o JSON estruturado da API e consultará um documento explícito de regras comerciais (`policies.md`), garantindo que nenhum desconto seja oferecido quando o ROI esperado for negativo e que termos técnicos (como "SHAP" ou "score") nunca vazem para o cliente final.

---

## 4. Estrutura Planejada do Repositório

```text
churn-prediction-retention-copilot/
├── data/
│   ├── raw/                    # Dataset original (ignorado no versionamento)
│   └── churn.db                # Banco SQLite gerado na etapa de ingestão
├── sql/
│   └── feature_engineering.sql # Queries analíticas para criação de variáveis
├── src/
│   ├── features/
│   │   └── build_features.py   # Pipeline de ingestão e leitura das features SQL
│   ├── models/
│   │   └── train.py            # Pré-processamento, treino, calibração e SHAP
│   ├── api/
│   │   ├── main.py             # Aplicação FastAPI e regras de decisão de ROI
│   │   └── schemas.py          # Contratos de entrada/saída via Pydantic
│   └── agent/
│       ├── retention_bot.py    # Orquestração do agente copiloto de retenção
│       └── policies.md         # Regras comerciais e tom de voz por perfil de risco
├── tests/
│   └── test_api.py             # Testes automatizados de contrato e inferência
├── artifacts/                  # Modelos e transformadores serializados (.joblib)
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 5. Roadmap

- [x] Definição da arquitetura, escopo de negócio e estrutura inicial do repositório
- [ ] **Etapa 1 — Dados & SQL:** Ingestão do dataset *E-Commerce Customer Churn* (Kaggle) no SQLite e construção das queries de *feature engineering*
- [ ] **Etapa 2 — Modelagem & Calibração:** Construção do pipeline Scikit-Learn/LightGBM, comparação de métricas pré e pós-calibração e integração do `shap.TreeExplainer`
- [ ] **Etapa 3 — API de Inferência:** Implementação dos endpoints `/health` e `/predict-churn` no FastAPI com cálculo de ROI esperado
- [ ] **Etapa 4 — Agente de Retenção:** Criação da base de políticas comerciais (`policies.md`) e integração do agente para geração de diagnóstico e mensagem personalizada
- [ ] **Etapa 5 — Qualidade & Empacotamento:** Escrita dos testes unitários/contrato com `pytest`, criação do `Dockerfile` e documentação dos resultados finais obtidos

---

## 6. Tecnologias

* **Linguagem & Manipulação de Dados:** Python 3.15.2, Pandas, NumPy, SQLite
* **Machine Learning & Explicabilidade:** Scikit-Learn, LightGBM, SHAP, Joblib
* **API & Validação:** FastAPI, Uvicorn, Pydantic
* **GenAI & Agentes:** Google GenAI SDK (`gemini-3.5-flash`)
* **Testes & Qualidade:** Pytest, HTTPX

---

## 7. Configuração do Ambiente

```bash
# Clonar o repositório
git clone https://github.com/matheus-naves47/churn-retention-copilot.git
cd churn-retention-copilot

# Criar e ativar ambiente virtual
python -m venv .venv
source .venv/bin/activate

# Instalar dependências
pip install -r requirements.txt
```