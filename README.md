# RL Sequence Predictor (DQN)

Este projeto utiliza Aprendizagem por Reforço Profunda (Deep Reinforcement Learning) para aprender padrões em sequências temporais e realizar previsões em tempo real.

## 📋 Funcionalidades

- **Agente DQN (PyTorch)**: Rede neural profunda que aprende a política de apostas.
- **Treinamento Online**: O modelo se atualiza continuamente com base nos resultados reais coletados.
- **Web Scraping Integrado**: Coleta automática de dados de interfaces web.
- **Banco de Dados SQLite**: Armazenamento persistente de todo o histórico e performance.
- **Dashboard em Tempo Real**: Visualização de métricas via Streamlit.

## 🛠️ Instalação

1. Certifique-se de ter o Python 3.11+ instalado.
2. Clone o repositório:
   ```bash
   git clone https://github.com/kardecallan566/KardecJon_infinite.git
   cd KardecJon_infinite
   ```
3. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Como Executar

### 1. Iniciar o Agente de Previsão
Execute o loop principal para começar a coletar dados e treinar o modelo:
```bash
python main.py
```

### 2. Iniciar o Dashboard de Métricas
Em um novo terminal, execute o Streamlit para visualizar o desempenho:
```bash
streamlit run metrics/dashboard.py
```

## ⚙️ Configuração

Edite o arquivo `config/config.yaml` para ajustar:
- `url`: A URL do site para scraping.
- `interval`: Intervalo entre as rodadas.
- `learning_rate`, `gamma`, `epsilon`: Hiperparâmetros do agente RL.

## 📁 Estrutura do Projeto

- `/agent`: Implementação da rede neural e lógica do DQN.
- `/config`: Arquivos de configuração YAML.
- `/database`: Gerenciamento do banco de dados SQLite.
- `/environment`: Lógica do ambiente e engenharia de features.
- `/metrics`: Dashboard e ferramentas de visualização.
- `/scraper`: Lógica de extração de dados web.
- `main.py`: Ponto de entrada do sistema.
