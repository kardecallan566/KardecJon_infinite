# TODO - Desenvolvimento do Projeto de Reinforcement Learning

Este documento descreve as etapas concluídas e as funcionalidades implementadas no sistema.

## ✅ Concluído

1.  **Arquitetura Modular**: Organização do projeto em pastas (`agent`, `config`, `database`, `environment`, `metrics`, `scraper`).
2.  **Configuração**: Implementação de `config.yaml` para fácil ajuste de hiperparâmetros e URLs.
3.  **Banco de Dados**: Criação do `DatabaseManager` usando SQLite para persistência de resultados e predições.
4.  **Web Scraping**: Implementação do `Scraper` com suporte à estrutura HTML fornecida e modo Mock para testes.
5.  **Ambiente RL**: Desenvolvimento do `BettingEnv` com engenharia de características (frequência, sequências, alternância, etc.).
6.  **Agente DQN**: Implementação da Deep Q-Network usando PyTorch com Experience Replay e rede alvo (Target Network).
7.  **Loop Principal**: Script `main.py` que coordena a predição, coleta de dados, cálculo de recompensa e treinamento online.
8.  **Painel de Métricas**: Dashboard em tempo real usando Streamlit com gráficos de precisão e matriz de confusão.
9.  **Persistência**: Salvamento automático de pesos do modelo e histórico do banco de dados.
10. **Documentação**: Criação de `README.md` com instruções de instalação e execução.

## 🚀 Próximos Passos (Sugestões)

*   [ ] Implementar suporte a outros algoritmos de RL (ex: PPO).
*   [ ] Otimizar a extração de características com mais indicadores estatísticos.
*   [ ] Adicionar suporte a proxies no Scraper para evitar bloqueios.
*   [ ] Integrar notificações (Telegram/Discord) para alertas de predição.
