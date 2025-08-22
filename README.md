# 📈 Interface de Backtest IBKR

Uma aplicação em Python com interface gráfica para realizar backtests automatizados de estratégias de trading utilizando dados da Interactive Brokers (IBKR) e Machine Learning.

## 🎯 O que faz esta aplicação?

Esta ferramenta permite testar estratégias de trading em dados históricos reais, simulando como seus investimentos teriam performado no passado. Ela usa:

- **Dados reais** da Interactive Brokers
- **Inteligência Artificial** (XGBoost) para prever movimentos do mercado
- **Indicadores técnicos** como RSI, MACD e VWAP
- **Interface visual** para acompanhar resultados

## ✨ Principais funcionalidades

### 📊 **Análise Técnica Automática**
- Calcula indicadores como RSI, MACD, VWAP automaticamente
- Detecta padrões de volume anômalo
- Identifica pontos de entrada e saída

### 🤖 **Machine Learning**
- Treina um modelo XGBoost para prever direção dos preços
- Usa dados históricos do SPY para aprendizado
- Salva o modelo treinado para reutilização

### 💰 **Gestão de Risco**
- Sistema de stop-loss configurável
- Múltiplos níveis de take-profit (1.10x, 1.25x, 1.50x)
- Controle de alocação de capital por trade

### 📈 **Visualização Completa**
- Interface com abas organizadas
- Estatísticas detalhadas de performance
- Gráfico da evolução do capital
- Log em tempo real das operações

## 🛠️ Pré-requisitos

### Software necessário:
1. **Python 3.7+** instalado no seu computador
2. **TWS (Trader Workstation)** ou **IB Gateway** da Interactive Brokers
3. Conta na Interactive Brokers (pode ser demo)

### Configuração da Interactive Brokers:
1. Abra o TWS ou IB Gateway
2. Vá em **Configurações → API → Configurações**
3. Marque **"Enable ActiveX and Socket Clients"**
4. Adicione **127.0.0.1** à lista de IPs confiáveis
5. Configure a porta **7497** (paper trading) ou **7496** (live)

## 📦 Instalação

1. **Clone ou baixe este projeto**
```bash
git clone [seu-repositorio]
cd interface-backtest-ibkr
```

2. **Instale as dependências**
```bash
pip install -r requirements.txt
```

3. **Execute a aplicação**
```bash
python main.py
```

## 🚀 Como usar

### 1. **Aba Configuração**
- **Capital Inicial**: Quanto dinheiro você quer simular (ex: 100000)
- **Tickers**: Ações para testar, separadas por vírgula (ex: VXUS, IXUS, EFA)
- **Capital usado (%)**: Percentual do capital usado por trade (ex: 5%)
- **Stop-loss (%)**: Limite de perda por trade (ex: 3%)

### 2. **Iniciar Backtest**
- Certifique-se que o TWS/Gateway está rodando
- Clique em **"Iniciar Backtest"**
- Acompanhe o progresso no log

### 3. **Visualizar Resultados**
- **Aba Estatísticas**: Veja métricas de performance
- **Aba Gráfico Capital**: Visualize a evolução do seu capital

## 📊 Entendendo os resultados

### Estatísticas importantes:
- **Capital Final**: Valor final após todos os trades
- **Lucro Total**: Ganho ou perda líquida
- **Sharpe Ratio**: Relação risco/retorno (quanto maior, melhor)
- **Maior Ganho/Perda**: Extremos dos seus trades

### O que significam:
- **Sharpe > 1**: Estratégia interessante
- **Sharpe > 2**: Estratégia muito boa
- **Lucro Total positivo**: Estratégia lucrativa no período

## ⚙️ Como funciona a estratégia

### Condições para COMPRA:
1. Preço acima da VWAP (tendência de alta)
2. RSI entre 30-70 (não sobrecomprado/sobrevendido)
3. MACD acima do sinal (momentum positivo)
4. Volume 1.5x acima da média
5. IA prevê movimento de alta
6. Não é um "fake pump" (movimento exagerado)

### Condições para VENDA:
1. **Take Profit**: 3 níveis (10%, 25%, 50% de ganho)
2. **Stop Loss**: Perda máxima configurada
3. **Smart Exit**: Deterioração dos indicadores técnicos
