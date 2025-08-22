from tkinter import *
from tkinter import ttk 
from threading import Thread
import asyncio
import pandas as pd
import numpy as np
from ib_insync import IB, Stock, util
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os
import time
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

# ─── GUI Setup ───────────────────────────────────────────────────────────────
root = Tk()
root.title("Interface de Backtest IBKR")
root.geometry("700x750")


notebook = ttk.Notebook(root)
notebook.pack(fill=BOTH, expand=True, padx=10, pady=10)


config_frame = Frame(notebook)
notebook.add(config_frame, text="Configuração")

Label(config_frame, text="Capital Inicial").grid(row=0, column=0, sticky=W, padx=10)
initial_cap_entry = Entry(config_frame)
initial_cap_entry.insert(0, "100000")
initial_cap_entry.grid(row=0, column=1)

Label(config_frame, text="Tickers").grid(row=1, column=0, sticky=W, padx=10)
tickers_entry = Entry(config_frame)
tickers_entry.insert(0, "VXUS, IXUS, EFA, EEM, VEA, SCHF, IEUR, VWO, IPAC, SPDW")
tickers_entry.grid(row=1, column=1)

Label(config_frame, text="Capital usado (%)").grid(row=2, column=0, sticky=W, padx=10)
alloc_entry = Entry(config_frame)
alloc_entry.insert(0, "5")
alloc_entry.grid(row=2, column=1)

Label(config_frame, text="Stop-loss (%)").grid(row=3, column=0, sticky=W, padx=10)
sl_entry = Entry(config_frame)
sl_entry.insert(0, "3")
sl_entry.grid(row=3, column=1)

output_text = Text(config_frame, height=30, width=80, state='disabled', bg="#f0f0f0")
output_text.grid(row=5, column=0, columnspan=3, padx=10, pady=10)

def log(message):
    output_text.configure(state='normal')
    output_text.insert(END, message + "\n")
    output_text.see(END)
    output_text.configure(state='disabled')


stats_frame = Frame(notebook)
notebook.add(stats_frame, text="Estatísticas")


Label(stats_frame, text="Estatísticas do Backtest", font=("Arial", 14, "bold")).pack(pady=20)


final_cap_label = Label(stats_frame, text="Capital Final: --", font=("Arial", 12))
final_cap_label.pack(anchor=W, padx=20, pady=5)

total_trades_label = Label(stats_frame, text="Total de Trades: --", font=("Arial", 12))
total_trades_label.pack(anchor=W, padx=20, pady=5)

total_profit_label = Label(stats_frame, text="Lucro Total: --", font=("Arial", 12))
total_profit_label.pack(anchor=W, padx=20, pady=5)

gross_profit_label = Label(stats_frame, text="Lucro Bruto: --", font=("Arial", 12))
gross_profit_label.pack(anchor=W, padx=20, pady=5)

gross_loss_label = Label(stats_frame, text="Prejuízo Bruto: --", font=("Arial", 12))
gross_loss_label.pack(anchor=W, padx=20, pady=5)

sharpe_label = Label(stats_frame, text="Sharpe Ratio: --", font=("Arial", 12))
sharpe_label.pack(anchor=W, padx=20, pady=5)

greatest_win_label = Label(stats_frame, text="Maior Ganho: --", font=("Arial", 12))
greatest_win_label.pack(anchor=W, padx=20, pady=5)

greatest_loss_label = Label(stats_frame, text="Maior Perda: --", font=("Arial", 12))
greatest_loss_label.pack(anchor=W, padx=20, pady=5)

average_win_label = Label(stats_frame, text="Média dos Ganhos: --", font=("Arial", 12))
average_win_label.pack(anchor=W, padx=20, pady=5)

average_loss_label = Label(stats_frame, text="Média das Perdas: --", font=("Arial", 12))
average_loss_label.pack(anchor=W, padx=20, pady=5)


graph_frame = Frame(notebook)
notebook.add(graph_frame, text="Gráfico Capital")

graph_canvas = None

# ─── Backtest Logic ──────────────────────────────────────────────────────────
loading_label = Label(config_frame, text="", font=("Arial", 12, "italic"), fg="blue")
loading_label.grid(row=6, column=0, columnspan=2, pady=5)

loading_frames = [
    "⋅..", ".⋅.", "..⋅"
]

def run_backtest():
    asyncio.set_event_loop(asyncio.new_event_loop())
    try:
        initial_cap = float(initial_cap_entry.get())
        tickers = [x.strip().upper() for x in tickers_entry.get().split(",")]
        alloc_pct = float(alloc_entry.get()) / 100
        sl_thresh = float(sl_entry.get()) / 100

        ib = IB()
        try:
            ib.connect('127.0.0.1', 7497, clientId=1)
        except Exception as e:
            root.after(0, lambda: loading_label.config(text=""))
            log(f"❌ IB connection error: {e}")
            return

        def fetch_data(symbol, duration='14 D'):
            contract = Stock(symbol, 'SMART', 'USD', primaryExchange='ARCA')
            ib.qualifyContracts(contract)
            for _ in range(3):
                bars = ib.reqHistoricalData(contract, '', duration, '1 min', 'TRADES', True)
                if bars:
                    return util.df(bars).set_index('date')
                time.sleep(2)
            raise RuntimeError(f"Failed fetching {symbol}")

        def compute_indicators(df):
            df['vwap'] = (df['close'] * df['volume']).rolling(20).sum() / df['volume'].rolling(20).sum()
            d = df['close'].diff()
            gain = d.clip(lower=0)
            loss = -d.clip(upper=0)
            avg_gain = gain.rolling(14).mean()
            avg_loss = loss.rolling(14).mean()
            rs = avg_gain / (avg_loss + 1e-5)
            df['rsi'] = 100 - (100 / (1 + rs))
            e12 = df['close'].ewm(span=12, adjust=False).mean()
            e26 = df['close'].ewm(span=26, adjust=False).mean()
            df['macd'] = e12 - e26
            df['signal'] = df['macd'].ewm(span=9, adjust=False).mean()
            df['avg_vol'] = df['volume'].rolling(20).mean()
            return df.dropna()

        model_file = 'xgb_odte_model.pkl'
        if os.path.exists(model_file):
            model = joblib.load(model_file)
        else:
            train_df = fetch_data('SPY', '90 D')
            train_df = compute_indicators(train_df)
            train_df['future'] = train_df['close'].shift(-10)
            train_df['label'] = (train_df['future'] > train_df['close']).astype(int)
            feats = train_df[['vwap', 'rsi', 'macd', 'signal', 'avg_vol']]
            labels = train_df['label']
            Xtr, Xte, ytr, yte = train_test_split(feats, labels, test_size=0.2, shuffle=False)
            model = xgb.XGBClassifier(n_estimators=100, max_depth=3, use_label_encoder=False, eval_metric='logloss')
            model.fit(Xtr, ytr)
            joblib.dump(model, model_file)

        capital = initial_cap
        capital_history = [(None, capital)]
        TP_TIERS = [1.10, 1.25, 1.50]
        DELTA = 0.5
        logs = []
        total_trades = 0
        trade_pnls = []

        for sym in tickers:
            # Start loading animation for this ticker
            loading = {"running": True}
            def animate_loading(idx=0, ticker=sym, loading=loading):
                if not loading["running"]:
                    loading_label.config(text="")
                    return
                frame = loading_frames[idx % len(loading_frames)]
                loading_label.config(text=f"Backtesting {ticker} {frame}")
                root.after(400, animate_loading, idx+1, ticker, loading)
            root.after(0, animate_loading)

            df = fetch_data(sym, '14 D')
            df = compute_indicators(df)
            log(f"\n🔁 Backtesting {sym}")

            position = None
            targets = []
            ti = 0

            for i in range(20, len(df)):
                row = df.iloc[i]
                p = row['close']
                t = row.name

                fake = (p > row['vwap'] and df.iloc[i - 1]['close'] > p and row['volume'] > 2 * row['avg_vol'])
                base = (p > row['vwap'] and 30 < row['rsi'] < 70 and row['macd'] > row['signal']
                        and row['volume'] > 1.5 * row['avg_vol'] and not fake)
                feat = np.array([[row['vwap'], row['rsi'], row['macd'], row['signal'], row['avg_vol']]])
                aiok = model.predict(feat)[0] == 1

                if position is None and base and aiok:
                    prem = p * 0.02
                    cnt = int((capital * alloc_pct) / (prem * 100))
                    position = {'entry_t': t, 'entry_p': p, 'cnt': cnt}
                    targets = [p * x for x in TP_TIERS]
                    ti = 0
                    log(f"[ENTRY] {t} {sym} @ {p:.2f} → {cnt} contracts")
                    total_trades += 1
                    capital_history.append((t, capital))

                elif position:
                    move = p - position['entry_p']
                    pnl = move * DELTA * 100 * position['cnt']

                    if ti < len(targets) and p >= targets[ti]:
                        gain = (targets[ti] - position['entry_p']) * DELTA * 100 * (position['cnt'] // len(targets))
                        capital += gain
                        trade_pnls.append(gain)
                        log(f"[TIER{ti + 1}] {t} {sym} PnL ${gain:.2f}")
                        ti += 1
                        capital_history.append((t, capital))
                        if ti >= len(targets):
                            position = None

                    elif pnl <= -initial_cap * alloc_pct * sl_thresh:
                        capital += pnl
                        trade_pnls.append(pnl)
                        log(f"[SL] {t} {sym} PnL ${pnl:.2f}")
                        position = None
                        capital_history.append((t, capital))

                    elif (row['rsi'] < df.iloc[i - 1]['rsi'] and
                          row['macd'] < df.iloc[i - 1]['macd'] and
                          row['volume'] < row['avg_vol']):
                        capital += pnl
                        trade_pnls.append(pnl)
                        log(f"[SmartExit] {t} {sym} PnL ${pnl:.2f}")
                        position = None
                        capital_history.append((t, capital))

            # Stop loading animation for this ticker
            loading["running"] = False
            root.after(0, lambda: loading_label.config(text=""))

        log("Backtest Completo")

        capital_history_sorted = sorted(
            [(t, c) for t, c in capital_history if t is not None],
            key=lambda x: x[0]
        )

        total_profit = capital - initial_cap
        gross_profit = sum(p for p in trade_pnls if p > 0)
        gross_loss = sum(p for p in trade_pnls if p < 0)
        sharpe_ratio = 0
        if len(trade_pnls) > 1:
            mean_pnl = np.mean(trade_pnls)
            std_pnl = np.std(trade_pnls)
            sharpe_ratio = mean_pnl / std_pnl if std_pnl > 0 else 0

        wins = [p for p in trade_pnls if p > 0]
        losses = [p for p in trade_pnls if p < 0]
        greatest_win = max(wins) if wins else 0
        greatest_loss = min(losses) if losses else 0
        average_win = np.mean(wins) if wins else 0
        average_loss = np.mean(losses) if losses else 0

        def update_stats():
            final_cap_label.config(text=f"Capital Final: ${capital:,.2f}")
            total_trades_label.config(text=f"Total de Trades: {total_trades}")
            total_profit_label.config(text=f"Lucro Total: ${total_profit:,.2f}")
            gross_profit_label.config(text=f"Lucro Bruto: ${gross_profit:,.2f}")
            gross_loss_label.config(text=f"Prejuízo Bruto: ${gross_loss:,.2f}")
            sharpe_label.config(text=f"Sharpe Ratio: {sharpe_ratio:.2f}")
            greatest_win_label.config(text=f"Maior Ganho: ${greatest_win:,.2f}")
            greatest_loss_label.config(text=f"Maior Perda: ${greatest_loss:,.2f}")
            average_win_label.config(text=f"Média dos Ganhos: ${average_win:,.2f}")
            average_loss_label.config(text=f"Média das Perdas: ${average_loss:,.2f}")
        root.after(0, update_stats)

        def plot_graph():
            global graph_canvas
            for widget in graph_frame.winfo_children():
                widget.destroy()
            times = [t for t, c in capital_history_sorted]
            capitals = [c for t, c in capital_history_sorted]
            if not times or not capitals:
                Label(graph_frame, text="Sem dados para exibir.", font=("Arial", 12)).pack()
                return
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(times, capitals, marker='o')
            ax.set_title("Capital ao Longo do Tempo")
            ax.set_xlabel("")
            ax.set_xticklabels([])
            ax.set_ylabel("Capital ($)")
            ax.grid(True)
            fig.tight_layout()
            graph_canvas = FigureCanvasTkAgg(fig, master=graph_frame)
            graph_canvas.draw()
            graph_canvas.get_tk_widget().pack(fill=BOTH, expand=True)

        root.after(0, plot_graph)
        root.after(0, update_stats)

        ib.disconnect()
    except Exception as e:
        root.after(0, lambda: loading_label.config(text=""))
        log(f"❌ Error: {e}")

def start_backtest_thread():
    t = Thread(target=run_backtest)
    t.daemon = True 
    t.start()

Button(config_frame, text="Iniciar Backtest", command=start_backtest_thread, bg="#C93318", fg="white").grid(row=4, column=0, columnspan=2, pady=10)

root.mainloop()