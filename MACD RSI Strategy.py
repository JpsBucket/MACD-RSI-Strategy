import yfinance as yt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)

def main():
    tickers = []

    while True:
        req = input("ticker:").strip().upper()
        data_req = yt.download(req, period = "6mo")

        if data_req.empty:
            print("data n/A")
        else: 
            tickers.append(req)
            print("added")

        q = input("Would you like to add another ticker? ").strip().lower()

        if q == "no":
            break
        elif q != "yes":
            print("not a valid answer, assuming yes")
        

        print("Final tickers:", *tickers)
    
    manipulate(tickers)



def manipulate(tickers):

    all_summaries = []

    for t in tickers:

        df = yt.download(t, period="6mo", multi_level_index=False)
        if df.empty:
            print(f"{t}: download failed, skipping")
            continue
        df = df.dropna(subset=["Close"])


        df["daily_return"] = df["Close"].pct_change()
 
        #RSI 
        df["gain"] = 0.0
        df.loc[df["daily_return"] > 0, "gain"] = df["daily_return"]
        df["lose"] = 0.0
        df.loc[df["daily_return"] < 0, "lose"] = df["daily_return"] * -1

        df["avg_gain"] = df["gain"].rolling(window = 14).mean()
        df["avg_lose"] = df["lose"].rolling(window = 14).mean()
        df["RS"] = df["avg_gain"] / df["avg_lose"]
        df["RSI"] = 100 - (100/ (1 + df["RS"]))

        #MACD

        df["EMA_fast"] = df["Close"].ewm(span = 12, adjust = False).mean()
        df["EMA_slow"] = df["Close"].ewm(span = 26, adjust = False).mean()
        df["MACD"] = df["EMA_fast"] - df["EMA_slow"]
        df["EMA_signal"] = df["MACD"].ewm(span= 9, adjust = False).mean()

        ##entry

        #rsi trigger
        df["rsi_oversold_trigger"] = (df["RSI"] < 30) & (df["RSI"].shift(1) >= 30)
        df["rsi_signal"] = np.nan
        df.loc[ df["rsi_oversold_trigger"] , "rsi_signal"] = 1
        df["watch_state"] = df["rsi_signal"].ffill(limit=10).fillna(0)

        #entry trigger
        df["entry_signal"] = np.nan
        df["macd_bullish_trigger"] = (df["MACD"] > df["EMA_signal"]) & (df["MACD"].shift(1) <= df["EMA_signal"].shift(1))
        df.loc[(df["macd_bullish_trigger"]) & (df["watch_state"] == 1 ), "entry_signal"] = 1
        df["shift_entry_signal"] = df["entry_signal"].fillna(0).shift(1)

        #Average True range
        df["range"] = (df["High"] - df["Low"])
        df["high_yeste_close"] = ((df["High"]) - (df["Close"].shift(1))).abs()
        df["low_yeste_close"] = ((df["Low"]) - (df["Close"].shift(1))).abs()
        df["true_range"] = df[["range", "high_yeste_close", "low_yeste_close"]].max(axis = 1) 
        #max obvs takes the highest value of the list but axis = 1 flips from analysing a column to a row
        df["ATR"] = df["true_range"].rolling(window = 14).mean()      

        ##Exit
        df["trade_id"] = df["shift_entry_signal"].cumsum()
        
        #breakeven
        df["entry_price"] = df.groupby("trade_id")["Close"].transform("first")
        df["breakeven_trigger"] = np.nan
        df.loc[(df["Close"] - df["entry_price"]) >= (1 * df["ATR"]), "breakeven_trigger" ] = 1
        df["breakeven_active"] = df.groupby("trade_id")["breakeven_trigger"].ffill().fillna(0)

        #trailing stop
        df["running_high"] = df.groupby("trade_id")["Close"].cummax()
        df["trailing_stop"] = df["running_high"] - (1.5 * df["ATR"])
        
        df["exit_trigger"] = np.nan
        df.loc[(df["breakeven_active"] == 1) & (df["Close"] <= df["trailing_stop"]), "exit_trigger"] = 1 
        df["shift_exit_trigger"] = df["exit_trigger"].shift(1)
        df["raw_signal"] = np.nan
        df.loc[(df["shift_entry_signal"] == 1), "raw_signal"] = 1
        df.loc[(df["shift_exit_trigger"] == 1), "raw_signal"] = 0
        df["signal"] = df["raw_signal"].ffill().fillna(0)

        ##Analysis
        df["strategy_return"] = df["signal"] * df["daily_return"]
        df["cumulative_buyhold_return"] = (1 + df["daily_return"]).cumprod()
        
    
        #this gets the total days of in or out
        days_in_position = int((df["signal"] == 1).sum())
        days_out_of_position = int((df["signal"] == 0).sum())
 
         #changed is asking if theres a difference if they dont sum to 0
        df["changed"] = df["signal"].diff().ne(0)
        df["streak"] = df["changed"].cumsum()
 
        streak_summ = df.groupby("streak").agg(
            signal_value=("signal", "first"),
            days_long=("signal", "size")
        )
 
        in_position_streaks = streak_summ[streak_summ["signal_value"] == 1]
        avg_days_in_position = float(in_position_streaks["days_long"].mean())       

        #splitting train/test

        splt = int(len(df) * 0.7)
        train_data = df.iloc[:splt]
        test_data = df.iloc[splt:]

        train_cum = (1 + train_data["strategy_return"]).cumprod()
        test_cum = (1 + test_data["strategy_return"]).cumprod()

        train_drawdown = (train_cum - train_cum.cummax()) / train_cum.cummax()
        test_drawdown = (test_cum - test_cum.cummax()) / test_cum.cummax()       

        #sharpe
        sharpe_strategy_train = (train_data["strategy_return"].mean() / train_data["strategy_return"].std()) * np.sqrt(252)
        sharpe_strategy_test = (test_data["strategy_return"].mean() / test_data["strategy_return"].std()) * np.sqrt(252)
        sharpe_buyhold_train = (train_data["daily_return"].mean() / train_data["daily_return"].std()) * np.sqrt(252)
        sharpe_buyhold_test = (test_data["daily_return"].mean() / test_data["daily_return"].std()) * np.sqrt(252)

        #cumulative 1 + the stra retrun make it cum make it iloc -1
        cumulative_strategy_train = (1 + train_data["strategy_return"]).cumprod().iloc[-1]
        cumulative_strategy_test = (1 + test_data["strategy_return"]).cumprod().iloc[-1]
        cumulative_buyhold_train = (1 + train_data["daily_return"]).cumprod().iloc[-1]
        cumulative_buyhold_test = (1 + test_data["daily_return"]).cumprod().iloc[-1]

        #drawdown - 
        max_drawdown_train = float(train_drawdown.min())
        max_drawdown_test = float(test_drawdown.min())

        daily_return = float(df["daily_return"].iloc[-1])
        RSI = float(df["RSI"].iloc[-1])
        MACD = float(df["MACD"].iloc[-1])
        EMA_signal = float(df["EMA_signal"].iloc[-1])
        position_signal = float(df["signal"].iloc[-1])
        sharpe_strategy_train = float(sharpe_strategy_train)
        sharpe_strategy_test = float(sharpe_strategy_test)
        sharpe_buyhold_train = float(sharpe_buyhold_train)
        sharpe_buyhold_test = float(sharpe_buyhold_test)
        cumulative_strategy_train = float(cumulative_strategy_train)
        cumulative_strategy_test = float(cumulative_strategy_test)
        cumulative_buyhold_train = float(cumulative_buyhold_train)
        cumulative_buyhold_test = float(cumulative_buyhold_test)
        max_drawdown_train = float(max_drawdown_train)
        max_drawdown_test = float(max_drawdown_test)

        print(df["signal"].value_counts())
        print(test_data["signal"].mean())

        summary = pd.DataFrame([{
            "t": t,
            "daily_return": daily_return,
            "RSI": RSI,
            "MACD": MACD,
            "EMA_signal": EMA_signal,
            "position_signal": position_signal,
            "cumulative_strategy_train": cumulative_strategy_train,
            "cumulative_strategy_test": cumulative_strategy_test,
            "cumulative_buyhold_train": cumulative_buyhold_train,
            "cumulative_buyhold_test": cumulative_buyhold_test,
            "sharpe_strategy_train": sharpe_strategy_train,
            "sharpe_strategy_test": sharpe_strategy_test,
            "sharpe_buyhold_train": sharpe_buyhold_train,
            "sharpe_buyhold_test": sharpe_buyhold_test,
            "max_drawdown_train": max_drawdown_train,
            "max_drawdown_test": max_drawdown_test,
            "days_in_position": days_in_position,
            "days_out_of_position": days_out_of_position,
            "avg_days_in_position": avg_days_in_position,
        }])

        all_summaries.append(summary)

    combined = pd.concat( all_summaries, ignore_index=True)
    print(combined)

if __name__ == "__main__":
    main()