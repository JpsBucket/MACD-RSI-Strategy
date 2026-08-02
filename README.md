# RSI + MACD Strategy

## Overview/Motivation

A Python backtesting framework for evaluating a combined RSI and MACD confirmation strategy against a buy-and-hold control, with train/test validation to check performance on unseen data.

Built as the third project in this series, to explore a confluence-based entry (two separate indicators agreeing before a trade is taken) alongside a dynamic, volatility-based exit, rather than the fixed exit conditions used in the first two projects. It demonstrates applied financial analysis skills using metrics such as Sharpe ratio and maximum drawdown, for quant-focused roles.

## Methodology

This strategy waits for two separate signals to agree before entering a trade. RSI (Relative Strength Index) measures momentum by comparing the size of recent gains to recent losses over a 14-day window, scaled between 0 and 100. When RSI drops below 30, the asset is considered oversold. MACD (Moving Average Convergence Divergence) is the difference between a 12-day and 26-day exponential moving average, compared against a 9-day EMA of itself (the signal line). When MACD crosses above its signal line, this indicates short-term momentum has turned bullish.

RSI turning oversold on its own isn't treated as an entry, it only opens a watching state, since acting on RSI alone risks entering while a decline is still accelerating. The strategy then waits for MACD to confirm with a bullish crossover before actually entering. This watching state doesn't last indefinitely though, if MACD hasn't confirmed within 7 trading days of RSI first going oversold, the setup is considered stale and the watch expires. This window was chosen as roughly half the length of the RSI lookback itself, long enough to give MACD a fair chance to confirm, short enough that a signal from two weeks ago isn't still being acted on today.

Similar to the Bollinger Bands project, a plain comparison wasn't enough to build this signal, since it depends on remembering that RSI went oversold recently, not just what's true on the current day alone. A raw signal marks only the day RSI first crosses into oversold territory, which is then filled forward (`.ffill(limit=7)`) for up to 7 days, after which it reverts to 0 if MACD hasn't confirmed. Entry itself is only marked on the day MACD's bullish crossover happens while that watch state is still active.

The exit for this strategy is different from the first two projects, which each used a single fixed condition to close a position. Here, there's no fixed take-profit at all. Once a position has moved at least 1x ATR (Average True Range, a 14-day measure of typical daily price movement) in profit, the stop is moved to breakeven. From that point, a trailing stop follows 1.5x ATR behind the highest price reached since entry, and only moves up, never back down. A position is closed once price falls to that trailing level. This lets a strong move run without a predetermined cap, while breakeven protection means a trade that reaches profit can no longer turn into an outright loss.

Because the trailing stop depends on the highest price reached since a specific trade began, each trade needs its own running high rather than one continuous running high across the whole dataset, otherwise a new trade would inherit a stale peak from a previous, unrelated trade. Trades are grouped using a `trade_id`, a number that increases by one every time a new entry fires, and both the running high and breakeven state are calculated within each `trade_id` group separately using `groupby()`, so no trade's calculations are contaminated by another's.

The same look-ahead bias fix used in the first two projects applies here too, on both sides of the trade this time. Both the entry trigger and the exit trigger are shifted forward by one day before being combined into the final position signal, so a position always reflects the previous day's known information rather than the same day's close.

The same metrics are used to measure this strategy as in the first two projects: cumulative return, Sharpe ratio, and max drawdown, split across a 70/30 train and test period the same way. One difference worth noting here: because this strategy trades far less often than the previous two, several tickers have periods with zero trades and therefore zero variance in returns, which makes the Sharpe ratio mathematically undefined (shown as N/A in the results). This isn't a bug, it reflects a genuinely inactive period rather than poor performance. Drawdown also had to be recalculated separately within each train and test slice rather than sliced from a single whole-dataset column, since a whole-dataset running peak could otherwise carry a high point from the training period into a flat test period and understate how calm that period actually was.

## How to run

### Prerequisites

This project requires Python along with the following libraries: `yfinance`, `pandas`, `numpy`, and `matplotlib`.

### Installation

```
pip install yfinance pandas numpy matplotlib
```

### Running the script

```
python "MACD RSI Strategy.py"
```

You'll be prompted to enter a ticker symbol (e.g. `SPY`), then asked whether you'd like to add another. Repeat as many times as needed, then type `no` to finish. The script will then download data, calculate the strategy for each ticker, and print a combined summary table with performance metrics for every ticker in the terminal.

## Results

Each cell shows train / test values. Cumulative return of 1.00 = break-even; values are growth of $1 invested. N/A means the Sharpe ratio was undefined for that period, since the strategy held no position and therefore had zero variance in returns.

| Ticker | Cum. Return (Strategy) | Cum. Return (Buy & Hold) | Sharpe (Strategy) | Sharpe (Buy & Hold) | Max Drawdown (Strategy) |
|--------|------------------------|---------------------------|---------------------|------------------------|----------------------------|
| SPY    | 1.12 / 1.00            | 1.09 / 0.97                | 4.33 / N/A           | 1.96 / -1.28            | -0.02 / 0.00                |
| UPRO   | 1.37 / 1.00            | 1.24 / 0.89                | 4.22 / N/A           | 1.72 / -1.51            | -0.06 / 0.00                |
| AAPL   | 1.22 / 1.01            | 1.20 / 1.09                | 3.76 / 0.62          | 2.28 / 1.92             | -0.03 / -0.04               |
| TSLA   | 1.01 / 1.00            | 0.98 / 0.70                | 0.21 / N/A           | 0.07 / -3.29            | -0.16 / 0.00                |
| ORCL   | 1.00 / 1.00            | 1.40 / 0.51                | N/A / N/A            | 1.92 / -7.73            | 0.00 / 0.00                 |
| UNH    | 1.00 / 1.02            | 1.32 / 1.12                | N/A / 0.77           | 2.72 / 2.84             | 0.00 / -0.04                |

The clearest pattern across this set is how selective this strategy is compared to the first two projects. Four of the six tickers (SPY, UPRO, TSLA, ORCL) never entered a single position during the test period, and ORCL never entered a position across the entire 6 months. Only AAPL and UNH traded during the test window, which is a much lower hit rate than either the moving average or Bollinger Bands strategy produced on the same tickers.

ORCL is the most extreme case, entering zero trades across the whole period rather than just the test period. ORCL was in a strong, sustained downtrend over this window (buy-and-hold lost around 49% in test alone), and while RSI dropping into oversold territory is common in a decline like this, MACD confirming with a bullish crossover within the 7-day watch window apparently never happened, meaning bearish momentum stayed too persistent to produce a genuine confirmation. This shows the strategy's confluence requirement working as intended, refusing to buy into a falling knife just because RSI looked cheap, though it also means the strategy captured none of ORCL's eventual moves either way.

AAPL is the only ticker where the strategy traded in both periods. In test, the strategy returned 1.01 against buy-and-hold's 1.09, and a Sharpe of 0.62 against buy-and-hold's 1.92. Despite genuinely entering and exiting trades here, the strategy still lagged the simple control, most likely because waiting for both RSI to go oversold and MACD to then confirm means missing the earliest, often strongest part of a recovery, the same dynamic that caused the moving average strategy to underperform on trending assets.

UNH shows the clearest single example of this strategy actually working end to end, a lone trade lasting 34 days that covered almost the entire test period. Even so, it returned 1.02 against buy-and-hold's 1.12, and a Sharpe of 0.77 against 2.84, underperforming on both counts. This is the same pattern as AAPL, and the same pattern seen with AAPL in the moving average project: a confirmation-based entry means missing part of a strong move before the strategy is even willing to get in.

Testing the watch window at 5, 7, and 10 days across all six tickers isolated exactly one sensitivity: TSLA had no trade at all with a 5 day window, but the same 47-day trade appeared at both 7 and 10 days. Every other ticker was unaffected regardless of window size. This pins down TSLA's MACD confirmation to somewhere between day 6 and day 7 after RSI first went oversold, close enough to the boundary that a single day's difference in the window setting determined whether the strategy traded TSLA at all.

Taken together, these results raise a genuine question about this strategy's practicality: is only trading with real conviction four times across six tickers over six months disciplined behaviour worth having, or is it too restrictive to be useful on its own? Given how few opportunities it actually acts on, this strategy may be better suited as a confirmation filter layered on top of a more active strategy, rather than a standalone approach.

## Limitations and improvements

- This strategy is highly selective by design. Four of six tickers had zero trades in the test period, and one had zero trades across the full six months. With so few trades per ticker, a single trade can dominate a ticker's entire result (as with UNH), which makes these results fragile and not statistically reliable on their own.
- The watch window sensitivity was tested by comparing 5, 7, and 10 day windows across all six tickers. Every ticker's results stayed identical across all three window sizes, except TSLA, which had no trade at all with a 5 day window but the same 47 day trade with both the 7 and 10 day windows. This shows TSLA's MACD confirmation landed right at the boundary between 5 and 7 days, and that this single parameter choice can be the difference between a ticker trading at all or not, even though most tickers in this set were unaffected by it.
- The trailing stop distance (1.5x ATR) and breakeven trigger (1x ATR) are both fixed and untested against other multiples.
- No fixed take-profit was used, by design, so a small number of trades can carry disproportionate weight in the results, for better or worse, depending on how far that single trade happened to run.
- The model assumes zero transaction costs and slippage when entering or exiting positions, which makes the results more favourable than real-world trading conditions would allow.
- Data is sourced via the yfinance library, which occasionally returns errors or incomplete data for valid tickers due to connectivity or reliability issues, rather than genuine data unavailability.
- Unlike the first two projects, no equity curve visualization was built for this one.
- This project currently evaluates a single strategy (RSI and MACD confluence with an ATR-based trailing stop); findings reflect this specific approach and shouldn't be generalized to trading strategies more broadly.

This is exactly why the next phase of this project moves to a Donchian Channel breakout strategy, testing a structure-based entry on daily data before extending the same approach to intraday data.

## Tech Stack

- **Python** - core language
- **Pandas** - data handling, rolling and exponential averages, groupby-based per-trade calculations
- **Numpy** - vectorized calculations
- **yfinance** - historical price data

## Column glossary

- **daily_return** - this is the percentage change of today's close price to yesterday's close price
- **RSI** - Relative Strength Index, measures momentum on a 0-100 scale by comparing the size of recent gains to recent losses over a 14-day window. Below 30 is treated as oversold
- **MACD** - the difference between a 12-day and 26-day exponential moving average, shows whether short-term momentum is accelerating faster than the longer-term trend
- **EMA_signal** - a 9-day exponential moving average of the MACD line itself, used as the confirmation line MACD is compared against
- **watch_state** - marks whether RSI has recently gone oversold and the strategy is waiting for MACD to confirm, active for up to 7 days after RSI first crosses below 30, then expires if MACD hasn't confirmed by then
- **entry_signal** - marks the single day both conditions are met: RSI has been oversold within the watch window and MACD crosses above its signal line on that day
- **ATR** - Average True Range, a 14-day rolling average of the true range (the largest of today's high minus low, high minus yesterday's close, or low minus yesterday's close), used as a measure of typical daily price movement
- **trade_id** - a number that increases by one every time a new trade is entered, used to group each trade's own calculations (like entry price and running high) separately from every other trade
- **entry_price** - the closing price on the day a specific trade was entered, held constant for every day of that same trade
- **breakeven_active** - marks whether a trade has moved at least 1x ATR in profit since entry. Once true, the stop is treated as moved up to entry price and stays active for the rest of that trade
- **running_high** - the highest closing price reached since a specific trade began, resetting fresh for each new trade
- **trailing_stop** - the running high minus 1.5x ATR, the level price needs to fall to for a position to be closed once breakeven is active
- **signal** - the final, continuous record of whether a position is held on any given day, built by combining the shifted entry and exit triggers and filling forward between them
- **position_signal** - the most recent day's value of `signal` (1 if a position was held, 0 if not), taken from the last row of the dataset
- **strategy_return** - `daily_return` earned only on days `signal` = 1, zero on days out of the market
- **train** - refers to the 70% of the data used to evaluate how the strategy performs before testing it on unseen data
- **test** - refers to the last 30% of the data used to simulate unseen and out-of-sample performance of the algorithm
- **cumulative_strategy** - this is the combined result of the trading strategy. This applies to both train and test
- **cumulative_buy/hold** - this column is to have a standard/control variable to compare our strategy to, and refers to simply buying and holding the asset. This applies to both train and test
- **sharpe_strategy** - this takes the Sharpe value of our RSI and MACD strategy, Sharpe refers to how much we are risking to generate profit
- **sharpe_buy/hold** - this takes the Sharpe value of the buy-hold control, Sharpe refers to how much we are risking to generate profit
- **max_drawdown** - the largest percentage drop from a peak in the strategy's cumulative value to a subsequent low, showing the worst loss an investor following this strategy would have experienced at any point
- **days_in_position / days_out_of_position** - total number of days across the whole period spent in or out of a position
- **avg_days_in_position** - the average length, in days, of a single continuous "in position" streak, i.e. how long a typical trade lasted
