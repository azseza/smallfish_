# Trading Concepts and Indicators

This document provides definitions and explanations for the key technical indicators and trading concepts used in the `bybit_scalper` project.

---

## Core Trading Signals

### 1. Orderbook Imbalance (OBI)

**Definition:** Orderbook Imbalance is a measure of the difference between the volume of buy orders (bids) and sell orders (asks) at various price levels in the order book. It provides a real-time snapshot of supply and demand pressure.

**Formula:** A common way to calculate the imbalance ratio is:
`Imbalance = (Bid Volume - Ask Volume) / (Bid Volume + Ask Volume)`

- A positive value suggests higher buying pressure.
- A negative value suggests higher selling pressure.
- A value near zero suggests a balanced market.

An **OBI Burst** refers to a rapid *change* in this imbalance, indicating aggressive new orders entering the market.

### 2. Pull-&-Replace Trap (PRT)

**Definition:** A trading pattern that identifies potentially deceptive behavior (spoofing). It occurs when a market participant quickly cancels a large number of orders on one side and immediately "replaces" them with new orders further away from the current price, all without the market price moving as expected.

**Signal Logic:** The strategy is to "fade" (trade against) the fake pressure.
- **Long Signal:** A large number of **sell orders** are pulled and reposted at a *higher* price, but the market price remains stable. This suggests the selling pressure was artificial, creating a long entry opportunity.
- **Short Signal:** A large number of **buy orders** are pulled and reposted at a *lower* price, but the market price remains stable. This suggests the buying pressure was artificial, creating a short entry opportunity.

### 3. Micro-Momentum (UMOM)

**Definition:** A high-frequency strategy focused on profiting from very short-term price trends or "micro-trends."

**Implementation:** In this project, it's calculated using two Exponential Moving Averages (EMAs) with very short periods (e.g., 1-second and 5-second).
- A **long signal** is generated when the faster EMA crosses above the slower EMA.
- A **short signal** is generated when the faster EMA crosses below the slower EMA.
This signal is often "gated" by other conditions like volatility and spread to ensure market conditions are suitable for a momentum trade.

---

## "Whale Lens" Features (Large Player Activity)

### 4. Large Trade Burst (LTB)

**Definition:** A signal that detects a "burst" of aggressive, large trades executing in a very short time frame. It's a method of quantifying the activity of "whales" or large market participants.

**Implementation:**
1.  **Define a Large Trade:** A trade is considered "large" if its size exceeds a dynamic threshold (e.g., `max(absolute_size, k * median_trade_size_60s)`).
2.  **Measure Bursts:** The system sums the volume of large *aggressive* buy trades and sell trades over a short window (e.g., 300ms).
3.  **Score:** The scores (`s_ltb_long`, `s_ltb_short`) represent the dominance of either the buy-side or sell-side large trades in that brief period.

### 5. Order Book Sweep

**Definition:** A sweep is an aggressive market event where a single large market order clears out multiple levels of resting orders (liquidity) from the order book almost instantaneously. This indicates a strong urgency to buy or sell.

**Characteristics:**
- A sudden, massive spike in volume.
- Rapid price movement in one direction.
- Multiple price levels on the order book are depleted.

### 6. Iceberg Order

**Definition:** An iceberg order is a large order that has been algorithmically broken down into smaller limit orders. Only a small, visible portion (the "tip of the iceberg") is shown on the order book at any time. As the visible part is filled, a new small order is automatically placed, until the total hidden volume is executed.

**Detection:** Traders identify icebergs by noticing a price level that constantly replenishes itself despite being traded against. This indicates a large hidden order is absorbing pressure at that price.

---

## General Technical Concepts

### 7. Exponentially Weighted Moving Average (EWMA)

**Definition:** A type of moving average that places greater weight and significance on the most recent data points. It reacts more quickly to recent price changes than a Simple Moving Average (SMA).

**Formula:** `EWMA_t = α * price_t + (1 - α) * EWMA_{t-1}`
- `α` is the smoothing factor, often calculated as `2 / (N + 1)`, where `N` is the number of periods.

### 8. Realized Volatility (RV)

**Definition:** A measure of price variation calculated directly from historical price data over a specific period.

**Calculation:**
1.  Collect a series of prices (e.g., from trades in the last 10 seconds).
2.  Calculate the logarithmic returns between each consecutive price: `r_i = ln(P_i / P_{i-1})`.
3.  The **Realized Variance** is the sum of the squares of these log returns: `RV = Σ(r_i^2)`.
4.  The **Realized Volatility** is the square root of the Realized Variance: `rv = sqrt(RV)`.

### 9. Sigmoid Function

**Definition:** A mathematical function that maps any real number to a value between 0 and 1. It has a characteristic "S"-shaped curve.

**Formula:** `σ(x) = 1 / (1 + e⁻ˣ)`

**Use in this Project:** It's used to convert a raw signal score (which can be any number) into a "confidence" score between 0 and 1, which is easier to interpret and use for decision-making.

---

## Order Types & Execution Concepts

### 10. One-Cancels-the-Other (OCO) Order

**Definition:** An OCO order is a pair of orders (typically a stop-loss and a take-profit) linked together. When one of the orders is executed, the other is automatically canceled. This is a key tool for managing risk and securing profits on an open position.

### 11. Immediate-Or-Cancel (IOC) Order

**Definition:** An IOC order is a limit order that must be filled immediately. Any portion of the order that cannot be filled instantly is canceled. Partial fills are allowed.

**Use in this Project:** It serves as a fallback entry tactic. If a patient `Post-Only` order fails, the bot can use an IOC order to enter the trade immediately if conditions are still favorable, without the risk of the order lingering on the book.

### 12. Post-Only Order

**Definition:** A Post-Only order is a limit order that is only accepted if it does *not* immediately match with an existing order. This ensures the order is placed on the order book, making you a "maker" and typically resulting in lower trading fees or even a rebate. If the order would cause an immediate trade (i.e., be a "taker"), it is automatically canceled.

### 13. Slippage

**Definition:** Slippage is the difference between the expected price of a trade and the price at which the trade is actually executed. It often occurs in volatile markets or when executing large orders in a market with low liquidity.

### 14. MAE / MFE

**Definition:**
- **Maximum Adverse Excursion (MAE):** Measures the maximum *unrealized loss* a trade experienced during its lifetime. It shows how far the price moved *against* the position. It is used to analyze and optimize stop-loss placement.
- **Maximum Favorable Excursion (MFE):** Measures the maximum *unrealized profit* a trade reached during its lifetime. It shows the peak profit potential of a trade and is used to optimize take-profit strategies.
