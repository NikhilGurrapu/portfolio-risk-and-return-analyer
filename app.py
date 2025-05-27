import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
from datetime import date
import plotly.graph_objects as go
import statsmodels.api as sm
from scipy.stats import norm
from scipy.optimize import minimize



# Load available tickers from CSV
@st.cache_data
def load_tickers():
    df = pd.read_csv('nse_tickers.csv')
    return df['ticker'].dropna().unique().tolist()

# Fetch historical closing prices
@st.cache_data
def fetch_data(tickers, start, end):
    return yf.download(tickers, start=start, end=end)["Close"]

# Calculate returns, volatility, Sharpe ratio, and cumulative returns
def analyze_portfolio(data, weights):
    daily_returns = data.pct_change().dropna()
    mean_daily_returns = daily_returns.mean()
    cov_matrix = daily_returns.cov()

    annual_return = np.sum(mean_daily_returns * weights) * 252
    annual_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
    sharpe_ratio = annual_return / annual_volatility

    cum_returns = (1 + daily_returns).cumprod()
    portfolio_cum_returns = (cum_returns * weights).sum(axis=1)

    return annual_return, annual_volatility, sharpe_ratio, portfolio_cum_returns, daily_returns

def monte_carlo_simulation(start_value, mean_daily_return, daily_volatility, days=252, num_simulations=10000):
    simulations = np.zeros((days, num_simulations))
    for sim in range(num_simulations):
        prices = [start_value]
        for day in range(1, days):
            shock = np.random.normal(loc=mean_daily_return, scale=daily_volatility)
            price = prices[-1] * (1 + shock)
            prices.append(price)
        simulations[:, sim] = prices
    return simulations

# Calculate drawdown from peak
def calculate_drawdown(cum_returns):
    running_max = cum_returns.cummax()
    drawdown = (cum_returns - running_max) / running_max
    return drawdown

# Calculate rolling Sharpe ratio
def rolling_sharpe(daily_returns, window=63):
    rolling_return = daily_returns.rolling(window).mean() * 252
    rolling_volatility = daily_returns.rolling(window).std() * np.sqrt(252)
    rolling_sharpe_ratio = rolling_return / rolling_volatility
    return rolling_sharpe_ratio

# optimize the weights
def optimize_portfolio(daily_returns):
    mean_returns = daily_returns.mean() * 252
    cov_matrix = daily_returns.cov() * 252
    num_assets = len(mean_returns)

    def portfolio_metrics(weights):
        port_return = np.dot(weights, mean_returns)
        port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        return -port_return / port_vol  # Negative Sharpe (for minimization)

    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(num_assets))
    init_guess = num_assets * [1. / num_assets]

    result = minimize(portfolio_metrics, init_guess, method='SLSQP', bounds=bounds, constraints=constraints)

    return result.x if result.success else init_guess



# ---- Streamlit UI ----
st.set_page_config(page_title="Portfolio Risk & Return Analyzer", layout="wide")
st.title("📈 Portfolio Risk & Return Analyzer")

# Load tickers
all_tickers = load_tickers()

# Ticker and Benchmark Selection
selected_tickers = st.multiselect("Select your stock tickers", all_tickers, default=all_tickers[:3])

benchmark_map = {
    "NIFTY 50": "^NSEI",
    "S&P 500": "^GSPC"
}
benchmark_name = st.selectbox("Select a benchmark index", list(benchmark_map.keys()))

# Investment Inputs
if selected_tickers:
    st.subheader("💰 Investment Allocation (₹ per stock)")
    investment_amounts = {}
    total_investment = 0

    for ticker in selected_tickers:
        amount = st.number_input(f"{ticker}:", min_value=0.0, step=100.0, key=ticker)
        investment_amounts[ticker] = amount
        total_investment += amount

    if total_investment <= 0:
        st.error("⚠️ Please enter valid investment amounts to continue.")
    else:
        weights = np.array([investment_amounts[t] / total_investment for t in selected_tickers])

        # Date range input
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", value=date(2020, 1, 1))
        with col2:
            end_date = st.date_input("End Date", value=date.today())

        # Analysis Trigger
        if st.button("🔍 Analyze Portfolio"):
            try:
                data = fetch_data(selected_tickers, start_date, end_date)
                benchmark_data = fetch_data(benchmark_map[benchmark_name], start_date, end_date)

                port_return, port_vol, port_sharpe, port_cum_returns, daily_returns = analyze_portfolio(data, weights)

                benchmark_returns = benchmark_data.pct_change().dropna()
                benchmark_cum_returns = (1 + benchmark_returns).cumprod()

                # Handle benchmark formats
                if isinstance(benchmark_returns, pd.DataFrame):
                    benchmark_returns = benchmark_returns.iloc[:, 0]

                bench_return = benchmark_returns.mean() * 252
                bench_vol = benchmark_returns.std() * np.sqrt(252)
                bench_sharpe = bench_return / bench_vol

                # Drawdowns
                port_drawdown = calculate_drawdown(port_cum_returns)
                bench_drawdown = calculate_drawdown(benchmark_cum_returns.squeeze())

                # Rolling Sharpe
                port_rolling_sharpe = rolling_sharpe(daily_returns.dot(weights))
                bench_rolling_sharpe = rolling_sharpe(benchmark_returns)

                # --- Display Metrics ---
                st.success("📊 Portfolio Summary")
                st.metric("Annual Return", f"{port_return:.2%}")
                st.metric("Annual Volatility", f"{port_vol:.2%}")
                st.metric("Sharpe Ratio", f"{port_sharpe:.2f}")

                st.info(f"📌 {benchmark_name} Benchmark")
                st.write(f"**Annual Return:** {bench_return:.2%}")
                st.write(f"**Volatility:** {bench_vol:.2%}")
                st.write(f"**Sharpe Ratio:** {bench_sharpe:.2f}")

                # --- Cumulative Returns Plot ---
                fig_cum = go.Figure()
                fig_cum.add_trace(go.Scatter(x=port_cum_returns.index, y=port_cum_returns, name="Portfolio", line=dict(color='deepskyblue')))
                fig_cum.add_trace(go.Scatter(x=benchmark_cum_returns.index, y=benchmark_cum_returns.squeeze(), name=benchmark_name, line=dict(color='orange')))
                fig_cum.update_layout(title="📈 Cumulative Returns", xaxis_title="Date", yaxis_title="Cumulative Return", template="plotly_dark")
                st.plotly_chart(fig_cum, use_container_width=True)

                # --- Drawdown Plot ---
                fig_dd = go.Figure()
                fig_dd.add_trace(go.Scatter(x=port_drawdown.index, y=port_drawdown, name="Portfolio"))
                fig_dd.add_trace(go.Scatter(x=bench_drawdown.index, y=bench_drawdown, name=f"{benchmark_name}"))
                fig_dd.update_layout(title="📉 Drawdown", yaxis_title="Drawdown", template="plotly_dark")
                st.plotly_chart(fig_dd, use_container_width=True)

                # --- Rolling Sharpe Plot ---
                fig_rs = go.Figure()
                fig_rs.add_trace(go.Scatter(x=port_rolling_sharpe.index, y=port_rolling_sharpe, name="Portfolio"))
                fig_rs.add_trace(go.Scatter(x=bench_rolling_sharpe.index, y=bench_rolling_sharpe, name=f"{benchmark_name}"))
                fig_rs.update_layout(title="📊 Rolling Sharpe Ratio (3 months)", yaxis_title="Sharpe Ratio", template="plotly_dark")
                st.plotly_chart(fig_rs, use_container_width=True)

            except Exception as e:
                st.error(f"🚨 An error occurred: {e}")

            st.subheader("🧠 Suggested Optimal Investments (%) (MVO)")

            opt_weights = optimize_portfolio(daily_returns[selected_tickers])

            opt_weight_df = pd.DataFrame({
                "Ticker": selected_tickers,
                "Original Weight": weights*total_investment,
                "Optimal Weight": np.round(opt_weights, 4)*total_investment
            })

            st.dataframe(opt_weight_df.set_index("Ticker"))

            # Plot optional bar chart
            fig_wt = go.Figure()
            fig_wt.add_trace(go.Bar(name="Original", x=selected_tickers, y=weights))
            fig_wt.add_trace(go.Bar(name="Optimized", x=selected_tickers, y=opt_weights))
            fig_wt.update_layout(barmode='group', title="Portfolio Weights Comparison", template="plotly_dark")
            st.plotly_chart(fig_wt, use_container_width=True)

            st.subheader("📊 Optimized Portfolio Performance (MVO)")

            # Re-calculate performance for optimized weights
            opt_port_return, opt_port_vol, opt_sharpe, _, _ = analyze_portfolio(data, np.array(opt_weights))

            # Show comparison table
            comparison_df = pd.DataFrame({
                "Metric": ["Annual Return", "Volatility", "Sharpe Ratio"],
                "Original Portfolio": [f"{port_return:.2%}", f"{port_vol:.2%}", f"{port_sharpe:.2f}"],
                "Optimized Portfolio": [f"{opt_port_return:.2%}", f"{opt_port_vol:.2%}", f"{opt_sharpe:.2f}"]
            })

            st.table(comparison_df.set_index("Metric"))


            # Monte Carlo Simulation Section
            st.subheader("🔮 Monte Carlo Simulation (1-Year Forecast)")

            initial_value = total_investment
            mean_daily_ret = daily_returns.dot(np.array(weights)).mean()
            daily_volatility = daily_returns.dot(np.array(weights)).std()

            simulations = monte_carlo_simulation(
                start_value=initial_value,
                mean_daily_return=mean_daily_ret,
                daily_volatility=daily_volatility,
                days=252,
                num_simulations=500  # adjust if needed
            )

            # Plot Monte Carlo simulation paths
            fig_mc = go.Figure()
            for i in range(10):  # Show only 25 paths for performance
                fig_mc.add_trace(go.Scatter(
                    x=list(range(252)),
                    y=simulations[:, i],
                    mode='lines',
                    line=dict(width=1),
                    showlegend=False
                ))

            fig_mc.update_layout(
                title="Monte Carlo Simulation of Portfolio Value (Next 1 Year)",
                xaxis_title="Days",
                yaxis_title="Portfolio Value",
                template="plotly_dark",
                height=500
            )
            st.plotly_chart(fig_mc, use_container_width=True)

            # Plot histogram of final values
            final_values = simulations[-1, :]
            fig_hist = go.Figure(data=[go.Histogram(x=final_values, nbinsx=50)])
            fig_hist.update_layout(
                title="Histogram of Final Portfolio Values (1 Year)",
                xaxis_title="Final Value",
                yaxis_title="Frequency",
                template="plotly_dark",
                height=400
            )
            st.plotly_chart(fig_hist, use_container_width=True)

            # Show probability of profit
            prob_profit = np.mean(final_values > initial_value)
            st.metric("📈 Probability Portfolio Increases in Value", f"{prob_profit:.2%}")


            # CAPM Analysis
            st.subheader("📘 CAPM Analysis (Capital Asset Pricing Model)")


            
            try:
                # Align portfolio and benchmark returns safely
                port_ret = daily_returns.dot(weights)
                port_ret, bench_ret = port_ret.align(benchmark_returns, join='inner')


                # Add intercept to benchmark returns
                X = sm.add_constant(bench_ret)
                model = sm.OLS(port_ret, X).fit()

                # Extract CAPM metrics
                alpha = model.params['const']
                benchmark_ticker = benchmark_map[benchmark_name]
                beta = model.params[benchmark_ticker]


                st.markdown(f"**Alpha (α):** `{alpha:.6f}`")
                st.markdown(f"**Beta (β):** `{beta:.3f}`")

                with st.expander("📄 CAPM Regression Summary"):
                    st.text(model.summary())

                # Create CAPM Expected Return Line
                expected_returns = alpha + beta * benchmark_returns
                capm_cum_returns = (1 + expected_returns).cumprod() * total_investment  # Simulated portfolio growth

                # Your actual portfolio cumulative returns (already calculated earlier)
                actual_cum_returns = (1 + port_ret).cumprod() * total_investment

                # Plot side-by-side
                fig_capm_line = go.Figure()

                fig_capm_line.add_trace(go.Scatter(
                    x=actual_cum_returns.index,
                    y=actual_cum_returns,
                    name="Actual Portfolio Value",
                    line=dict(color='dodgerblue')
                ))

                fig_capm_line.add_trace(go.Scatter(
                    x=capm_cum_returns.index,
                    y=capm_cum_returns,
                    name="CAPM Expected Value",
                    line=dict(color='orange', dash='dash')
                ))

                fig_capm_line.update_layout(
                    title="📈 Actual Portfolio vs CAPM Expected Return",
                    xaxis_title="Date",
                    yaxis_title="Portfolio Value",
                    template="plotly_dark"
                )

                st.plotly_chart(fig_capm_line, use_container_width=True)

                st.markdown("### 📘 What does this mean?")
                st.info(f'''
                - **Beta (`β`) = {beta:.2f}**:
                  This means your portfolio is {"more" if beta > 1 else "less"} volatile than the market.
                  - β > 1 → More sensitive to market movements (higher risk, higher reward)
                  - β < 1 → Less sensitive (more stable, lower swings)
                
                - **Alpha (`α`) = {alpha:.4f}**:
                  This is the extra return your portfolio generated compared to the market **after accounting for risk**.
                  - α > 0 → Outperformed
                  - α < 0 → Underperformed
                ''')

            except Exception as e:
                st.error(f"⚠️ CAPM analysis failed: {e}")

            st.subheader("⚠️ Value at Risk (VaR)")

            try:
                # Portfolio returns
                port_daily_ret = daily_returns.dot(weights)

                # Investment value
                current_value = total_investment

                # Confidence level
                confidence_level = 0.95
                alpha = 1 - confidence_level

                # Historical VaR
                hist_var = np.percentile(port_daily_ret, 100 * alpha)
                var_hist = -hist_var * current_value

                # Parametric VaR
                mean_ret = port_daily_ret.mean()
                std_ret = port_daily_ret.std()
                z_score = norm.ppf(alpha)
                var_parametric = -(mean_ret + z_score * std_ret) * current_value

                # Show results
                st.metric(label="Historical VaR (1-day, 95%)", value=f"₹{var_hist:,.0f}")
                st.metric(label="Parametric VaR (1-day, 95%)", value=f"₹{var_parametric:,.0f}")

                st.caption("This means you are 95% confident your portfolio will not lose more than this in a single day.")

            except Exception as e:
                st.error(f"VaR calculation failed: {e}")

