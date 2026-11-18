import warnings
warnings.filterwarnings("ignore")

from datetime import date, timedelta
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Investment Robo Advisor",
    page_icon="📈",
    layout="wide"
)

# ----------------------------- Asset universe -----------------------------
ASSET_UNIVERSE = {
    "Pakistan Stock Exchange (PSX)": {
        "stocks": {
            "FFC": "FFC.KA",
            "Meezan Bank": "MEBL.KA",
            "Lucky Cement": "LUCK.KA",
            "Hub Power": "HUBC.KA",
            "Pakistan Petroleum": "PPL.KA",
            "Engro Holdings": "ENGROH.KA",
            "Systems Limited": "SYS.KA",
            "Pakistan Oilfields": "POL.KA",
            "Bank of Punjab": "BOP.KA",
            "Fauji Foods": "FFL.KA",
        },
        "benchmark": "^KSE"
    },
    "United States": {
        "stocks": {
            "Apple": "AAPL",
            "Microsoft": "MSFT",
            "NVIDIA": "NVDA",
            "Amazon": "AMZN",
            "Alphabet": "GOOGL",
            "Berkshire Hathaway": "BRK-B",
            "Coca-Cola": "KO",
            "JPMorgan Chase": "JPM",
            "Johnson & Johnson": "JNJ",
            "SPDR S&P 500 ETF": "SPY",
        },
        "benchmark": "SPY"
    }
}

RISK_PROFILES = {
    "Conservative": {
        "stocks": 20, "bonds": 55, "cash": 25,
        "description": "Focuses on capital preservation and relatively lower volatility."
    },
    "Moderate": {
        "stocks": 50, "bonds": 35, "cash": 15,
        "description": "Balances long-term growth with a stabilizing allocation."
    },
    "Aggressive": {
        "stocks": 75, "bonds": 15, "cash": 10,
        "description": "Targets higher long-term growth and accepts larger fluctuations."
    }
}

def money(value, currency):
    return f"{currency} {value:,.2f}"

@st.cache_data(ttl=3600, show_spinner=False)
def download_prices(tickers, period):
    """Download adjusted historical prices. Returns an empty frame on data errors."""
    data = yf.download(
        tickers=list(tickers),
        period=period,
        auto_adjust=True,
        progress=False,
        group_by="column",
        threads=True
    )
    if data is None or data.empty:
        return pd.DataFrame()

    if isinstance(data.columns, pd.MultiIndex):
        if "Close" in data.columns.get_level_values(0):
            prices = data["Close"].copy()
        else:
            prices = data.xs(data.columns.levels[0][0], axis=1, level=0)
    else:
        # yfinance may return a single ticker with ordinary columns.
        if "Close" not in data.columns:
            return pd.DataFrame()
        prices = data[["Close"]].rename(columns={"Close": tickers[0]})

    if isinstance(prices, pd.Series):
        prices = prices.to_frame(name=tickers[0])
    prices = prices.dropna(how="all")
    return prices

def calculate_metrics(prices):
    """Calculate annualized return, volatility, Sharpe ratio and max drawdown."""
    results = []
    for ticker in prices.columns:
        series = prices[ticker].dropna()
        if len(series) < 30:
            continue
        daily_returns = series.pct_change().dropna()
        total_return = (series.iloc[-1] / series.iloc[0]) - 1
        years = max((len(daily_returns) / 252), 1 / 252)
        cagr = (1 + total_return) ** (1 / years) - 1 if total_return > -1 else -1
        volatility = daily_returns.std() * np.sqrt(252)
        risk_free_rate = 0.05
        sharpe = ((daily_returns.mean() * 252) - risk_free_rate) / volatility if volatility > 0 else np.nan
        running_max = series.cummax()
        drawdown = (series / running_max) - 1
        results.append({
            "Ticker": ticker,
            "Total Return": total_return,
            "Annualized Return": cagr,
            "Annualized Volatility": volatility,
            "Sharpe Ratio*": sharpe,
            "Maximum Drawdown": drawdown.min()
        })
    return pd.DataFrame(results)

def build_allocation(profile, monthly_investment, emergency_months, monthly_expenses):
    """Suggest a high-level allocation; this is an educational model, not personal advice."""
    weights = RISK_PROFILES[profile].copy()
    emergency_target = monthly_expenses * emergency_months
    return weights, emergency_target, monthly_investment

# ----------------------------- Sidebar inputs -----------------------------
st.title("📈 Investment Robo Advisor")
st.caption("An educational portfolio-planning and historical-return analysis app.")

with st.sidebar:
    st.header("Your financial profile")
    currency = st.selectbox("Currency", ["PKR", "USD", "GBP", "EUR"])
    monthly_income = st.number_input("Monthly income", min_value=0.0, value=150000.0, step=5000.0)
    monthly_expenses = st.number_input("Monthly expenses", min_value=0.0, value=90000.0, step=5000.0)
    existing_savings = st.number_input("Current savings / emergency fund", min_value=0.0, value=100000.0, step=10000.0)
    risk_profile = st.selectbox("Risk appetite", list(RISK_PROFILES.keys()), index=1)
    horizon = st.selectbox("Investment time horizon", ["Less than 1 year", "1–3 years", "3–5 years", "5–10 years", "More than 10 years"], index=3)
    emergency_months = st.slider("Emergency fund target (months of expenses)", 3, 12, 6)
    invest_pct = st.slider("Share of monthly surplus to invest (%)", 0, 100, 70)
    market = st.selectbox("Market for stock research", list(ASSET_UNIVERSE.keys()))
    period_label = st.selectbox("Historical analysis period", ["1 year", "2 years", "5 years", "10 years"], index=2)

monthly_surplus = max(monthly_income - monthly_expenses, 0)
monthly_investment = monthly_surplus * invest_pct / 100
profile = RISK_PROFILES[risk_profile]
allocation, emergency_target, monthly_investment = build_allocation(
    risk_profile, monthly_investment, emergency_months, monthly_expenses
)
period_map = {"1 year": "1y", "2 years": "2y", "5 years": "5y", "10 years": "10y"}
period = period_map[period_label]

# ----------------------------- Overview -----------------------------
st.subheader("1. Your financial snapshot")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Monthly surplus", money(monthly_surplus, currency))
c2.metric("Suggested monthly investment", money(monthly_investment, currency))
c3.metric("Savings rate", f"{(monthly_surplus / monthly_income * 100) if monthly_income else 0:.1f}%")
c4.metric("Emergency fund gap", money(max(emergency_target - existing_savings, 0), currency))

if monthly_income <= monthly_expenses:
    st.warning("Your expenses are equal to or higher than your income. Consider building a positive monthly surplus before investing.")
if existing_savings < emergency_target:
    st.info(
        f"Your selected emergency-fund target is {money(emergency_target, currency)}. "
        "Consider prioritizing this reserve before investing money you may need soon."
    )

# ----------------------------- Allocation -----------------------------
st.subheader("2. Suggested asset allocation")
st.write(profile["description"])
st.caption(
    "The percentages below are illustrative model allocations based on the risk profile selected. "
    "The app does not assess your complete financial circumstances."
)

allocation_df = pd.DataFrame({
    "Asset class": ["Stocks / equity", "Bonds / fixed income", "Cash / liquid reserves"],
    "Allocation (%)": [allocation["stocks"], allocation["bonds"], allocation["cash"]]
})
allocation_df["Monthly amount"] = allocation_df["Allocation (%)"] / 100 * monthly_investment

left, right = st.columns([1, 1])
with left:
    fig = px.pie(
        allocation_df, names="Asset class", values="Allocation (%)",
        hole=0.42, title="Target portfolio mix"
    )
    st.plotly_chart(fig, use_container_width=True)
with right:
    display_allocation = allocation_df.copy()
    display_allocation["Allocation (%)"] = display_allocation["Allocation (%)"].map(lambda x: f"{x:.0f}%")
    display_allocation["Monthly amount"] = display_allocation["Monthly amount"].map(lambda x: money(x, currency))
    st.dataframe(display_allocation, hide_index=True, use_container_width=True)
    st.metric("Illustrative monthly investment", money(monthly_investment, currency))

# ----------------------------- Stock analysis -----------------------------
st.subheader("3. Stock recommendations and historical return analysis")
universe = ASSET_UNIVERSE[market]["stocks"]
selected_names = st.multiselect(
    "Choose companies / ETFs to analyze",
    options=list(universe.keys()),
    default=list(universe.keys())[:5]
)

if selected_names:
    ticker_map = {name: universe[name] for name in selected_names}
    tickers = list(ticker_map.values())
    with st.spinner("Downloading historical market data..."):
        prices = download_prices(tuple(tickers), period)

    if prices.empty:
        st.error(
            "Market data could not be retrieved. Check your internet connection, "
            "ticker availability, or try another market / period."
        )
    else:
        # Rename tickers to company names for display.
        reverse_map = {ticker: name for name, ticker in ticker_map.items()}
        prices = prices.rename(columns=reverse_map)
        prices = prices.dropna(axis=1, how="all")
        metrics = calculate_metrics(prices.rename(columns={name: ticker_map[name] for name in prices.columns}))
        if not metrics.empty:
            metrics["Company"] = metrics["Ticker"].map(reverse_map)
            metrics = metrics[["Company", "Ticker", "Total Return", "Annualized Return",
                               "Annualized Volatility", "Sharpe Ratio*", "Maximum Drawdown"]]
            for col in ["Total Return", "Annualized Return", "Annualized Volatility", "Maximum Drawdown"]:
                metrics[col] = metrics[col].map(lambda x: f"{x:.2%}" if pd.notna(x) else "N/A")
            metrics["Sharpe Ratio*"] = metrics["Sharpe Ratio*"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
            st.dataframe(metrics, hide_index=True, use_container_width=True)

        normalized = prices.apply(lambda col: col / col.dropna().iloc[0] * 100 if col.notna().any() else col)
        chart_data = normalized.reset_index().melt(
            id_vars=normalized.index.name or "Date",
            var_name="Company", value_name="Growth of 100"
        )
        date_col = chart_data.columns[0]
        fig_prices = px.line(
            chart_data, x=date_col, y="Growth of 100", color="Company",
            title=f"Price performance (starting value = 100) — {period_label}"
        )
        st.plotly_chart(fig_prices, use_container_width=True)

        st.markdown("**Simple model-based watchlist**")
        st.write(
            "The table ranks the selected securities by historical risk-adjusted performance where data is available. "
            "A higher Sharpe ratio is not a guarantee of future returns and should not be used alone to select a stock."
        )
        raw_metrics = calculate_metrics(prices.rename(columns={name: ticker_map[name] for name in prices.columns}))
        if not raw_metrics.empty:
            raw_metrics["Company"] = raw_metrics["Ticker"].map(reverse_map)
            ranked = raw_metrics.sort_values("Sharpe Ratio*", ascending=False, na_position="last")
            ranked = ranked[["Company", "Ticker", "Annualized Return", "Annualized Volatility", "Sharpe Ratio*", "Maximum Drawdown"]]
            st.dataframe(
                ranked.style.format({
                    "Annualized Return": "{:.2%}",
                    "Annualized Volatility": "{:.2%}",
                    "Sharpe Ratio*": "{:.2f}",
                    "Maximum Drawdown": "{:.2%}"
                }),
                use_container_width=True
            )
else:
    st.info("Select at least one company to view historical return analysis.")

# ----------------------------- Guidance -----------------------------
st.subheader("4. Planning notes")
notes = []
if horizon in ["Less than 1 year", "1–3 years"]:
    notes.append("Your time horizon is relatively short. Avoid relying heavily on volatile shares for money needed soon.")
elif horizon == "3–5 years":
    notes.append("For a medium-term goal, balance growth assets with lower-volatility holdings.")
else:
    notes.append("A longer horizon may allow more time to ride out market fluctuations, but losses are still possible.")

if monthly_surplus == 0:
    notes.append("Review spending and debt obligations before committing to a monthly investment plan.")
if existing_savings < emergency_target:
    notes.append("Build the emergency reserve in an accessible, lower-risk account before increasing market exposure.")
notes.append("Diversify across companies and sectors; review allocations periodically rather than reacting to daily price movements.")
for note in notes:
    st.write(f"• {note}")

with st.expander("Methodology, assumptions and important disclaimer"):
    st.markdown(
        """
        - Asset allocation uses fixed illustrative stock / bond / cash mixes for Conservative, Moderate and Aggressive profiles.
        - Suggested monthly investment is calculated from monthly income less monthly expenses, multiplied by the investment percentage you select.
        - Historical returns use adjusted prices downloaded through Yahoo Finance (`yfinance`). Data availability and ticker conventions can vary, especially for PSX securities.
        - Annualized return is calculated from the selected historical price window; volatility uses daily returns annualized by √252.
        - The Sharpe ratio uses an assumed 5% annual risk-free rate. Maximum drawdown is the largest peak-to-trough decline in the observed price series.
        - Historical performance is not a prediction. Prices may be delayed, incomplete, adjusted, or unavailable. Verify figures with official exchange and company disclosures.

        **Important:** This application is an educational demonstration, not regulated investment, tax, or legal advice.
        It does not know your liabilities, dependants, tax position, investment experience, liquidity needs, or full financial goals.
        Stocks and ETFs can lose value, including your original capital. Consult a licensed financial adviser before making investment decisions.
        """
    )

st.caption("Built with Streamlit, yfinance, pandas, NumPy and Plotly.")


