# ============================================================
#  UNEMPLOYMENT ANALYSIS WITH PYTHON
#  Task 2 — OASIS INFOBYTE Data Science Internship
#  Author  : Umer
#  Datasets: Unemployment_in_India.csv
#            Unemployment_Rate_upto_11_2020.csv
# ============================================================

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

warnings.filterwarnings("ignore")

# ── config ───────────────────────────────────────────────────
PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

DATA1 = os.path.join("data", "Unemployment_in_India.csv")
DATA2 = os.path.join("data", "Unemployment_Rate_upto_11_2020.csv")

PALETTE  = "Set2"
COVID_COLOR  = "#e74c3c"
PRE_COLOR    = "#3498db"
ACCENT_COLOR = "#f39c12"

# ─────────────────────────────────────────────────────────────
# 1.  LOAD & CLEAN
# ─────────────────────────────────────────────────────────────
def load_and_clean():
    df1 = pd.read_csv(DATA1)
    df2 = pd.read_csv(DATA2)

    # Strip whitespace from column names
    df1.columns = df1.columns.str.strip()
    df2.columns = df2.columns.str.strip()

    # Strip whitespace from string values
    for df in [df1, df2]:
        for col in df.select_dtypes("object").columns:
            df[col] = df[col].str.strip()

    # Parse dates
    df1["Date"] = pd.to_datetime(df1["Date"], dayfirst=True)
    df2["Date"] = pd.to_datetime(df2["Date"], dayfirst=True)

    # Rename long columns for convenience
    rename = {
        "Estimated Unemployment Rate (%)": "Unemployment_Rate",
        "Estimated Employed":              "Employed",
        "Estimated Labour Participation Rate (%)": "Labour_Participation",
    }
    df1.rename(columns=rename, inplace=True)
    df2.rename(columns={**rename, "Region.1": "Geo_Region"}, inplace=True)

    # Drop nulls
    df1.dropna(inplace=True)
    df2.dropna(inplace=True)

    print("=" * 58)
    print("  UNEMPLOYMENT ANALYSIS — Dataset Summary")
    print("=" * 58)
    print(f"\n  Dataset 1  :  {df1.shape[0]} rows × {df1.shape[1]} cols")
    print(f"  Date range :  {df1['Date'].min().date()}  →  {df1['Date'].max().date()}")
    print(f"  States     :  {df1['Region'].nunique()}")
    print(f"\n  Dataset 2  :  {df2.shape[0]} rows × {df2.shape[1]} cols")
    print(f"  Date range :  {df2['Date'].min().date()}  →  {df2['Date'].max().date()}")
    print(f"  States     :  {df2['Region'].nunique()}")
    print(f"\n  Dataset 1 — Nulls:\n{df1.isnull().sum().to_string()}")
    print(f"\n  Dataset 2 — Nulls:\n{df2.isnull().sum().to_string()}")

    print(f"\n  Dataset 1 — Descriptive Stats:")
    print(df1[["Unemployment_Rate","Employed","Labour_Participation"]].describe().round(2).to_string())

    return df1, df2


# ─────────────────────────────────────────────────────────────
# 2.  HELPERS
# ─────────────────────────────────────────────────────────────
def save(fig, name):
    path = os.path.join(PLOTS_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✔  Saved → {path}")


# ─────────────────────────────────────────────────────────────
# 3.  PLOT 1 — Overall Unemployment Rate Over Time
# ─────────────────────────────────────────────────────────────
def plot_overall_trend(df1):
    monthly = (df1.groupby("Date")["Unemployment_Rate"]
                  .mean()
                  .reset_index())

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(monthly["Date"], monthly["Unemployment_Rate"],
            color=PRE_COLOR, linewidth=2.2, marker="o", markersize=4)
    ax.fill_between(monthly["Date"], monthly["Unemployment_Rate"],
                    alpha=0.15, color=PRE_COLOR)

    # Shade COVID period
    covid_start = pd.Timestamp("2020-03-01")
    ax.axvline(covid_start, color=COVID_COLOR, linestyle="--", linewidth=1.6)
    ax.axvspan(covid_start, monthly["Date"].max(),
               alpha=0.07, color=COVID_COLOR)
    ax.text(covid_start + pd.Timedelta(days=5), ax.get_ylim()[1]*0.92,
            "COVID-19\nLockdown", color=COVID_COLOR, fontsize=9, fontweight="bold")

    ax.set(title="Average Unemployment Rate in India Over Time",
           xlabel="Date", ylabel="Unemployment Rate (%)")
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%b %Y"))
    plt.xticks(rotation=45)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    save(fig, "01_overall_trend.png")


# ─────────────────────────────────────────────────────────────
# 4.  PLOT 2 — State-wise Average Unemployment Rate
# ─────────────────────────────────────────────────────────────
def plot_statewise_avg(df1):
    state_avg = (df1.groupby("Region")["Unemployment_Rate"]
                    .mean()
                    .sort_values(ascending=True))

    colors = [COVID_COLOR if v > state_avg.mean() + state_avg.std()
              else PRE_COLOR for v in state_avg.values]

    fig, ax = plt.subplots(figsize=(11, 9))
    bars = ax.barh(state_avg.index, state_avg.values,
                   color=colors, edgecolor="white", height=0.7)
    for bar in bars:
        ax.text(bar.get_width() + 0.15, bar.get_y() + bar.get_height()/2,
                f"{bar.get_width():.1f}%",
                va="center", fontsize=8)
    ax.axvline(state_avg.mean(), color="black", linestyle="--",
               linewidth=1.3, label=f"National avg: {state_avg.mean():.1f}%")
    ax.set(title="State-wise Average Unemployment Rate",
           xlabel="Unemployment Rate (%)", xlim=(0, state_avg.max()+3))
    ax.legend(fontsize=10)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    save(fig, "02_statewise_avg.png")


# ─────────────────────────────────────────────────────────────
# 5.  PLOT 3 — Rural vs Urban Unemployment
# ─────────────────────────────────────────────────────────────
def plot_rural_urban(df1):
    area_monthly = (df1.groupby(["Date","Area"])["Unemployment_Rate"]
                       .mean()
                       .reset_index())

    fig, ax = plt.subplots(figsize=(13, 5))
    for area, color, ls in [("Rural", "#27ae60", "-"),
                              ("Urban", "#8e44ad", "--")]:
        sub = area_monthly[area_monthly["Area"] == area]
        ax.plot(sub["Date"], sub["Unemployment_Rate"],
                label=area, color=color, linewidth=2, linestyle=ls,
                marker="o", markersize=3)

    covid_start = pd.Timestamp("2020-03-01")
    ax.axvline(covid_start, color=COVID_COLOR, linestyle=":", linewidth=1.5)
    ax.text(covid_start + pd.Timedelta(days=5),
            ax.get_ylim()[1]*0.9 if ax.get_ylim()[1] > 0 else 30,
            "COVID-19", color=COVID_COLOR, fontsize=9)

    ax.set(title="Rural vs Urban Unemployment Rate Over Time",
           xlabel="Date", ylabel="Unemployment Rate (%)")
    ax.legend(title="Area", fontsize=10)
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%b %Y"))
    plt.xticks(rotation=45)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    save(fig, "03_rural_vs_urban.png")


# ─────────────────────────────────────────────────────────────
# 6.  PLOT 4 — COVID Impact (Pre vs During)
# ─────────────────────────────────────────────────────────────
def plot_covid_impact(df1):
    df1 = df1.copy()
    df1["Period"] = df1["Date"].apply(
        lambda d: "During COVID\n(Mar–Dec 2020)"
        if d >= pd.Timestamp("2020-03-01")
        else "Pre-COVID\n(May 2019–Feb 2020)")

    state_period = (df1.groupby(["Region","Period"])["Unemployment_Rate"]
                       .mean()
                       .unstack())

    if "Pre-COVID\n(May 2019–Feb 2020)" not in state_period.columns:
        print("  [!] Pre-COVID column missing — skipping plot 4")
        return
    if "During COVID\n(Mar–Dec 2020)" not in state_period.columns:
        print("  [!] During-COVID column missing — skipping plot 4")
        return

    state_period["Change"] = (state_period["During COVID\n(Mar–Dec 2020)"]
                               - state_period["Pre-COVID\n(May 2019–Feb 2020)"])
    state_period = state_period.sort_values("Change", ascending=True)

    fig, ax = plt.subplots(figsize=(11, 9))
    colors = [COVID_COLOR if c > 0 else "#27ae60"
              for c in state_period["Change"]]
    bars = ax.barh(state_period.index, state_period["Change"],
                   color=colors, edgecolor="white", height=0.7)
    for bar in bars:
        val = bar.get_width()
        ax.text(val + (0.1 if val >= 0 else -0.1),
                bar.get_y() + bar.get_height()/2,
                f"{val:+.1f}%", va="center", fontsize=8,
                ha="left" if val >= 0 else "right")
    ax.axvline(0, color="black", linewidth=1)
    ax.set(title="Change in Unemployment Rate: Pre-COVID vs During COVID",
           xlabel="Change in Unemployment Rate (percentage points)")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    save(fig, "04_covid_impact.png")


# ─────────────────────────────────────────────────────────────
# 7.  PLOT 5 — Heatmap: State × Month
# ─────────────────────────────────────────────────────────────
def plot_heatmap(df1):
    df1 = df1.copy()
    df1["Month"] = df1["Date"].dt.to_period("M").astype(str)
    pivot = (df1.groupby(["Region","Month"])["Unemployment_Rate"]
                .mean()
                .unstack())
    pivot = pivot.reindex(sorted(pivot.columns), axis=1)

    fig, ax = plt.subplots(figsize=(16, 10))
    sns.heatmap(pivot, cmap="YlOrRd", linewidths=0.3, linecolor="white",
                annot=False, fmt=".1f", ax=ax,
                cbar_kws={"label": "Unemployment Rate (%)", "shrink": 0.7})
    ax.set(title="Unemployment Rate Heatmap — State × Month",
           xlabel="Month", ylabel="State")
    plt.xticks(rotation=60, ha="right", fontsize=7)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    save(fig, "05_heatmap_state_month.png")


# ─────────────────────────────────────────────────────────────
# 8.  PLOT 6 — Top 10 Most Affected States (COVID peak)
# ─────────────────────────────────────────────────────────────
def plot_top10_covid(df2):
    covid = df2[df2["Date"] >= "2020-03-01"]
    top10 = (covid.groupby("Region")["Unemployment_Rate"]
                  .max()
                  .sort_values(ascending=False)
                  .head(10))

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(top10.index, top10.values,
                  color=sns.color_palette("Reds_r", len(top10)),
                  edgecolor="black", width=0.6)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.4,
                f"{bar.get_height():.1f}%",
                ha="center", fontsize=9, fontweight="bold")
    ax.set(title="Top 10 States — Peak Unemployment Rate During COVID-19",
           xlabel="State", ylabel="Peak Unemployment Rate (%)")
    plt.xticks(rotation=30, ha="right")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    save(fig, "06_top10_covid_states.png")


# ─────────────────────────────────────────────────────────────
# 9.  PLOT 7 — Geo-region Comparison
# ─────────────────────────────────────────────────────────────
def plot_geo_region(df2):
    geo = (df2.groupby(["Date","Geo_Region"])["Unemployment_Rate"]
              .mean()
              .reset_index())

    fig, ax = plt.subplots(figsize=(13, 5))
    regions = geo["Geo_Region"].unique()
    palette = sns.color_palette(PALETTE, len(regions))
    for region, color in zip(sorted(regions), palette):
        sub = geo[geo["Geo_Region"] == region]
        ax.plot(sub["Date"], sub["Unemployment_Rate"],
                label=region, color=color, linewidth=2,
                marker="o", markersize=3)

    ax.axvline(pd.Timestamp("2020-03-01"), color=COVID_COLOR,
               linestyle="--", linewidth=1.5)
    ax.text(pd.Timestamp("2020-03-05"), ax.get_ylim()[1]*0.9,
            "COVID-19", color=COVID_COLOR, fontsize=9)
    ax.set(title="Unemployment Rate by Geographic Region (2020)",
           xlabel="Date", ylabel="Unemployment Rate (%)")
    ax.legend(title="Region", fontsize=9)
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%b %Y"))
    plt.xticks(rotation=30)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    save(fig, "07_geo_region_trend.png")


# ─────────────────────────────────────────────────────────────
# 10. PLOT 8 — Labour Participation vs Unemployment (scatter)
# ─────────────────────────────────────────────────────────────
def plot_scatter_labour(df1):
    df1 = df1.copy()
    df1["Period"] = df1["Date"].apply(
        lambda d: "During COVID" if d >= pd.Timestamp("2020-03-01")
        else "Pre-COVID")

    fig, ax = plt.subplots(figsize=(9, 6))
    for period, color, marker in [
            ("Pre-COVID", PRE_COLOR, "o"),
            ("During COVID", COVID_COLOR, "^")]:
        sub = df1[df1["Period"] == period]
        ax.scatter(sub["Labour_Participation"], sub["Unemployment_Rate"],
                   c=color, label=period, alpha=0.55, s=40,
                   marker=marker, edgecolors="black", linewidths=0.2)

    # Regression line
    x = df1["Labour_Participation"]
    y = df1["Unemployment_Rate"]
    m, b = np.polyfit(x, y, 1)
    x_line = np.linspace(x.min(), x.max(), 200)
    ax.plot(x_line, m*x_line + b, color="black",
            linestyle="--", linewidth=1.3, label=f"Trend (slope={m:.2f})")

    ax.set(title="Labour Participation Rate vs Unemployment Rate",
           xlabel="Labour Participation Rate (%)",
           ylabel="Unemployment Rate (%)")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    save(fig, "08_labour_vs_unemployment.png")


# ─────────────────────────────────────────────────────────────
# 11. PLOT 9 — Monthly Distribution (Boxplot)
# ─────────────────────────────────────────────────────────────
def plot_monthly_boxplot(df1):
    df1 = df1.copy()
    df1["Month_Name"] = df1["Date"].dt.strftime("%b")
    month_order = ["Jan","Feb","Mar","Apr","May","Jun",
                   "Jul","Aug","Sep","Oct","Nov","Dec"]
    df1["Month_Name"] = pd.Categorical(df1["Month_Name"],
                                        categories=month_order, ordered=True)

    fig, ax = plt.subplots(figsize=(12, 5))
    sns.boxplot(data=df1, x="Month_Name", y="Unemployment_Rate",
                palette="coolwarm", ax=ax, linewidth=1.2)
    ax.set(title="Monthly Distribution of Unemployment Rate",
           xlabel="Month", ylabel="Unemployment Rate (%)")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    save(fig, "09_monthly_boxplot.png")


# ─────────────────────────────────────────────────────────────
# 12. PLOT 10 — Employed vs Unemployed (area chart)
# ─────────────────────────────────────────────────────────────
def plot_employment_area(df2):
    monthly = df2.groupby("Date")[["Employed","Labour_Participation"]].mean().reset_index()
    monthly["Unemployed_Est"] = (monthly["Employed"]
                                  * monthly["Labour_Participation"] / 100
                                  * monthly["Labour_Participation"] / 100)

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.fill_between(monthly["Date"], monthly["Employed"] / 1e6,
                    alpha=0.6, color="#27ae60", label="Estimated Employed (M)")
    ax.plot(monthly["Date"], monthly["Employed"] / 1e6,
            color="#27ae60", linewidth=2)
    ax.axvline(pd.Timestamp("2020-03-01"), color=COVID_COLOR,
               linestyle="--", linewidth=1.6)
    ax.text(pd.Timestamp("2020-03-05"),
            (monthly["Employed"] / 1e6).max() * 0.95,
            "COVID-19\nLockdown", color=COVID_COLOR, fontsize=9)
    ax.set(title="Estimated Employed Population Over Time (2020)",
           xlabel="Date", ylabel="Employed (Millions)")
    ax.legend(fontsize=10)
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%b %Y"))
    plt.xticks(rotation=30)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    save(fig, "10_employment_trend.png")


# ─────────────────────────────────────────────────────────────
# 13. KEY INSIGHTS
# ─────────────────────────────────────────────────────────────
def print_insights(df1, df2):
    print("\n" + "=" * 58)
    print("  KEY INSIGHTS")
    print("=" * 58)

    pre  = df1[df1["Date"] <  "2020-03-01"]["Unemployment_Rate"].mean()
    post = df1[df1["Date"] >= "2020-03-01"]["Unemployment_Rate"].mean()
    peak = df1["Unemployment_Rate"].max()
    peak_date  = df1.loc[df1["Unemployment_Rate"].idxmax(), "Date"]
    peak_state = df1.loc[df1["Unemployment_Rate"].idxmax(), "Region"]

    top_state  = (df1.groupby("Region")["Unemployment_Rate"]
                     .mean().idxmax())
    low_state  = (df1.groupby("Region")["Unemployment_Rate"]
                     .mean().idxmin())

    rural_avg  = df1[df1["Area"]=="Rural"]["Unemployment_Rate"].mean()
    urban_avg  = df1[df1["Area"]=="Urban"]["Unemployment_Rate"].mean()

    print(f"\n  Pre-COVID avg unemployment   :  {pre:.2f}%")
    print(f"  During-COVID avg unemployment:  {post:.2f}%")
    print(f"  Increase due to COVID        :  +{post-pre:.2f} percentage points")
    print(f"\n  Peak unemployment rate       :  {peak:.2f}%")
    print(f"  Peak date                    :  {peak_date.date()}")
    print(f"  Peak state                   :  {peak_state}")
    print(f"\n  Highest avg unemployment     :  {top_state}")
    print(f"  Lowest  avg unemployment     :  {low_state}")
    print(f"\n  Rural avg unemployment       :  {rural_avg:.2f}%")
    print(f"  Urban avg unemployment       :  {urban_avg:.2f}%")
    print(f"  Urban is higher by           :  {urban_avg-rural_avg:+.2f}%")
    print()


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():
    # Load
    df1, df2 = load_and_clean()

    # Plots
    print("\n[Generating plots …]")
    plot_overall_trend(df1)
    plot_statewise_avg(df1)
    plot_rural_urban(df1)
    plot_covid_impact(df1)
    plot_heatmap(df1)
    plot_top10_covid(df2)
    plot_geo_region(df2)
    plot_scatter_labour(df1)
    plot_monthly_boxplot(df1)
    plot_employment_area(df2)

    # Insights
    print_insights(df1, df2)

    print("=" * 58)
    print("  All plots saved to  →  plots/")
    print("=" * 58)


if __name__ == "__main__":
    main()
