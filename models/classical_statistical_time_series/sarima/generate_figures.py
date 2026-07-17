"""Generate presentation figures from the recorded SARIMA/SARIMAX notebooks."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).parent
BASE = ROOT / "figures" / "baseline"
SARIMA = ROOT / "figures" / "model_experiment_sarima"
SARIMAX = ROOT / "figures" / "model_experiment_sarimax"
for directory in (BASE, SARIMA, SARIMAX):
    directory.mkdir(parents=True, exist_ok=True)

BLUE, GREEN, ORANGE, RED, PURPLE, SLATE = "#2563EB", "#16A34A", "#F59E0B", "#DC2626", "#7C3AED", "#475569"
SEASONAL_NAIVE, BASELINE_WMAE = 1800.1736, 1856.8605
ORDERS = [(p, d, q) for p in range(3) for d in range(2) for q in range(3)]
SARIMA_LAST = [15952.3237,15911.9607,15195.4592,2941.1402,2183.7088,1846.0585,5573.9570,2667.0633,1831.6176,2776.0371,1856.8605,2310.0948,4080.6585,2621.5860,6095.5643,2715.9486,2132.7671,1889.0005]
SARIMA_BLEND = [15952.3237,15911.9346,15197.5197,3149.6627,2425.3955,1994.0740,5657.3866,2886.6423,2004.4807,2995.4548,2050.0330,2551.2186,4215.6973,2843.4550,6168.5405,2939.1328,2373.5821,2040.4990]
SARIMAX_LAST = [2563.6915,2686.9430,2768.5633,3840.8970,3545.5354,3863.5144,2793.6988,2900.3397,2997.0660,3630.4139,3656.7977,4306.2966,2936.6539,2948.3561,2691.2308,3660.7590,3625.5305,3956.4040]


def save(fig, directory, name, rect=None):
    fig.tight_layout(rect=rect)
    fig.savefig(directory / name, dpi=230, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def flow(directory, name, title, steps, colors):
    fig, ax = plt.subplots(figsize=(17, 5)); ax.axis("off")
    xs = np.linspace(.06, .94, len(steps))
    for i, (x, label, color) in enumerate(zip(xs, steps, colors)):
        ax.text(x, .5, label, ha="center", va="center", transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=.65", fc="white", ec=color, lw=2.4))
        if i < len(steps) - 1:
            ax.annotate("", xy=(xs[i+1]-.06, .5), xytext=(x+.06, .5), xycoords=ax.transAxes,
                        arrowprops=dict(arrowstyle="->", lw=2.1, color=SLATE))
    ax.set_title(title, fontsize=20, weight="bold")
    save(fig, directory, name)


def bars(directory, name, title, labels, values, colors, note=None, ylim=None):
    fig, ax = plt.subplots(figsize=(11, 6)); objects = ax.bar(labels, values, color=colors)
    ax.set(title=title, ylabel="Validation WMAE — lower is better")
    if ylim: ax.set_ylim(*ylim)
    for bar, value in zip(objects, values):
        ax.text(bar.get_x()+bar.get_width()/2, value + max(values)*.018, f"{value:,.2f}", ha="center", weight="bold")
    if note: ax.text(.5, .90, note, transform=ax.transAxes, ha="center", color=RED, weight="bold")
    ax.grid(axis="y", alpha=.2); save(fig, directory, name)


def table(directory, name, title, rows, columns=("Item", "Recorded value"), color=BLUE):
    fig, ax = plt.subplots(figsize=(11, 8)); ax.axis("off")
    grid = ax.table(cellText=rows, colLabels=columns, cellLoc="left", colLoc="left", loc="center")
    grid.auto_set_font_size(False); grid.set_fontsize(11.5); grid.scale(1, 1.65)
    for (r, _), cell in grid.get_celld().items():
        cell.set_edgecolor("#CBD5E1"); cell.set_facecolor(color if r == 0 else ("#F8FAFC" if r % 2 == 0 else "white"))
        if r == 0: cell.set_text_props(color="white", weight="bold")
    ax.set_title(title, fontsize=20, weight="bold"); save(fig, directory, name)


def baseline_figures():
    bars(BASE, "01_validation_benchmark.png", "Recorded SARIMA-folder baseline benchmark",
         ["Seasonal naive\n52-week", "Aggregate model\n(1,1,1)"], [SEASONAL_NAIVE, BASELINE_WMAE], [GREEN, ORANGE],
         "Aggregate model is 3.15% worse", (0, 2150))
    flow(BASE, "02_training_inference_flow.png", "Aggregate baseline training and inference flow",
         ["421,570 Store–Dept\nrows", "Aggregate by Date\n143 weekly totals", "SARIMAX engine\norder (1,1,1)", "Forecast 39\nweekly totals", "Last-year Store–Dept\nshare", "Row predictions"],
         [BLUE, BLUE, PURPLE, ORANGE, GREEN, GREEN])
    table(BASE, "03_baseline_configuration.png", "Baseline recorded configuration", [
        ["Order", "(1,1,1)"], ["Validation", "Final 39 weeks"], ["Training", "104 weekly totals"],
        ["Allocation", "Last-year share + series-mean fallback"], ["Validation WMAE", "1,856.86"],
        ["Seasonal-naive WMAE", "1,800.17"], ["Explicit seasonal order", "Not configured"],
        ["External regressors", "None"],
    ])
    flow(BASE, "04_architecture_truth.png", "Why the current baseline is not a true seasonal SARIMA",
         ["Class name\nSARIMAX", "order=(1,1,1)\nnon-seasonal", "seasonal_order\nnot supplied", "No (P,D,Q,52)\ncomponent", "Architecture behaves\nlike aggregate ARIMA"],
         [PURPLE, BLUE, RED, RED, ORANGE])


def sarima_figures():
    labels=["Seasonal naive", "Baseline\n(1,1,1)", "Best search\n(1,0,2)"]
    bars(SARIMA, "01_result_progression.png", "SARIMA-folder order-search progression", labels,
         [SEASONAL_NAIVE, BASELINE_WMAE, 1831.6176], [GREEN, ORANGE, BLUE], "Order tuning improves baseline, but not seasonal naive", (0, 2150))
    idx=np.argsort(SARIMA_LAST)[:10][::-1]
    fig,ax=plt.subplots(figsize=(11,7)); vals=np.array(SARIMA_LAST)[idx]; labs=[str(ORDERS[i]) for i in idx]
    ax.barh(labs,vals,color=[GREEN if v==min(SARIMA_LAST) else BLUE for v in vals]); ax.axvline(SEASONAL_NAIVE,color=RED,ls="--",lw=2,label="Seasonal naive")
    ax.set(xlim=(1780,3100),title="Top recorded non-seasonal order candidates",xlabel="Validation WMAE",ylabel="Order (p,d,q)")
    for y,v in enumerate(vals): ax.text(v+4,y,f"{v:,.1f}",va="center",fontsize=9)
    ax.grid(axis="x",alpha=.2); ax.legend(); save(fig,SARIMA,"02_order_leaderboard.png")
    x=np.arange(len(ORDERS)); fig,ax=plt.subplots(figsize=(16,7)); ax.plot(x,SARIMA_LAST,"o-",label="Last-year share",color=BLUE); ax.plot(x,SARIMA_BLEND,"o-",label="Blended share",color=ORANGE)
    ax.axhline(SEASONAL_NAIVE,color=GREEN,ls="--",label="Seasonal naive"); ax.set_xticks(x); ax.set_xticklabels([str(o) for o in ORDERS],rotation=55,ha="right",fontsize=9)
    ax.set(title="All 18 orders and two allocation strategies",xlabel="Order (p,d,q)",ylabel="Validation WMAE"); ax.set_ylim(1700,6000); ax.grid(alpha=.2); ax.legend(); save(fig,SARIMA,"03_orders_and_allocations.png")
    table(SARIMA,"04_best_result_table.png","Best SARIMA-folder experiment",[
        ["Best order","(1,0,2)"],["Allocation","last_year_share"],["Validation WMAE","1,831.62"],
        ["Gain vs baseline","25.24 WMAE"],["Gap vs seasonal naive","+31.44 WMAE / 1.75% worse"],
        ["Orders tested","18"],["Allocation methods","2"],["Explicit seasonal order","Disabled / absent"],
    ])
    flow(SARIMA,"05_search_architecture.png","What the SARIMA-folder experiment actually searched",
         ["p ∈ {0,1,2}","d ∈ {0,1}","q ∈ {0,1,2}","18 (p,d,q) orders","2 allocation methods","36 evaluations","No (P,D,Q,52) search"],
         [BLUE,BLUE,BLUE,PURPLE,ORANGE,GREEN,RED])
    fig,ax=plt.subplots(figsize=(14,5)); ax.axis("off")
    for x,label in zip(np.linspace(.13,.87,4),["One aggregate\ntime series","No Store–Dept model\ndynamics","No explicit annual\nseasonal order","52-week structure enters\nonly through allocation"]):
        ax.text(x,.5,label,ha="center",va="center",transform=ax.transAxes,bbox=dict(boxstyle="round,pad=.75",fc="#FEF2F2",ec=RED,lw=2.4))
    ax.set_title("Main limitations of the current SARIMA experiment",fontsize=20,weight="bold"); save(fig,SARIMA,"06_limitations.png")


def sarimax_figures():
    bars(SARIMAX,"01_model_comparison.png","Recorded SARIMAX-folder comparison",
         ["Seasonal naive","Best aggregate\norder search","Best SARIMAX\nwith exog"],[SEASONAL_NAIVE,1831.6176,2563.6915],[GREEN,BLUE,RED],
         "Aggregate exogenous specification degraded validation WMAE",(0,3000))
    idx=np.argsort(SARIMAX_LAST)[:10][::-1]; vals=np.array(SARIMAX_LAST)[idx]; labs=[str(ORDERS[i]) for i in idx]
    fig,ax=plt.subplots(figsize=(11,7)); ax.barh(labs,vals,color=[GREEN if v==min(SARIMAX_LAST) else PURPLE for v in vals]); ax.axvline(SEASONAL_NAIVE,color=RED,ls="--",lw=2,label="Seasonal naive")
    ax.set(title="SARIMAX top 10 orders with external regressors",xlabel="Validation WMAE",ylabel="Order (p,d,q)")
    for y,v in enumerate(vals): ax.text(v+10,y,f"{v:,.1f}",va="center",fontsize=9)
    ax.grid(axis="x",alpha=.2); ax.legend(); save(fig,SARIMAX,"02_order_leaderboard.png")
    flow(SARIMAX,"03_feature_engineering_flow.png","SARIMAX feature-engineering flow",
         ["features.csv\nStore × Date","Fill Markdown=0\nother numeric=median","Aggregate each week\nmeans + sums","Calendar sin/cos\nmonth + week","Training-only correlation\nand collinearity filter","Up to 8 exogenous\nfeatures"],
         [BLUE,ORANGE,BLUE,PURPLE,PURPLE,GREEN])
    table(SARIMAX,"04_feature_table.png","Candidate aggregate SARIMAX features",[
        ["holiday_share","Mean IsHoliday across stores"],["Temperature","Weekly store mean"],["Fuel_Price","Weekly store mean"],
        ["CPI","Weekly store mean"],["Unemployment","Weekly store mean"],["MarkDown1–5","Weekly sums"],
        ["total_markdown","Sum of Markdown1–5"],["month_sin / month_cos","Calendar cyclic encoding"],
        ["week_sin / week_cos","Week-of-year cyclic encoding"],
    ],columns=("Feature","Construction"),color=PURPLE)
    flow(SARIMAX,"05_architecture.png","Current SARIMAX architecture",
         ["Weekly aggregate\nsales target","Selected future\nexogenous matrix","SARIMAX engine\n(p,d,q)","seasonal_order\n(0,0,0,0)","39 aggregate\nforecasts","Last-year / blended\nallocation","Store–Dept rows"],
         [BLUE,PURPLE,PURPLE,RED,ORANGE,GREEN,GREEN])
    table(SARIMAX,"06_best_result_table.png","Best recorded SARIMAX-folder result",[
        ["Best order","(0,0,0)"],["External regressors","Enabled"],["Allocation","last_year_share"],
        ["Validation WMAE","2,563.69"],["Seasonal-naive WMAE","1,800.17"],["Gap","+763.52 / 42.41% worse"],
        ["Feature-selection threshold","|correlation| ≥ 0.05"],["Maximum exogenous features","8"],["Explicit seasonal order","(0,0,0,0): disabled"],
    ],color=PURPLE)
    fig,ax=plt.subplots(figsize=(15,5)); ax.axis("off")
    for x,label in zip(np.linspace(.12,.88,4),["Aggregate features lose\nStore–Dept interactions","Raw Markdown totals\nare highly skewed","Correlation selection on\n104 weeks is unstable","No explicit annual\n(P,D,Q,52) component"]):
        ax.text(x,.5,label,ha="center",va="center",transform=ax.transAxes,bbox=dict(boxstyle="round,pad=.75",fc="#FEF2F2",ec=RED,lw=2.4))
    ax.set_title("Why the recorded SARIMAX experiment was weak",fontsize=20,weight="bold"); save(fig,SARIMAX,"07_failure_analysis.png")


if __name__ == "__main__":
    baseline_figures(); sarima_figures(); sarimax_figures()
    print(f"Generated SARIMA/SARIMAX figures under {ROOT / 'figures'}")
