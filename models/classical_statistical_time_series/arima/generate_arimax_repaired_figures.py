"""Generate presentation graphics for the repaired 50-order ARIMAX run."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).parent
OUT = ROOT / "figures" / "arimax_repaired"
OUT.mkdir(parents=True, exist_ok=True)
BLUE, GREEN, ORANGE, RED, PURPLE, SLATE = "#2563EB", "#16A34A", "#F59E0B", "#DC2626", "#7C3AED", "#475569"
SEASONAL_NAIVE, OLD_ARIMAX, PURE_ARIMA, KAGGLE = 1800.1736, 2563.6915, 1829.8800, 3200.0

VALUES = [
    15952.3233,15952.2889,15951.8164,15915.4702,15929.1974,3695.8199,1830.1113,1854.0596,1902.4908,1960.7045,
    4208.9964,1824.4816,1835.2707,2287.2427,2129.7850,3543.7731,1876.1873,1832.4395,1925.9041,1868.0885,
    3859.4077,1926.7086,3478.3896,2055.3510,2046.7295,3456.7824,1870.3414,1943.0403,1864.3769,1867.3321,
    3831.7135,2297.3082,3093.7297,1924.7280,2045.8723,3317.8184,3238.2880,2995.4492,1863.6190,1879.5669,
    3763.5419,3670.6685,3077.1364,3137.4893,1958.4302,3512.3860,3275.5380,3077.0954,1927.0116,1919.8670,
]


def frame():
    orders = [(p, d, q) for p in range(5) for d in range(2) for q in range(5)]
    return pd.DataFrame({"trial": range(50), "order": orders, "p": [x[0] for x in orders], "d": [x[1] for x in orders], "q": [x[2] for x in orders], "wmae": VALUES})


def save(fig, name, rect=None):
    fig.tight_layout(rect=rect)
    fig.savefig(OUT / name, dpi=230, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def result_progression():
    labels=["Seasonal naive","Pure ARIMA\n(1,0,2)","Old ARIMAX\n(0,0,0)","Repaired ARIMAX\n(1,0,1)","Repaired ARIMAX\nKaggle"]
    vals=[SEASONAL_NAIVE,PURE_ARIMA,OLD_ARIMAX,1824.4816,KAGGLE]; colors=[GREEN,BLUE,RED,PURPLE,ORANGE]
    fig,ax=plt.subplots(figsize=(13,6)); bars=ax.bar(labels,vals,color=colors); ax.set(title="ARIMAX repair: validation improved, Kaggle remained weak",ylabel="WMAE — lower is better")
    for b,v in zip(bars,vals): ax.text(b.get_x()+b.get_width()/2,v+55,f"{v:,.2f}",ha="center",weight="bold")
    ax.grid(axis="y",alpha=.2); save(fig,"01_result_progression.png")


def leaderboard():
    df=frame().nsmallest(15,"wmae").sort_values("wmae",ascending=True)
    fig,ax=plt.subplots(figsize=(11,8)); colors=[GREEN if v==df.wmae.min() else BLUE for v in df.wmae]
    bars=ax.barh([str(x) for x in df.order],df.wmae,color=colors); ax.axvline(SEASONAL_NAIVE,color=RED,ls="--",lw=2,label=f"Seasonal naive {SEASONAL_NAIVE:,.2f}")
    ax.set(xlim=(1780,1990),title="Repaired ARIMAX top 15 orders",xlabel="Validation WMAE",ylabel="Order (p,d,q)")
    for b,v in zip(bars,df.wmae): ax.text(v+3,b.get_y()+b.get_height()/2,f"{v:,.1f}",va="center",fontsize=9)
    ax.grid(axis="x",alpha=.2); ax.legend(); save(fig,"02_top15_order_leaderboard.png")


def heatmaps():
    df=frame(); fig,axes=plt.subplots(1,2,figsize=(14,6))
    for d,ax in enumerate(axes):
        mat=df[df.d==d].pivot(index="p",columns="q",values="wmae").to_numpy(); shown=np.clip(mat,1800,4500)
        im=ax.imshow(shown,cmap="RdYlGn_r",aspect="auto",vmin=1800,vmax=4500)
        ax.set_xticks(range(5)); ax.set_yticks(range(5)); ax.set(xlabel="q",ylabel="p",title=f"d={d}")
        for i in range(5):
            for j in range(5): ax.text(j,i,f"{mat[i,j]:,.0f}",ha="center",va="center",fontsize=8,weight="bold")
    fig.subplots_adjust(right=.88,top=.84,wspace=.16)
    cax=fig.add_axes([.90,.18,.018,.58]); fig.colorbar(im,cax=cax,label="WMAE (color clipped at 4,500)")
    fig.suptitle("ARIMAX validation WMAE across all 50 (p,d,q) orders",fontsize=19,weight="bold")
    fig.savefig(OUT/"03_p_d_q_heatmaps.png",dpi=230,bbox_inches="tight",facecolor="white"); plt.close(fig)


def order_stability():
    df=frame(); stable=df.wmae<2000; usable=df.wmae<3000
    labels=["< 2,000\ncompetitive","2,000–3,000\nweak","3,000–5,000\nunstable","> 5,000\nexploded"]
    vals=[stable.sum(),((df.wmae>=2000)&(df.wmae<3000)).sum(),((df.wmae>=3000)&(df.wmae<5000)).sum(),(df.wmae>=5000).sum()]
    fig,ax=plt.subplots(figsize=(10,6)); bars=ax.bar(labels,vals,color=[GREEN,ORANGE,RED,"#7F1D1D"]); ax.set(title="Stability of the 50-order search",ylabel="Number of orders")
    for b,v in zip(bars,vals): ax.text(b.get_x()+b.get_width()/2,v+.4,str(v),ha="center",weight="bold")
    ax.grid(axis="y",alpha=.2); save(fig,"04_order_stability_distribution.png")


def differencing():
    df=frame(); clipped=df.assign(clipped=df.wmae.clip(upper=5000)); stats=df.groupby("d").wmae.agg(["median","min"])
    fig,axes=plt.subplots(1,2,figsize=(13,6));
    for i,col in enumerate(["median","min"]):
        vals=stats[col].values; bars=axes[i].bar(["d=0","d=1"],vals,color=[BLUE,PURPLE]); axes[i].set_title(f"{col.title()} WMAE across orders"); axes[i].grid(axis="y",alpha=.2)
        for b,v in zip(bars,vals): axes[i].text(b.get_x()+b.get_width()/2,v+40,f"{v:,.1f}",ha="center",weight="bold")
    fig.suptitle("Differencing sensitivity",fontsize=19,weight="bold"); save(fig,"05_differencing_sensitivity.png")


def feature_flow():
    fig,ax=plt.subplots(figsize=(17,5)); ax.axis("off")
    steps=["features.csv\nStore × Date","Weekly aggregation\nmean holiday / sum markdown","log1p(total markdown)\nreduce extreme skew","Fourier seasonality\n1st + 2nd harmonics","Train-only standardization\n(x−μ)/σ","6 exogenous inputs\nto SARIMAX engine"]
    xs=np.linspace(.07,.93,len(steps))
    for i,(x,t) in enumerate(zip(xs,steps)):
        ax.text(x,.5,t,ha="center",va="center",transform=ax.transAxes,bbox=dict(boxstyle="round,pad=.65",fc="white",ec=PURPLE,lw=2.4))
        if i<len(steps)-1: ax.annotate("",xy=(xs[i+1]-.065,.5),xytext=(x+.065,.5),xycoords=ax.transAxes,arrowprops=dict(arrowstyle="->",lw=2,color=SLATE))
    ax.set_title("Repaired ARIMAX feature-engineering flow",fontsize=20,weight="bold"); save(fig,"06_feature_engineering_flow.png")


def feature_table():
    rows=[["week_sin","sin(2π·week/52)","Annual seasonal position"],["week_cos","cos(2π·week/52)","Annual seasonal position"],["week_sin_2","sin(4π·week/52)","Second seasonal harmonic"],["week_cos_2","cos(4π·week/52)","Second seasonal harmonic"],["holiday_share","Mean IsHoliday across stores","Known holiday signal"],["log_total_markdown","log1p(sum MarkDown1–5)","Promotion intensity with reduced skew"]]
    fig,ax=plt.subplots(figsize=(14,7)); ax.axis("off"); t=ax.table(cellText=rows,colLabels=["Feature","Construction","Purpose"],cellLoc="left",loc="center"); t.auto_set_font_size(False); t.set_fontsize(11); t.scale(1,1.8)
    for (r,c),cell in t.get_celld().items(): cell.set_edgecolor("#CBD5E1"); cell.set_facecolor(PURPLE if r==0 else ("#F8FAFC" if r%2==0 else "white")); cell.set_text_props(color="white" if r==0 else "black",weight="bold" if r==0 else "normal")
    ax.set_title("Selected exogenous features",fontsize=20,weight="bold"); save(fig,"07_selected_features_table.png")


def transformations():
    fig,ax=plt.subplots(figsize=(15,5)); ax.axis("off")
    labels=[("Aggregate target","z = log1p(TotalSales)",BLUE),("Exogenous scaling","x′ = (x−μtrain)/σtrain",PURPLE),("Model forecast","ẑ from ARIMAX",ORANGE),("Return to sales","ŷ = expm1(ẑ)",GREEN)]
    xs=np.linspace(.10,.90,4)
    for i,(x,(title,formula,color)) in enumerate(zip(xs,labels)):
        ax.text(x,.5,title+"\n\n"+formula,ha="center",va="center",transform=ax.transAxes,bbox=dict(boxstyle="round,pad=.75",fc="white",ec=color,lw=2.5))
        if i<3: ax.annotate("",xy=(xs[i+1]-.10,.5),xytext=(x+.10,.5),xycoords=ax.transAxes,arrowprops=dict(arrowstyle="->",lw=2.5,color=SLATE))
    ax.set_title("Target and feature transformations",fontsize=20,weight="bold"); save(fig,"08_target_and_scaling_transformations.png")


def architecture():
    fig,ax=plt.subplots(figsize=(17,5)); ax.axis("off")
    steps=["Weekly total sales\nlog1p target","6 standardized\nfuture exogenous features","ARIMAX(1,0,1)\nAR + MA + regression","39 weekly total\nforecasts","expm1 to sales scale","Last-year Store–Dept\nshare allocation","Row predictions"]
    xs=np.linspace(.05,.95,len(steps))
    for i,(x,t) in enumerate(zip(xs,steps)):
        ax.text(x,.5,t,ha="center",va="center",transform=ax.transAxes,bbox=dict(boxstyle="round,pad=.6",fc="white",ec=[BLUE,PURPLE,PURPLE,ORANGE,GREEN,GREEN,GREEN][i],lw=2.3))
        if i<len(steps)-1: ax.annotate("",xy=(xs[i+1]-.055,.5),xytext=(x+.055,.5),xycoords=ax.transAxes,arrowprops=dict(arrowstyle="->",lw=2,color=SLATE))
    ax.set_title("Best repaired ARIMAX architecture",fontsize=20,weight="bold"); save(fig,"09_best_arimax_architecture.png")


def best_table():
    rows=[["Best order","(1,0,1)"],["Validation WMAE","1,824.48"],["Seasonal naive WMAE","1,800.17"],["Gap vs seasonal naive","+24.31 / 1.35% worse"],["Old ARIMAX WMAE","2,563.69"],["Gain vs old ARIMAX","739.21 / 28.83%"],["Pure ARIMA WMAE","1,829.88"],["Gain vs pure ARIMA","5.40 / 0.30%"],["Kaggle score","3,200"],["Allocation","last_year_share"]]
    fig,ax=plt.subplots(figsize=(10,8)); ax.axis("off"); t=ax.table(cellText=rows,colLabels=["Item","Recorded value"],cellLoc="left",loc="center"); t.auto_set_font_size(False); t.set_fontsize(12); t.scale(1,1.65)
    for (r,c),cell in t.get_celld().items(): cell.set_edgecolor("#CBD5E1"); cell.set_facecolor(BLUE if r==0 else ("#F8FAFC" if r%2==0 else "white")); cell.set_text_props(color="white" if r==0 else "black",weight="bold" if r==0 else "normal")
    ax.set_title("Best repaired ARIMAX result",fontsize=20,weight="bold"); save(fig,"10_best_result_table.png")


def validation_kaggle_gap():
    vals=[1824.4816,3200]; fig,ax=plt.subplots(figsize=(9,6)); bars=ax.bar(["Validation WMAE","Kaggle WMAE"],vals,color=[GREEN,RED]); ax.set(title="Repaired ARIMAX generalization gap",ylabel="WMAE")
    for b,v in zip(bars,vals): ax.text(b.get_x()+b.get_width()/2,v+55,f"{v:,.1f}",ha="center",weight="bold")
    ax.text(.5,.82,"+1,375.5 WMAE / +75.4%",transform=ax.transAxes,ha="center",color=RED,fontsize=15,weight="bold"); ax.grid(axis="y",alpha=.2); save(fig,"11_validation_vs_kaggle_gap.png")


def failure_explanation():
    labels=["No differencing +\nweak dynamics","High p/q with only\n104 weekly observations","39-step recursive\nforecast instability","Aggregate covariates lose\nStore–Dept interactions"]
    fig,ax=plt.subplots(figsize=(15,5)); ax.axis("off")
    for x,t in zip(np.linspace(.12,.88,4),labels): ax.text(x,.5,t,ha="center",va="center",transform=ax.transAxes,bbox=dict(boxstyle="round,pad=.75",fc="#FEF2F2",ec=RED,lw=2.5))
    ax.set_title("Why several repaired ARIMAX orders still exploded",fontsize=20,weight="bold"); save(fig,"12_unstable_order_explanation.png")


if __name__ == "__main__":
    result_progression(); leaderboard(); heatmaps(); order_stability(); differencing(); feature_flow(); feature_table(); transformations(); architecture(); best_table(); validation_kaggle_gap(); failure_explanation()
    print(f"Generated repaired ARIMAX figures in {OUT}")
