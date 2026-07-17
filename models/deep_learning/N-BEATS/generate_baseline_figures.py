"""Generate presentation figures from the recorded N-BEATS baseline run."""

import base64
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).parent
OUT = ROOT / "figures" / "baseline"
OUT.mkdir(parents=True, exist_ok=True)
BLUE, GREEN, ORANGE, RED, PURPLE, SLATE = "#2563EB", "#16A34A", "#F59E0B", "#DC2626", "#7C3AED", "#475569"
EPOCHS = np.arange(1, 31)
TRAIN = np.array([.52564,.48887,.47662,.46865,.46304,.45853,.44676,.44219,.43948,.43696,.43067,.42836,.42689,.42570,.42229,.42124,.42037,.41968,.41818,.41784,.41716,.41684,.41602,.41577,.41560,.41534,.41512,.41477,.41467,.41464])
VAL_WMAE = np.array([2222.4863,2157.9829,2191.0672,2187.5795,2232.7754,2276.7705,2259.4241,2272.4103,2242.9868,2255.7016,2243.6740,2241.0964,2248.6327,2261.2279,2231.4239,2247.4403,2239.8265,2239.5547,2235.9763,2235.4153,2246.5828,2244.8094,2238.8615,2238.8336,2245.2901,2241.1469,2243.4024,2245.9230,2243.7287,2244.5165])
VAL_MAE = np.array([2201.2600,2128.4329,2167.0232,2156.6147,2204.9548,2245.2104,2230.0337,2242.9980,2215.7717,2222.7783,2212.0752,2207.9929,2222.3823,2230.0217,2204.9895,2217.7820,2209.0139,2211.6094,2206.8523,2207.5635,2218.0393,2216.5146,2210.3735,2211.5884,2217.2983,2213.0525,2215.6760,2217.8774,2215.4714,2216.0029])


def save(fig, name, rect=None):
    fig.tight_layout(rect=rect)
    fig.savefig(OUT/name, dpi=230, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def learning_curve():
    fig, axes = plt.subplots(1,2,figsize=(15,6))
    axes[0].plot(EPOCHS,TRAIN,color=BLUE,lw=2.5,marker="o",ms=3); axes[0].set(title="Training L1 on normalized log-sales",xlabel="Epoch",ylabel="L1 loss")
    axes[1].plot(EPOCHS,VAL_WMAE,color=ORANGE,lw=2.5,marker="o",ms=3); axes[1].scatter([2],[VAL_WMAE[1]],color=GREEN,s=100,zorder=4,label="Best epoch 2")
    axes[1].set(title="Validation WMAE",xlabel="Epoch",ylabel="WMAE — lower is better"); axes[1].legend()
    for ax in axes: ax.grid(alpha=.2)
    fig.suptitle("N-BEATS baseline learning curves",fontsize=19,weight="bold"); save(fig,"01_learning_curves.png")


def benchmark():
    labels=["Previous 32-week\nblock repeat","True seasonal naive\nlag 52","N-BEATS baseline\nbest epoch 2"]; vals=[3902.8521,1831.6223,2157.9829]
    fig,ax=plt.subplots(figsize=(9,6)); bars=ax.bar(labels,vals,color=[SLATE,GREEN,BLUE]); ax.set(title="Baseline validation benchmark",ylabel="WMAE — lower is better")
    for b,v in zip(bars,vals): ax.text(b.get_x()+b.get_width()/2,v+60,f"{v:,.1f}",ha="center",weight="bold")
    ax.grid(axis="y",alpha=.2); save(fig,"02_nbeats_vs_seasonal_naive.png")


def reference_correction():
    labels=["Recorded notebook reference\nrepeat previous 32 values","Correct annual reference\nexact lag 52"]
    vals=[3902.8521,1831.6223]
    fig,ax=plt.subplots(figsize=(10,6)); bars=ax.bar(labels,vals,color=[RED,GREEN]); ax.set_ylabel("WMAE — lower is better"); ax.set_title("Correcting the N-BEATS reference baseline")
    for b,v in zip(bars,vals): ax.text(b.get_x()+b.get_width()/2,v+55,f"{v:,.2f}",ha="center",weight="bold")
    ax.text(.5,.12,"The old 3,902.85 value was y(t−32), not y(t−52)",transform=ax.transAxes,ha="center",weight="bold",color=RED); ax.grid(axis="y",alpha=.2)
    save(fig,"12_reference_baseline_correction.png")


def overlapping_windows():
    fig,ax=plt.subplots(figsize=(15,6)); ax.set_xlim(0,88); ax.set_ylim(0,5); ax.axis("off")
    for row,start in enumerate([1,2,3],1):
        y=4.3-row
        ax.broken_barh([(start,52)],(y-.25,.5),facecolors=BLUE)
        ax.broken_barh([(start+52,32)],(y-.25,.5),facecolors=ORANGE)
        ax.text(0,y,f"Window {row}",ha="right",va="center",weight="bold")
        ax.text(start+26,y,"52-week input",ha="center",va="center",color="white",weight="bold")
        ax.text(start+68,y,"32-week target",ha="center",va="center",color="white",weight="bold")
    ax.text(44,.4,"Adjacent inputs share 51 of 52 weeks → many samples, limited independent information",ha="center",fontsize=13,weight="bold",color=RED)
    ax.set_title("How stride-1 training windows overlap",fontsize=20,weight="bold"); save(fig,"13_sliding_window_overlap.png")


def capacity_breakdown():
    labels=["Input layer\n52→256","Three hidden\n256→256","Backcast head\n256→52","Forecast head\n256→32"]
    vals=[13568,197376,13364,8224]
    fig,ax=plt.subplots(figsize=(12,6)); bars=ax.bar(labels,vals,color=[BLUE,PURPLE,RED,GREEN]); ax.set(title="Trainable parameters inside one baseline block",ylabel="Parameter count")
    for b,v in zip(bars,vals): ax.text(b.get_x()+b.get_width()/2,v+4500,f"{v:,}",ha="center",weight="bold")
    ax.text(.98,.92,"232,532 per block\n× 4 blocks = 930,128 total",transform=ax.transAxes,ha="right",va="top",fontsize=13,weight="bold",bbox=dict(boxstyle="round,pad=.5",fc="#F8FAFC",ec=SLATE)); ax.grid(axis="y",alpha=.2)
    save(fig,"14_parameter_capacity_breakdown.png")


def block_contributions():
    fig,ax=plt.subplots(figsize=(16,6)); ax.axis("off")
    xs=[.10,.30,.50,.70]; colors=[BLUE,PURPLE,ORANGE,RED]
    for i,(x,c) in enumerate(zip(xs,colors),1):
        ax.text(x,.65,f"Block {i}\npartial forecast f{i}\n32 values",ha="center",va="center",transform=ax.transAxes,bbox=dict(boxstyle="round,pad=.65",fc="white",ec=c,lw=2.5))
        ax.annotate("",xy=(.86,.35),xytext=(x,.53),xycoords=ax.transAxes,arrowprops=dict(arrowstyle="->",color=GREEN,lw=2))
    ax.text(.86,.35,"Σ",ha="center",va="center",transform=ax.transAxes,fontsize=25,weight="bold",bbox=dict(boxstyle="circle,pad=.35",fc="#F0FDF4",ec=GREEN,lw=3))
    ax.text(.86,.10,"Final forecast = f₁ + f₂ + f₃ + f₄",ha="center",transform=ax.transAxes,fontsize=14,weight="bold")
    ax.text(.5,.92,"Conceptual decomposition — numeric per-block outputs were not persisted",ha="center",transform=ax.transAxes,color=SLATE)
    ax.set_title("How block forecasts combine",fontsize=20,weight="bold"); save(fig,"15_block_forecast_aggregation.png")


def residual_mechanics():
    fig,ax=plt.subplots(figsize=(16,5)); ax.axis("off")
    labels=["Original input\nx₁","Residual after block 1\nx₂=x₁−b₁","Residual after block 2\nx₃=x₂−b₂","Residual after block 3\nx₄=x₃−b₃","Final unexplained\nresidual"]
    xs=np.linspace(.08,.92,len(labels))
    for i,(x,t) in enumerate(zip(xs,labels)):
        ax.text(x,.5,t,ha="center",va="center",transform=ax.transAxes,bbox=dict(boxstyle="round,pad=.65",fc="#FAF5FF",ec=PURPLE,lw=2))
        if i<len(labels)-1: ax.annotate("",xy=(xs[i+1]-.075,.5),xytext=(x+.075,.5),xycoords=ax.transAxes,arrowprops=dict(arrowstyle="->",lw=2,color=SLATE))
    ax.set_title("Backcast residual path across N-BEATS blocks",fontsize=20,weight="bold"); save(fig,"16_residual_path_across_blocks.png")


def objective_mismatch():
    fig,ax=plt.subplots(figsize=(15,5)); ax.axis("off")
    ax.text(.15,.55,"Training objective\nMAE on normalized log-sales\nall weeks equal",ha="center",va="center",transform=ax.transAxes,bbox=dict(boxstyle="round,pad=.8",fc="#EFF6FF",ec=BLUE,lw=3))
    ax.text(.50,.55,"Model updates optimize\nsmall normalized errors",ha="center",va="center",transform=ax.transAxes,bbox=dict(boxstyle="round,pad=.8",fc="#FAF5FF",ec=PURPLE,lw=3))
    ax.text(.85,.55,"Validation objective\nWMAE on dollar sales\nholidays weighted ×5",ha="center",va="center",transform=ax.transAxes,bbox=dict(boxstyle="round,pad=.8",fc="#FFF7ED",ec=ORANGE,lw=3))
    for a,b in [(.25,.40),(.60,.75)]: ax.annotate("",xy=(b,.55),xytext=(a,.55),xycoords=ax.transAxes,arrowprops=dict(arrowstyle="->",lw=2.5,color=SLATE))
    ax.set_title("Why lower training loss may not lower validation WMAE",fontsize=20,weight="bold"); save(fig,"17_training_validation_objective_mismatch.png")


def horizon_mechanics():
    fig,ax=plt.subplots(figsize=(14,5)); ax.axis("off")
    for i in range(1,9):
        x=.08+(i-1)*.105; color=GREEN if i<=2 else (ORANGE if i<=5 else RED)
        ax.text(x,.48,f"h={i if i<8 else '32'}",ha="center",va="center",transform=ax.transAxes,bbox=dict(boxstyle="round,pad=.4",fc="white",ec=color,lw=2))
        if i<8: ax.annotate("",xy=(x+.075,.48),xytext=(x+.035,.48),xycoords=ax.transAxes,arrowprops=dict(arrowstyle="->",color=SLATE))
    ax.text(.5,.18,"All 32 future weeks are predicted directly; uncertainty generally grows farther from the observed context",ha="center",transform=ax.transAxes,fontsize=13,weight="bold")
    ax.set_title("Direct 32-week forecast horizon",fontsize=20,weight="bold"); save(fig,"18_forecast_horizon_mechanics.png")


def overfit():
    fig,ax=plt.subplots(figsize=(11,6)); ax.plot(EPOCHS,VAL_WMAE,color=ORANGE,lw=2.4,label="Validation WMAE"); ax.axvline(2,color=GREEN,ls="--",label="Best epoch = 2")
    ax.fill_between(EPOCHS,VAL_WMAE,VAL_WMAE[1],where=EPOCHS>2,color=RED,alpha=.12,label="Generalization deterioration")
    ax.set(title="Baseline begins overfitting after epoch 2",xlabel="Epoch",ylabel="Validation WMAE"); ax.grid(alpha=.2); ax.legend(); save(fig,"03_early_overfitting.png")


def preprocessing():
    fig,ax=plt.subplots(figsize=(16,4)); ax.axis("off")
    steps=["421,570 sales rows","Store–Dept weekly panel\n143 Fridays","Keep 2,921 series\n≥ 84 observations","Forward/backward fill\nthen zero fallback","log1p + per-series\nstandardization","52-week context →\n32-week target"]
    xs=np.linspace(.07,.93,len(steps))
    for i,(x,t) in enumerate(zip(xs,steps)):
        ax.text(x,.5,t,ha="center",va="center",transform=ax.transAxes,bbox=dict(boxstyle="round,pad=.7",fc="#EFF6FF",ec=BLUE,lw=2))
        if i<len(steps)-1: ax.annotate("",xy=(xs[i+1]-.065,.5),xytext=(x+.065,.5),xycoords=ax.transAxes,arrowprops=dict(arrowstyle="->",lw=2,color=SLATE))
    ax.set_title("N-BEATS baseline preprocessing",fontsize=18,weight="bold"); save(fig,"04_preprocessing_workflow.png")


def architecture():
    fig,ax=plt.subplots(figsize=(16,6)); ax.axis("off")
    ax.text(.08,.5,"52-week\nnormalized context",ha="center",va="center",transform=ax.transAxes,bbox=dict(boxstyle="round,pad=.8",fc="#EFF6FF",ec=BLUE,lw=3))
    for i,x in enumerate([.27,.43,.59,.75],1):
        ax.text(x,.5,f"Block {i}\n4 × Dense(256) + ReLU\nDropout 0.10\nBackcast + Forecast",ha="center",va="center",transform=ax.transAxes,bbox=dict(boxstyle="round,pad=.7",fc="#F5F3FF",ec=PURPLE,lw=2))
        prev=.08 if i==1 else [.27,.43,.59][i-2]; ax.annotate("",xy=(x-.06,.5),xytext=(prev+.06,.5),xycoords=ax.transAxes,arrowprops=dict(arrowstyle="->",lw=2,color=SLATE))
    ax.text(.92,.5,"Sum block forecasts\n→ 32 weeks",ha="center",va="center",transform=ax.transAxes,bbox=dict(boxstyle="round,pad=.8",fc="#F0FDF4",ec=GREEN,lw=3)); ax.annotate("",xy=(.85,.5),xytext=(.83,.5),xycoords=ax.transAxes,arrowprops=dict(arrowstyle="->",lw=2,color=SLATE))
    ax.set_title("N-BEATS baseline architecture",fontsize=19,weight="bold"); save(fig,"05_architecture.png")


def detailed_block_architecture():
    fig, ax = plt.subplots(figsize=(18, 10))
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 10)
    ax.axis("off")

    box = dict(boxstyle="round,pad=.45", linewidth=2)
    ax.text(1.25, 6.0, "Input window\n52 weeks", ha="center", va="center",
            fontsize=12, bbox={**box, "fc": "#EFF6FF", "ec": BLUE})

    block_x = [3.8, 7.3, 10.8, 14.3]
    for index, x in enumerate(block_x, 1):
        ax.add_patch(plt.Rectangle((x - 1.35, 4.15), 2.7, 3.7,
                                   facecolor="#FAF5FF", edgecolor=PURPLE,
                                   linewidth=2.3))
        ax.text(x, 7.5, f"N-BEATS Block {index}", ha="center", va="center",
                fontsize=12, weight="bold", color=PURPLE)
        ax.text(x, 6.7, "Dense(256) + ReLU", ha="center", va="center", fontsize=10,
                bbox=dict(boxstyle="round,pad=.25", fc="white", ec="#C4B5FD"))
        ax.text(x, 6.15, "× 4 layers", ha="center", va="center", fontsize=10, weight="bold")
        ax.text(x, 5.6, "Dropout 0.10", ha="center", va="center", fontsize=10,
                bbox=dict(boxstyle="round,pad=.25", fc="white", ec="#C4B5FD"))
        ax.text(x - .65, 4.65, "Backcast\n52", ha="center", va="center", fontsize=9,
                bbox=dict(boxstyle="round,pad=.25", fc="#FEF2F2", ec=RED))
        ax.text(x + .65, 4.65, "Forecast\n32", ha="center", va="center", fontsize=9,
                bbox=dict(boxstyle="round,pad=.25", fc="#F0FDF4", ec=GREEN))

    ax.annotate("", xy=(2.4, 6.0), xytext=(1.9, 6.0),
                arrowprops=dict(arrowstyle="->", lw=2.5, color=SLATE))
    for left, right in zip(block_x[:-1], block_x[1:]):
        ax.annotate("", xy=(right - 1.4, 6.0), xytext=(left + 1.4, 6.0),
                    arrowprops=dict(arrowstyle="->", lw=2.5, color=SLATE))
        ax.text((left + right) / 2, 6.35, "residual", ha="center", fontsize=9, color=SLATE)
        ax.text((left + right) / 2, 5.75, "xₙ₊₁ = xₙ − backcastₙ",
                ha="center", fontsize=9, color=RED)

    sum_x, sum_y = 16.8, 2.25
    ax.text(sum_x, sum_y, "Σ", ha="center", va="center", fontsize=25, weight="bold",
            bbox=dict(boxstyle="circle,pad=.35", fc="#ECFDF5", ec=GREEN, lw=2.5))
    for x in block_x:
        ax.annotate("", xy=(sum_x - .45, sum_y + .15), xytext=(x + .65, 4.25),
                    arrowprops=dict(arrowstyle="->", lw=1.8, color=GREEN,
                                    connectionstyle="arc3,rad=-.08"))

    ax.text(16.8, 6.0, "Final residual\n(not forecast output)", ha="center", va="center",
            fontsize=10, color=SLATE, bbox=dict(boxstyle="round,pad=.4", fc="#F8FAFC", ec=SLATE))
    ax.annotate("", xy=(16.0, 6.0), xytext=(15.7, 6.0),
                arrowprops=dict(arrowstyle="->", lw=2, color=SLATE))
    ax.text(16.8, .75, "Final 32-week forecast\n= f₁ + f₂ + f₃ + f₄",
            ha="center", va="center", fontsize=12, weight="bold",
            bbox=dict(boxstyle="round,pad=.55", fc="#F0FDF4", ec=GREEN, lw=2.5))
    ax.annotate("", xy=(16.8, 1.35), xytext=(16.8, 1.75),
                arrowprops=dict(arrowstyle="->", lw=2.5, color=GREEN))

    ax.text(1.0, 2.55, "Residual path", color=SLATE, weight="bold", fontsize=11)
    ax.plot([1.0, 2.0], [2.2, 2.2], color=SLATE, lw=2.5)
    ax.text(1.0, 1.55, "Forecast aggregation path", color=GREEN, weight="bold", fontsize=11)
    ax.plot([1.0, 2.0], [1.2, 1.2], color=GREEN, lw=2.5)
    ax.text(9, 9.45, "N-BEATS block structure and connections", ha="center",
            fontsize=21, weight="bold")
    ax.text(9, 8.95,
            "Each block explains part of the history with a backcast and contributes part of the future forecast",
            ha="center", fontsize=12, color=SLATE)
    save(fig, "11_detailed_block_architecture.png")


def parameter_table():
    rows=[["context_length","52"],["forecast_horizon","32"],["batch_size","128"],["max_epochs","30"],["optimizer","AdamW"],["learning_rate","0.001"],["weight_decay","0.0001"],["hidden_units","256"],["num_blocks","4"],["num_layers per block","4"],["dropout","0.10"],["loss","L1 on normalized log-sales"],["gradient clipping","1.0"],["holiday weight","5 (evaluation only)"]]
    fig,ax=plt.subplots(figsize=(10,9)); ax.axis("off"); table=ax.table(cellText=rows,colLabels=["Parameter","Baseline value"],cellLoc="left",colLoc="left",loc="center",colWidths=[.55,.45]); table.auto_set_font_size(False); table.set_fontsize(12); table.scale(1,1.6)
    for (r,c),cell in table.get_celld().items():
        cell.set_edgecolor("#CBD5E1"); cell.set_facecolor(BLUE if r==0 else ("#F8FAFC" if r%2==0 else "white"));
        if r==0: cell.set_text_props(color="white",weight="bold")
    ax.set_title("N-BEATS baseline configuration",fontsize=20,weight="bold"); save(fig,"06_hyperparameters_table.png")


def split():
    fig,ax=plt.subplots(figsize=(14,3)); ax.barh([0],[111],left=[0],color=BLUE,label="Training: 2010-02-05 to 2012-03-16"); ax.barh([0],[32],left=[111],color=ORANGE,label="Validation: last 32 weeks"); ax.set_xlim(0,143); ax.set_yticks([]); ax.set_xlabel("Weekly index"); ax.set_title("Chronological validation split — no random shuffling",weight="bold"); ax.legend(loc="lower center",bbox_to_anchor=(.5,-.55),ncol=2); save(fig,"07_chronological_split.png")


def data_summary():
    labels=["Weekly dates","Usable series","Training windows","Validation series"]; vals=[143,2921,81788,2921]
    fig,ax=plt.subplots(figsize=(10,6)); bars=ax.bar(labels,vals,color=[BLUE,GREEN,PURPLE,ORANGE]); ax.set_yscale("log"); ax.set_title("N-BEATS baseline data dimensions",weight="bold"); ax.set_ylabel("Count — logarithmic scale")
    for b,v in zip(bars,vals): ax.text(b.get_x()+b.get_width()/2,v*1.12,f"{v:,}",ha="center",weight="bold"); save(fig,"08_data_dimensions.png")


def dashboard():
    fig,axes=plt.subplots(2,2,figsize=(14,9)); axes=axes.flat
    axes[0].bar(["Previous 32-week\nrepeat","N-BEATS"],[3902.85,2157.98],color=[SLATE,GREEN]); axes[0].set_title("Validation WMAE")
    axes[1].plot(EPOCHS,VAL_WMAE,color=ORANGE); axes[1].axvline(2,color=GREEN,ls="--"); axes[1].set_title("Best epoch = 2")
    axes[2].plot(EPOCHS,TRAIN,color=BLUE); axes[2].set_title("Training loss keeps decreasing")
    axes[3].axis("off"); axes[3].text(.5,.5,"No additional features\nPure Store–Dept sales history\n\nValidation WMAE: 2,157.98\nKaggle WMAE: 4,700",ha="center",va="center",fontsize=16,bbox=dict(boxstyle="round,pad=1",fc="#F5F3FF",ec=PURPLE,lw=3))
    for ax in axes[:3]: ax.grid(alpha=.2)
    fig.suptitle("N-BEATS baseline dashboard",fontsize=20,weight="bold"); save(fig,"09_baseline_dashboard.png")


def extract_notebook_diagnostic():
    nb=json.loads((ROOT/"baseline_N-BEATS.ipynb").read_text())
    for output in nb["cells"][17].get("outputs",[]):
        data=output.get("data",{})
        if "image/png" in data:
            payload=data["image/png"]; payload="".join(payload) if isinstance(payload,list) else payload
            (OUT/"10_recorded_validation_diagnostics.png").write_bytes(base64.b64decode(payload)); return


if __name__=="__main__":
    learning_curve(); benchmark(); overfit(); preprocessing(); architecture(); parameter_table(); split(); data_summary(); dashboard(); extract_notebook_diagnostic(); detailed_block_architecture(); reference_correction(); overlapping_windows(); capacity_breakdown(); block_contributions(); residual_mechanics(); objective_mismatch(); horizon_mechanics()
    print(f"Generated baseline figures in {OUT}")
