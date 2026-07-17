"""Generate presentation figures from recorded N-BEATS experiments and Optuna run."""

import ast
import base64
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT=Path(__file__).parent; OUT=ROOT/"figures"/"model_experiment"; OUT.mkdir(parents=True,exist_ok=True)
BLUE,GREEN,ORANGE,RED,PURPLE,SLATE="#2563EB","#16A34A","#F59E0B","#DC2626","#7C3AED","#475569"
BASELINE=2157.9829
TRIALS={
0:(2191.4117,{'optimizer':'adam','loss_type':'regular_l1','batch_size':64,'learning_rate':.000130667,'hidden_units':128,'dropout':.141615,'weight_decay':1.1528e-6}),
1:(2400.4774,{'optimizer':'sgd','loss_type':'regular_l1','batch_size':128,'learning_rate':.001120761,'hidden_units':128,'dropout':.122371,'weight_decay':2.6211e-6}),
2:(2199.4592,{'optimizer':'adam','loss_type':'holiday_weighted_l1','batch_size':128,'learning_rate':.001530485,'hidden_units':256,'dropout':.034105,'weight_decay':1.5673e-6}),
3:(2229.2762,{'optimizer':'adam','loss_type':'regular_l1','batch_size':64,'learning_rate':.000108623,'hidden_units':128,'dropout':.181533,'weight_decay':.000651921}),
4:(2227.1291,{'optimizer':'adam','loss_type':'regular_l1','batch_size':64,'learning_rate':.009206469,'hidden_units':128,'dropout':.011099,'weight_decay':5.6574e-5}),
6:(2250.7428,{'optimizer':'adam','loss_type':'regular_l1','batch_size':64,'learning_rate':.000495755,'hidden_units':128,'dropout':.197426,'weight_decay':.000972858}),
7:(2220.8643,{'optimizer':'adam','loss_type':'regular_l1','batch_size':64,'learning_rate':.007538507,'hidden_units':128,'dropout':.070633,'weight_decay':1.5073e-5}),
}


def save(fig,name,rect=None): fig.tight_layout(rect=rect); fig.savefig(OUT/name,dpi=230,bbox_inches="tight",facecolor="white"); plt.close(fig)


def experiment_evolution():
    labels=["Baseline","Exp 1\nlow LR + ES","Exp 2\n78-week context","Exp 3\nholiday loss","Best Optuna"]
    vals=[2157.9829,2186.5015,2662.8061,2185.1366,2191.4117]; colors=[GREEN,BLUE,RED,ORANGE,PURPLE]
    fig,ax=plt.subplots(figsize=(12,6)); bars=ax.bar(labels,vals,color=colors); ax.set_ylim(2050,2750); ax.axhline(BASELINE,color=GREEN,ls="--",alpha=.7); ax.set(title="N-BEATS validation experiment evolution",ylabel="Best validation WMAE — lower is better")
    for b,v in zip(bars,vals): ax.text(b.get_x()+b.get_width()/2,v+18,f"{v:,.1f}",ha="center",weight="bold")
    ax.grid(axis="y",alpha=.2); save(fig,"01_experiment_wmae_evolution.png")


def kaggle_comparison():
    labels=["N-BEATS","DLinear","XGBoost"]; vals=[4700,3500,2806]
    fig,ax=plt.subplots(figsize=(9,6)); bars=ax.bar(labels,vals,color=[RED,ORANGE,GREEN]); ax.set(title="Kaggle WMAE comparison",ylabel="WMAE — lower is better")
    for b,v in zip(bars,vals): ax.text(b.get_x()+b.get_width()/2,v+70,f"{v:,}",ha="center",weight="bold")
    ax.grid(axis="y",alpha=.2); save(fig,"02_kaggle_model_comparison.png")


def optuna_leaderboard():
    ids=list(TRIALS); vals=[TRIALS[i][0] for i in ids]; colors=[GREEN if i==0 else BLUE for i in ids]
    fig,ax=plt.subplots(figsize=(12,6)); bars=ax.bar([f"Trial {i}" for i in ids],vals,color=colors); ax.axhline(BASELINE,color=RED,ls="--",label=f"Baseline = {BASELINE:,.1f}"); ax.set_ylim(2100,2460); ax.set(title="N-BEATS Optuna leaderboard — 7 completed, Trial 5 pruned",ylabel="Best validation WMAE")
    for b,v in zip(bars,vals): ax.text(b.get_x()+b.get_width()/2,v+10,f"{v:,.1f}",ha="center",fontsize=9)
    ax.legend(); ax.grid(axis="y",alpha=.2); save(fig,"03_optuna_trial_leaderboard.png")


def best_trial_curve():
    epochs=np.arange(1,16); train=np.array([.58965,.54097,.52763,.51977,.51408,.50946,.50564,.50236,.49937,.49703,.49470,.49293,.49089,.48933,.48778]); val=np.array([2288.6889,2275.3199,2262.7157,2273.7745,2240.0859,2244.2290,2216.4581,2229.7406,2216.7735,2214.4598,2191.4117,2218.9207,2226.3977,2220.1973,2211.8650])
    fig,axes=plt.subplots(1,2,figsize=(15,6)); axes[0].plot(epochs,train,color=BLUE,marker="o"); axes[0].set(title="Training loss",xlabel="Epoch",ylabel="Normalized L1")
    axes[1].plot(epochs,val,color=ORANGE,marker="o"); axes[1].scatter([11],[val[10]],color=GREEN,s=110,zorder=4,label="Best epoch 11"); axes[1].axhline(BASELINE,color=RED,ls="--",label="Baseline reference"); axes[1].set(title="Validation WMAE",xlabel="Epoch",ylabel="WMAE")
    for ax in axes: ax.grid(alpha=.2)
    axes[1].legend()
    fig.suptitle("Best Optuna candidate — Trial 0 learning curve",fontsize=18,weight="bold"); save(fig,"04_best_optuna_trial_curve.png")


def best_params_table():
    rows=[["optimizer","Adam"],["loss_type","regular_l1"],["batch_size","64"],["learning_rate","0.00013067"],["hidden_units","128"],["dropout","0.1416"],["weight_decay","0.00000115"],["context_length","52"],["forecast_horizon","32"],["num_blocks","4"],["layers per block","4"],["max_epochs","20"],["early stopping patience","4"],["best epoch","11"]]
    fig,ax=plt.subplots(figsize=(10,9)); ax.axis("off"); t=ax.table(cellText=rows,colLabels=["Hyperparameter","Best Trial 0 value"],cellLoc="left",colLoc="left",loc="center",colWidths=[.55,.45]); t.auto_set_font_size(False); t.set_fontsize(12); t.scale(1,1.6)
    for (r,c),cell in t.get_celld().items(): cell.set_edgecolor("#CBD5E1"); cell.set_facecolor(PURPLE if r==0 else ("#F8FAFC" if r%2==0 else "white")); cell.set_text_props(color="white" if r==0 else "black",weight="bold" if r==0 else "normal")
    ax.set_title("Best N-BEATS Optuna candidate hyperparameters",fontsize=20,weight="bold"); save(fig,"05_best_hyperparameters_table.png")


def sensitivity():
    numeric=["learning_rate","batch_size","hidden_units","dropout","weight_decay"]
    fig,axes=plt.subplots(2,3,figsize=(15,10)); axes=axes.flat
    for ax,name in zip(axes,numeric):
        xs=np.array([TRIALS[i][1][name] for i in TRIALS]); ys=np.array([TRIALS[i][0] for i in TRIALS]); ax.set_xscale("log" if name in ("learning_rate","weight_decay") else "linear"); ax.scatter(xs,ys,color=BLUE,s=70); ax.scatter([TRIALS[0][1][name]],[TRIALS[0][0]],color=GREEN,s=120)
        for i,x,y in zip(TRIALS,xs,ys): ax.annotate(f"T{i}",(x,y),xytext=(4,4),textcoords="offset points",fontsize=8)
        ax.set(title=name,ylabel="Validation WMAE"); ax.grid(alpha=.2)
    ax=axes[5]; categories=["Adam\nregular", "Adam\nholiday", "SGD\nregular"]; values=[np.mean([v[0] for v in TRIALS.values() if v[1]['optimizer']=='adam' and v[1]['loss_type']=='regular_l1']),TRIALS[2][0],TRIALS[1][0]]; ax.bar(categories,values,color=[BLUE,ORANGE,RED]); ax.set_ylim(2100,2450); ax.set_title("Optimizer / loss groups")
    fig.suptitle("N-BEATS hyperparameter sensitivity — completed Optuna trials",fontsize=18,weight="bold"); fig.text(.5,.01,"Only seven trials completed and Trial 5 was pruned; interpret sensitivity directionally.",ha="center",color=RED,weight="bold"); save(fig,"06_hyperparameter_sensitivity.png",rect=[0,.04,1,.96])


def experiment_flow():
    fig,ax=plt.subplots(figsize=(16,6)); ax.axis("off"); stages=[("Baseline","52-week context\nregular L1\nbest 2158"),("Experiment 1","lower learning rate\n+ early stopping\n2187"),("Experiment 2","78-week context\n2663 — rejected"),("Experiment 3","holiday-weighted loss\n2185"),("Final Optuna","optimizer/loss/capacity search\nbest candidate 2191")]; xs=np.linspace(.08,.92,5)
    for i,(x,(title,body)) in enumerate(zip(xs,stages)):
        edge=GREEN if i==0 else (RED if i==2 else PURPLE); ax.text(x,.5,title+"\n\n"+body,ha="center",va="center",transform=ax.transAxes,bbox=dict(boxstyle="round,pad=.8",fc="white",ec=edge,lw=3))
        if i<4: ax.annotate("",xy=(xs[i+1]-.08,.5),xytext=(x+.08,.5),xycoords=ax.transAxes,arrowprops=dict(arrowstyle="->",lw=2,color=SLATE))
    ax.set_title("N-BEATS experiment progression",fontsize=19,weight="bold"); save(fig,"07_experiment_progression.png")


def registry_decision():
    fig,ax=plt.subplots(figsize=(14,5)); ax.axis("off"); ax.text(.18,.55,"Baseline reference\nWMAE = 2,157.98",ha="center",va="center",transform=ax.transAxes,bbox=dict(boxstyle="round,pad=1",fc="#F0FDF4",ec=GREEN,lw=3)); ax.text(.50,.55,"Best Optuna candidate\nWMAE = 2,191.41\nDid not beat baseline",ha="center",va="center",transform=ax.transAxes,bbox=dict(boxstyle="round,pad=1",fc="#FEF2F2",ec=RED,lw=3)); ax.text(.82,.55,"Registry fallback\nTrain baseline configuration\nAlias: candidate",ha="center",va="center",transform=ax.transAxes,bbox=dict(boxstyle="round,pad=1",fc="#F5F3FF",ec=PURPLE,lw=3));
    for a,b in [(.28,.39),(.61,.71)]: ax.annotate("",xy=(b,.55),xytext=(a,.55),xycoords=ax.transAxes,arrowprops=dict(arrowstyle="->",lw=3,color=SLATE)); ax.set_title("Model Registry selection logic",fontsize=19,weight="bold"); save(fig,"08_registry_fallback_decision.png")


def validation_kaggle_gap():
    fig,ax=plt.subplots(figsize=(9,6)); vals=[2157.9829,4700]; bars=ax.bar(["Best validation","Kaggle test"],vals,color=[GREEN,RED]); ax.set(title="N-BEATS validation-to-Kaggle generalization gap",ylabel="WMAE")
    for b,v in zip(bars,vals): ax.text(b.get_x()+b.get_width()/2,v+80,f"{v:,.1f}",ha="center",weight="bold")
    ax.text(.5,.85,"+2,542 WMAE",transform=ax.transAxes,ha="center",color=RED,fontsize=16,weight="bold"); ax.grid(axis="y",alpha=.2); save(fig,"09_validation_vs_kaggle_gap.png")


def workflow():
    fig,ax=plt.subplots(figsize=(16,4)); ax.axis("off"); steps=["Raw Store–Dept\nsales","Weekly panel +\nfill gaps","log1p + per-series\nnormalization","52 → 32\nwindows","N-BEATS residual\nblocks","WMAE + Optuna\nearly stopping","Registry fallback +\nrecursive test forecast"]; xs=np.linspace(.06,.94,len(steps))
    for i,(x,t) in enumerate(zip(xs,steps)):
        ax.text(x,.5,t,ha="center",va="center",transform=ax.transAxes,bbox=dict(boxstyle="round,pad=.65",fc="#EFF6FF",ec=BLUE,lw=2));
        if i<len(steps)-1: ax.annotate("",xy=(xs[i+1]-.055,.5),xytext=(x+.055,.5),xycoords=ax.transAxes,arrowprops=dict(arrowstyle="->",color=SLATE,lw=2))
    ax.set_title("Final N-BEATS training and inference workflow",fontsize=18,weight="bold"); save(fig,"10_training_inference_workflow.png")


def extract_diagnostics():
    nb=json.loads((ROOT/"model_experiment_N_BEATS.ipynb").read_text())
    names=[
        "18_recorded_final_diagnostics.png",
        "19_recorded_aggregate_actual_vs_forecast.png",
        "20_recorded_representative_series_forecasts.png",
        "25_recorded_real_backcast_forecast_decomposition.png",
        "21_recorded_error_by_forecast_horizon.png",
        "22_recorded_holiday_vs_nonholiday_error.png",
    ]
    image_index=0
    for output in nb["cells"][16].get("outputs",[]):
        data=output.get("data",{})
        if "image/png" in data:
            payload=data["image/png"]; payload="".join(payload) if isinstance(payload,list) else payload
            if image_index < len(names):
                (OUT/names[image_index]).write_bytes(base64.b64decode(payload))
            image_index += 1


def final_fast_optuna_summary():
    labels=["Recorded baseline","Old best Optuna","Fast Optuna final"]
    vals=[2157.9829,2191.4117,2122.1522]
    fig,ax=plt.subplots(figsize=(10,6)); bars=ax.bar(labels,vals,color=[BLUE,ORANGE,GREEN]); ax.set_ylim(2080,2230); ax.set(title="Final fast Optuna improvement",ylabel="Validation WMAE — lower is better")
    for b,v in zip(bars,vals): ax.text(b.get_x()+b.get_width()/2,v+5,f"{v:,.2f}",ha="center",weight="bold")
    ax.grid(axis="y",alpha=.2); save(fig,"23_final_fast_optuna_improvement.png")


def final_fast_params():
    rows=[["Trial","3"],["Best epoch","3"],["Validation WMAE","2,122.1522"],["Optimizer","AdamW"],["Loss","regular L1"],["Batch size","128"],["Learning rate","0.00097464"],["Hidden units","256"],["Blocks","2"],["Layers per block","3"],["Dropout","0.06220"],["Weight decay","0.00004408"],["Registry alias","champion"]]
    fig,ax=plt.subplots(figsize=(10,9)); ax.axis("off"); t=ax.table(cellText=rows,colLabels=["Item","Final selected value"],cellLoc="left",loc="center"); t.auto_set_font_size(False); t.set_fontsize(12); t.scale(1,1.55)
    for (r,c),cell in t.get_celld().items(): cell.set_edgecolor("#CBD5E1"); cell.set_facecolor(GREEN if r==0 else ("#F8FAFC" if r%2==0 else "white")); cell.set_text_props(color="white" if r==0 else "black",weight="bold" if r==0 else "normal")
    ax.set_title("Final selected N-BEATS configuration",fontsize=20,weight="bold"); save(fig,"24_final_selected_hyperparameters.png")


def architecture_comparison():
    rows=[["Blocks","4","4"],["Layers per block","4","4"],["Hidden units","256","128"],["Dropout","0.10","0.1416"],["Batch size","128","64"],["Learning rate","0.001","0.0001307"],["Optimizer","AdamW","Adam"],["Loss","Regular L1","Regular L1"],["Best epoch","2","11"],["Validation WMAE","2,157.98","2,191.41"]]
    fig,ax=plt.subplots(figsize=(12,8)); ax.axis("off"); t=ax.table(cellText=rows,colLabels=["Item","Baseline","Best recorded Optuna"],cellLoc="center",loc="center"); t.auto_set_font_size(False); t.set_fontsize(12); t.scale(1,1.7)
    for (r,c),cell in t.get_celld().items():
        cell.set_edgecolor("#CBD5E1"); cell.set_facecolor(PURPLE if r==0 else ("#F8FAFC" if r%2==0 else "white"));
        if r==0: cell.set_text_props(color="white",weight="bold")
    ax.set_title("Baseline versus best recorded Optuna architecture",fontsize=20,weight="bold"); save(fig,"12_baseline_vs_optuna_architecture.png")


def capacity_vs_score():
    labels=["Baseline\n256 hidden","Best Optuna\n128 hidden"]; params=[930128,268624]; scores=[2157.9829,2191.4117]
    fig,axes=plt.subplots(1,2,figsize=(13,6)); bars=axes[0].bar(labels,params,color=[BLUE,PURPLE]); axes[0].set_title("Trainable parameter count"); axes[0].set_ylabel("Parameters")
    for b,v in zip(bars,params): axes[0].text(b.get_x()+b.get_width()/2,v+18000,f"{v:,}",ha="center",weight="bold")
    bars=axes[1].bar(labels,scores,color=[GREEN,ORANGE]); axes[1].set_ylim(2100,2230); axes[1].set_title("Validation WMAE — lower is better")
    for b,v in zip(bars,scores): axes[1].text(b.get_x()+b.get_width()/2,v+5,f"{v:,.2f}",ha="center",weight="bold")
    for ax in axes: ax.grid(axis="y",alpha=.2)
    fig.suptitle("Smaller capacity reduced parameters but did not beat baseline",fontsize=19,weight="bold"); save(fig,"13_capacity_vs_validation_score.png")


def regularization_comparison():
    labels=["Baseline","Best Optuna","Holiday-loss trial"]; dropout=[.10,.1416,.0341]; wd=[1e-4,1.1528e-6,1.5673e-6]; score=[2157.98,2191.41,2199.46]
    fig,axes=plt.subplots(1,3,figsize=(16,6)); axes[0].bar(labels,dropout,color=[GREEN,PURPLE,ORANGE]); axes[0].set_title("Dropout")
    axes[1].bar(labels,wd,color=[GREEN,PURPLE,ORANGE]); axes[1].set_yscale("log"); axes[1].set_title("Weight decay — log scale")
    axes[2].bar(labels,score,color=[GREEN,PURPLE,ORANGE]); axes[2].set_ylim(2120,2230); axes[2].set_title("Validation WMAE")
    for ax in axes: ax.tick_params(axis="x",rotation=15); ax.grid(axis="y",alpha=.2)
    fig.suptitle("Regularization choices and recorded outcome",fontsize=19,weight="bold"); save(fig,"14_regularization_comparison.png")


def search_space_map():
    groups=[("Architecture","hidden: 128 / 256\n4 blocks × 4 layers"),("Optimization","Adam / SGD\nLR 1e-4–1e-2\nbatch 64 / 128"),("Regularization","dropout 0–0.20\nweight decay 1e-6–1e-3"),("Objective","regular L1 or\nholiday-weighted L1"),("Budget","8 requested\n7 completed\n1 pruned")]
    fig,ax=plt.subplots(figsize=(16,5)); ax.axis("off"); xs=np.linspace(.09,.91,len(groups))
    for i,(x,(title,body)) in enumerate(zip(xs,groups)):
        ax.text(x,.5,title+"\n\n"+body,ha="center",va="center",transform=ax.transAxes,bbox=dict(boxstyle="round,pad=.7",fc="white",ec=PURPLE,lw=2.5))
        if i<len(groups)-1: ax.annotate("",xy=(xs[i+1]-.08,.5),xytext=(x+.08,.5),xycoords=ax.transAxes,arrowprops=dict(arrowstyle="->",color=SLATE,lw=2))
    ax.set_title("Recorded Optuna search space and budget",fontsize=20,weight="bold"); save(fig,"15_optuna_search_space_map.png")


def holiday_loss_logic():
    fig,ax=plt.subplots(figsize=(14,5)); ax.axis("off")
    ax.text(.17,.55,"Normal-week error\nweight = 1",ha="center",va="center",transform=ax.transAxes,bbox=dict(boxstyle="round,pad=.8",fc="#EFF6FF",ec=BLUE,lw=3))
    ax.text(.50,.55,"Holiday-week error\nweight = 5",ha="center",va="center",transform=ax.transAxes,bbox=dict(boxstyle="round,pad=.8",fc="#FFF7ED",ec=ORANGE,lw=3))
    ax.text(.83,.55,"Experiment 3\nWMAE = 2,185.14\nBaseline = 2,157.98",ha="center",va="center",transform=ax.transAxes,bbox=dict(boxstyle="round,pad=.8",fc="#FEF2F2",ec=RED,lw=3))
    for a,b in [(.27,.39),(.61,.72)]: ax.annotate("",xy=(b,.55),xytext=(a,.55),xycoords=ax.transAxes,arrowprops=dict(arrowstyle="->",lw=2.5,color=SLATE))
    ax.text(.5,.14,"Loss emphasized holidays, but the model did not receive future holiday identity as an input feature",ha="center",transform=ax.transAxes,weight="bold",color=RED)
    ax.set_title("Holiday-weighted loss experiment",fontsize=20,weight="bold"); save(fig,"16_holiday_weighted_loss_logic.png")


def diagnostics_boundary():
    rows=[["Actual vs predicted scatter","Recorded notebook PNG","Available"],["Weekly validation MAE","Recorded notebook PNG","Available"],["Per-horizon N-BEATS WMAE","Raw validation predictions","Not persisted"],["Holiday vs non-holiday error","Raw validation predictions","Not persisted"],["Example Store–Dept forecasts","Raw validation predictions","Not persisted"],["Numeric block contributions","Per-block forward outputs","Not persisted"]]
    fig,ax=plt.subplots(figsize=(13,7)); ax.axis("off"); t=ax.table(cellText=rows,colLabels=["Diagnostic","Required evidence","Status"],cellLoc="left",loc="center"); t.auto_set_font_size(False); t.set_fontsize(11); t.scale(1,1.7)
    for (r,c),cell in t.get_celld().items():
        cell.set_edgecolor("#CBD5E1"); cell.set_facecolor(SLATE if r==0 else ("#F8FAFC" if r%2==0 else "white"));
        if r==0: cell.set_text_props(color="white",weight="bold")
        if r>0 and c==2: cell.set_text_props(color=GREEN if "Available"==cell.get_text().get_text() else RED,weight="bold")
    ax.set_title("Which N-BEATS diagnostics can be reconstructed honestly?",fontsize=20,weight="bold"); save(fig,"17_diagnostics_evidence_boundary.png")


if __name__=="__main__":
    experiment_evolution(); kaggle_comparison(); optuna_leaderboard(); best_trial_curve(); best_params_table(); sensitivity(); experiment_flow(); registry_decision(); validation_kaggle_gap(); workflow(); extract_diagnostics(); architecture_comparison(); capacity_vs_score(); regularization_comparison(); search_space_map(); holiday_loss_logic(); diagnostics_boundary(); final_fast_optuna_summary(); final_fast_params(); print(f"Generated model-experiment figures in {OUT}")
