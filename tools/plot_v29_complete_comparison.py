from pathlib import Path
import argparse,hashlib,json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

def main(directory):
    raw=(directory/"complete_comparison.json").read_bytes()
    report=json.loads(raw)
    proof=json.loads((directory/"local_terminal_scalar_recomputation.json").read_bytes())
    assert proof["status"]=="PASS_LOCAL_COMPLETE_V29_SCALARS_AND_GEOMETRY_RECEIPTS"
    assert report["source_summary_sha256"]==proof["source_summary_sha256"]
    assert report["scientific_checks"]==proof["scientific_checks"]
    assert len(report["all_identity_rows"])==21 and len(report["all_query_rows"])==571
    assert len(report["fold_metric_rows"])==30
    names=["baseline_only","fused","cnn","transformer","mamba"]
    labels=["Signal","Fused","CNN","Transformer","Mamba"]
    a=np.array([report["aggregate"]["control"][x]["mAP"] for x in names])
    b=np.array([report["aggregate"]["bounded_joint"][x]["mAP"] for x in names])
    rows=report["all_identity_rows"]
    gains=np.array([x["gains_percentage_points"]["fused"]["mAP"] for x in rows])
    assert sum(x["queries"] for x in rows)==571
    assert abs(sum(x["queries"]*g for x,g in zip(rows,gains))/571-report["matched_gains_mAP"]["fused"])<1e-10
    fig,(left,right)=plt.subplots(1,2,figsize=(13,8),gridspec_kw={"width_ratios":[1,1.45]})
    for i,(old,new) in enumerate(zip(a,b)):
        left.plot([old,new],[i,i],color="#9198a1",lw=2,zorder=1)
    left.scatter(a,np.arange(5),s=100,facecolors="white",edgecolors="#2463a6",linewidth=2,zorder=3)
    left.scatter(b,np.arange(5),s=40,c="#df7428",zorder=4)
    for i,(old,new) in enumerate(zip(a,b)):
        left.text(min(a.min(),b.min())-.45,i+.29,f"{old:.4f} -> {new:.4f}  ({new-old:+.4f} pp)",fontsize=9,color="#333333")
    left.set_yticks(np.arange(5),labels)
    left.set_ylim(4.7,-.8)
    left.set_xlim(min(a.min(),b.min())-.6,max(a.max(),b.max())+.6)
    left.set_xlabel("mAP (%)")
    left.set_title("All five outputs: matched control and candidate",loc="left",fontsize=11,fontweight="bold")
    left.grid(axis="x",alpha=.2)
    legend=[Line2D([0],[0],marker="o",linestyle="None",markerfacecolor="white",markeredgecolor="#2463a6",markeredgewidth=2,label="Control"),
            Line2D([0],[0],marker="o",linestyle="None",color="#df7428",label="Bounded joint")]
    left.legend(handles=legend,loc="upper left",bbox_to_anchor=(0,.985),ncol=2,frameon=False)
    colors=["#2d8275" if x>=0 else "#b8514a" for x in gains]
    right.barh(np.arange(21),gains,color=colors,height=.72)
    right.set_yticks(np.arange(21),[f"f{x['fold']}  {x['original_identity']}  (n={x['queries']})" for x in rows],fontsize=9)
    right.invert_yaxis()
    right.axvline(0,color="#555555",lw=.9)
    extent=max(float(np.max(np.abs(gains))),.1)
    right.set_xlim(-extent*1.35,extent*1.35)
    for i,value in enumerate(gains):
        right.text(value+(.025*extent if value>=0 else -.025*extent),i,f"{value:+.3f}",
                   va="center",ha="left" if value>=0 else "right",fontsize=8)
    right.set_xlabel("Fused mAP change (percentage points)")
    right.set_title("All 21 identities: no omitted declines",loc="left",fontsize=11,fontweight="bold")
    right.grid(axis="x",alpha=.18)
    for ax in (left,right):
        for side in ("top","right"):
            ax.spines[side].set_visible(False)
        ax.set_axisbelow(True)
    passed=sum(report["scientific_checks"].values())
    fig.suptitle(f"V29 fixed endpoint paired Q1 | {report['scientific_status']} ({passed}/5 gates)",fontsize=15,fontweight="bold",x=.05,ha="left")
    fig.text(.05,.925,"Internal identity-isolated development comparison; single seed 42; not official test.",fontsize=10,color="#444444")
    fig.text(.05,.026,f"All 3 folds, 6 endpoints, 571 eligible queries per arm, and 3,126 gallery records.\n"
             f"Fused gain {report['matched_gains_mAP']['fused']:+.4f} pp; identity-cluster bootstrap lower bound "
             f"{report['bootstrap']['lower_bound_95_mAP']:+.4f} pp. Reused OOF does not remove selection bias.",
             fontsize=9,color="#444444")
    fig.subplots_adjust(left=.09,right=.98,top=.855,bottom=.14,wspace=.60)
    artifacts=[]
    for extension in ("png","svg"):
        path=directory/("v29_complete_paired_comparison."+extension)
        assert not path.exists()
        fig.savefig(path,dpi=180,facecolor="white")
        artifacts.append({"path":str(path),"bytes":path.stat().st_size,"sha256":hashlib.sha256(path.read_bytes()).hexdigest()})
    plt.close(fig)
    receipt={"status":"COMPLETE_DATA_FIGURE_GENERATED_NOT_VISUALLY_REVIEWED","source_report_sha256":hashlib.sha256(raw).hexdigest(),"source_summary_sha256":report["source_summary_sha256"],"script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"all_identity_count":21,"all_output_count":5,"artifacts":artifacts,"new_training_or_model_reads":0}
    path=directory/"figure_receipt.json";assert not path.exists()
    path.write_bytes((json.dumps(receipt,indent=2)+"\n").encode())
    print(json.dumps(receipt))
if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--directory",type=Path,required=True)
    main(parser.parse_args().directory)
