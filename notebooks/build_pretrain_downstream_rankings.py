"""Build the rank-based pretrain comparison notebook.

The generated notebook reads the normalized metric snapshot from
outputs/graph_jepa_results_20260727/pretrain_downstream_metrics.csv.
"""

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "pretrain_downstream_rankings.ipynb"


def md(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbf.v4.new_code_cell(text.strip())


nb = nbf.v4.new_notebook()
nb["metadata"] = {
    "kernelspec": {
        "display_name": "Python (IAD)",
        "language": "python",
        "name": "python3",
    },
    "language_info": {"name": "python", "version": "3.11"},
}

nb["cells"] = [
    md(
        r"""
# Сравнение претрейнов по downstream-задачам

Цель — сравнить не отдельные абсолютные числа, а **место каждого претрейна среди остальных**.
Так RMSE, accuracy и угловые метрики становятся сопоставимыми, не требуя произвольного
масштабирования.

Структура:

1. force estimation;
2. pose estimation;
3. object classification;
4. профиль от локальной к глобальной информации;
5. общий топ.

Основное множество — только методы с претрейном. `Random / MLP / WL+MLP / GNN`
показаны как downstream-референсы, но не влияют на места претрейнов.
"""
    ),
    md(
        r"""
## Как считаются ранги

- Для RMSE меньше — лучше; для accuracy больше — лучше.
- Ничьи получают средний ранг.
- **Force rank**: ранг общего `force RMSE`.
- **Pose rank**: сначала поровну объединяются translation (`x`, `y`) и rotation (`θ`);
  внутри каждой части RMSE и accuracy имеют одинаковый вес.
- **Object rank**: ранг accuracy.
- **Overall rank**: среднее трёх downstream-рангов, поэтому pose с шестью колонками
  не получает больший вес, чем force или object.
- **Worst-task rank**: худшее место метода по трём downstream-задачам. Это отдельная
  minimax-оценка универсальности.

Шкала `force → pose XY → pose θ → object` ниже — аналитическая интерпретация
локальности/глобальности, а не измеренная физическая величина.
"""
    ),
    code(
        r"""
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, Markdown
from scipy.stats import spearmanr

warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option("display.max_colwidth", 100)
pd.set_option("display.precision", 4)

sns.set_theme(
    style="whitegrid",
    context="notebook",
    rc={
        "figure.dpi": 120,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "bold",
    },
)

COLORS = {
    "Graph JEPA": "#7C3AED",
    "Legacy JEPA": "#EA580C",
    "DINO / spatial pretrain": "#0284C7",
    "Downstream reference": "#64748B",
}

def find_repo_root():
    candidates = [Path.cwd(), Path.cwd().parent, Path.cwd().parent.parent]
    for candidate in candidates:
        if (candidate / "outputs" / "graph_jepa_results_20260727").exists():
            return candidate
    raise FileNotFoundError("Не найден outputs/graph_jepa_results_20260727")

ROOT = find_repo_root()
DATA_PATH = ROOT / "outputs" / "graph_jepa_results_20260727" / "pretrain_downstream_metrics.csv"
data_all = pd.read_csv(DATA_PATH)
data_all["is_pretrain"] = data_all["is_pretrain"].astype(bool)

print(f"Источник: {DATA_PATH.relative_to(ROOT)}")
print(f"Строк: {len(data_all)}; претрейнов: {data_all.is_pretrain.sum()}; референсов: {(~data_all.is_pretrain).sum()}")
"""
    ),
    code(
        r"""
SHORT_NAMES = {
    "DINO without taxel-type embedding or XYZ": "DINO no taxel/XYZ",
    "Original": "DINO original",
    "Pretrain WL (2)": "DINO + WL pretrain",
    "JEPA transformer (old)": "JEPA old",
    "JEPA transformer (1, 4)": "JEPA 1c4t",
    "JEPA (2, 4) — epoch 500": "JEPA 2c4t",
    "JEPA contiguous (1, 2)": "JEPA contiguous",
    "JEPA (2, 4) — mix random/contiguous": "JEPA mix R/C",
    "JEPA random (1, 2)": "JEPA random",
    "JEPA context contiguous → target contiguous (new)": "JEPA C→C",
    "JEPA context contiguous → target random (new)": "JEPA C→R",
    "JEPA context random → target contiguous (new)": "JEPA R→C",
    "Graph JEPA Dijkstra (2c, 4t; c40-65%, t5-11%)": "G: Dijkstra 2c4t small",
    "Graph JEPA Dijkstra (1c, 4t; c40-65%, t5-11%)": "G: Dijkstra 1c4t small",
    "Graph JEPA Dijkstra (1c, 2t; c50-80%, t10-22%)": "G: Dijkstra 1c2t",
    "Graph JEPA Dijkstra (1c, 4t; c50-90%, t10-18%)": "G: Dijkstra 1c4t",
    "Graph JEPA (1c, 4t; 2 local + 2 corridor; c50-90%, t10-18%)": "G: 2L + 2 corridor",
    "Graph JEPA (1c, 4t; 2 local + 2 endcaps; c50-90%, t10-18%)": "G: 2L + 2E",
    "Graph JEPA (1c, 4t; 2 local + 2 random; c50-90%, t10-18%)": "G: 2L + 2R",
    "Graph JEPA (1c, 4t; 2 local t5-11% + 2 endcaps t10-18%; c50-90%)": "G: small 2L + 2E",
    "Graph JEPA (1c, 4t; 2 local t5-11% + 1 endcap + 1 random t10-18%; c50-90%)": "G: small 2L + E + R",
    "Graph JEPA (1c, 4t; 2 BFS local + 2 weighted endcaps; c50-90%, t10-18%)": "G: BFS L + weighted E",
    "Graph JEPA (1c, 4t; 2 BFS local + 2 hop endcaps; c50-90%, t10-18%)": "G: BFS L + hop E",
    "Graph JEPA (1c, 4t; 2 local + 1 endcap + 1 random; c50-90%, t10-18%)": "G: 2L + E + R",
    "Graph JEPA (1c, 4t; 1 local + 2 endcaps + 1 random; c50-90%, t10-18%)": "G: L + 2E + R",
}

data_all["label"] = data_all["method"].map(SHORT_NAMES).fillna(data_all["method"])
pre = data_all[data_all.is_pretrain].copy().reset_index(drop=True)
refs = data_all[~data_all.is_pretrain].copy().reset_index(drop=True)

lower_is_better = [
    "force_rmse", "force_rmse_x", "force_rmse_y", "force_rmse_z",
    "pose_rmse_x", "pose_rmse_y", "pose_rmse_theta",
]
higher_is_better = ["pose_acc_x", "pose_acc_y", "pose_acc_theta", "object_acc"]

for metric in lower_is_better:
    pre[f"rank_{metric}"] = pre[metric].rank(method="average", ascending=True)
for metric in higher_is_better:
    pre[f"rank_{metric}"] = pre[metric].rank(method="average", ascending=False)

pre["force_rank"] = pre["rank_force_rmse"]
pre["translation_mean_rank"] = pre[
    ["rank_pose_rmse_x", "rank_pose_acc_x", "rank_pose_rmse_y", "rank_pose_acc_y"]
].mean(axis=1)
pre["translation_rank"] = pre["translation_mean_rank"].rank(method="average")
pre["rotation_mean_rank"] = pre[
    ["rank_pose_rmse_theta", "rank_pose_acc_theta"]
].mean(axis=1)
pre["rotation_rank"] = pre["rotation_mean_rank"].rank(method="average")
pre["pose_mean_rank"] = pre[["translation_rank", "rotation_rank"]].mean(axis=1)
pre["pose_rank"] = pre["pose_mean_rank"].rank(method="average")
pre["object_rank"] = pre["rank_object_acc"]

pre["overall_mean_rank"] = pre[["force_rank", "pose_rank", "object_rank"]].mean(axis=1)
pre["overall_rank"] = pre["overall_mean_rank"].rank(method="average")
pre["worst_task_rank"] = pre[["force_rank", "pose_rank", "object_rank"]].max(axis=1)
pre["global_mean_rank"] = pre[["rotation_rank", "object_rank"]].mean(axis=1)

n_methods = len(pre)
pre["overall_score_100"] = 100 * (n_methods - pre["overall_mean_rank"]) / (n_methods - 1)

family_counts = (
    pre.groupby("family").size().rename("Число методов").sort_values(ascending=False).to_frame()
)
display(family_counts)
"""
    ),
    code(
        r"""
def family_colors(frame):
    return frame["family"].map(COLORS).fillna("#64748B")

def rank_dotplot(frame, rank_col, value_col, title, value_fmt, top_n=15, ax=None):
    plot = frame.nsmallest(top_n, rank_col).sort_values(rank_col, ascending=False)
    if ax is None:
        _, ax = plt.subplots(figsize=(10, max(4.5, 0.38 * len(plot))))
    colors = family_colors(plot)
    ax.hlines(plot["label"], 1, plot[rank_col], color="#CBD5E1", linewidth=2)
    ax.scatter(plot[rank_col], plot["label"], c=colors, s=75, edgecolor="white", linewidth=0.8)
    for _, row in plot.iterrows():
        ax.text(
            row[rank_col] + 0.25,
            row["label"],
            value_fmt.format(row[value_col]),
            va="center",
            fontsize=8.5,
            color="#334155",
        )
    ax.set_xlim(0.5, max(top_n + 1.8, plot[rank_col].max() + 2.2))
    ax.set_xlabel("Ранг среди претрейнов (1 = лучше)")
    ax.set_ylabel("")
    ax.set_title(title, loc="left")
    ax.grid(axis="y", visible=False)
    return ax

def styled_table(frame, formats=None, gradient_cols=None):
    styler = frame.style.hide(axis="index")
    if formats:
        styler = styler.format(formats, na_rep="—")
    if gradient_cols:
        styler = styler.background_gradient(
            subset=gradient_cols, cmap="Purples_r", vmin=1, vmax=n_methods
        )
    return styler

def pareto_mask(frame, columns):
    values = frame[columns].to_numpy(float)
    keep = np.ones(len(values), dtype=bool)
    for i, point in enumerate(values):
        dominated = np.any(
            np.all(values <= point, axis=1) & np.any(values < point, axis=1)
        )
        keep[i] = not dominated
    return keep
"""
    ),
    md(
        r"""
---

# 1. Force estimation

Force — наиболее локальная из трёх задач: результат в первую очередь зависит от качества
кодирования текущего контакта. Основной ранг строится по общему RMSE; осевые метрики
используются для диагностики.
"""
    ),
    code(
        r"""
force_cols = ["force_rank", "label", "family", "force_rmse", "force_rmse_x", "force_rmse_y", "force_rmse_z"]
force_top = pre.nsmallest(10, "force_rank")[force_cols].rename(columns={
    "force_rank": "Место",
    "label": "Метод",
    "family": "Семейство",
    "force_rmse": "RMSE",
    "force_rmse_x": "RMSE x",
    "force_rmse_y": "RMSE y",
    "force_rmse_z": "RMSE z",
})
display(styled_table(
    force_top,
    formats={"Место": "{:.0f}", "RMSE": "{:.4f}", "RMSE x": "{:.4f}", "RMSE y": "{:.4f}", "RMSE z": "{:.4f}"},
    gradient_cols=["Место"],
))

rank_dotplot(pre, "force_rank", "force_rmse", "Force: лидеры по общему RMSE", "RMSE={:.4f}", top_n=15)
plt.show()
"""
    ),
    code(
        r"""
force_heat = (
    pre.nsmallest(12, "force_rank")
    .set_index("label")[["rank_force_rmse", "rank_force_rmse_x", "rank_force_rmse_y", "rank_force_rmse_z"]]
    .rename(columns={
        "rank_force_rmse": "overall",
        "rank_force_rmse_x": "x",
        "rank_force_rmse_y": "y",
        "rank_force_rmse_z": "z",
    })
)
plt.figure(figsize=(8, 5.8))
sns.heatmap(force_heat, annot=True, fmt=".0f", cmap="Purples_r", vmin=1, vmax=n_methods, cbar_kws={"label": "Ранг"})
plt.title("Force: профиль рангов по осям", loc="left")
plt.xlabel("")
plt.ylabel("")
plt.tight_layout()
plt.show()
"""
    ),
    md(
        r"""
---

# 2. Pose estimation

Pose разделён на две способности:

- **translation** — средний ранг по RMSE и accuracy для `x/y`;
- **rotation** — средний ранг по RMSE и accuracy для `θ`.

Итоговый pose rank даёт translation и rotation одинаковый вес.
"""
    ),
    code(
        r"""
pose_cols = [
    "pose_rank", "label", "family", "translation_rank", "rotation_rank",
    "pose_rmse_x", "pose_rmse_y", "pose_rmse_theta",
    "pose_acc_x", "pose_acc_y", "pose_acc_theta",
]
pose_top = pre.nsmallest(12, "pose_rank")[pose_cols].rename(columns={
    "pose_rank": "Pose место",
    "label": "Метод",
    "family": "Семейство",
    "translation_rank": "XY место",
    "rotation_rank": "θ место",
    "pose_rmse_x": "RMSE x",
    "pose_rmse_y": "RMSE y",
    "pose_rmse_theta": "RMSE θ",
    "pose_acc_x": "Acc x",
    "pose_acc_y": "Acc y",
    "pose_acc_theta": "Acc θ",
})
display(styled_table(
    pose_top,
    formats={
        "Pose место": "{:.0f}", "XY место": "{:.0f}", "θ место": "{:.0f}",
        "RMSE x": "{:.4f}", "RMSE y": "{:.4f}", "RMSE θ": "{:.3f}",
        "Acc x": "{:.3f}", "Acc y": "{:.3f}", "Acc θ": "{:.3f}",
    },
    gradient_cols=["Pose место", "XY место", "θ место"],
))
"""
    ),
    code(
        r"""
pose_heat_cols = [
    "rank_pose_rmse_x", "rank_pose_acc_x", "rank_pose_rmse_y",
    "rank_pose_acc_y", "rank_pose_rmse_theta", "rank_pose_acc_theta",
]
pose_heat = (
    pre.nsmallest(15, "pose_rank")
    .set_index("label")[pose_heat_cols]
    .rename(columns={
        "rank_pose_rmse_x": "RMSE x",
        "rank_pose_acc_x": "Acc x",
        "rank_pose_rmse_y": "RMSE y",
        "rank_pose_acc_y": "Acc y",
        "rank_pose_rmse_theta": "RMSE θ",
        "rank_pose_acc_theta": "Acc θ",
    })
)
plt.figure(figsize=(10, 7))
sns.heatmap(pose_heat, annot=True, fmt=".0f", cmap="Purples_r", vmin=1, vmax=n_methods, cbar_kws={"label": "Ранг"})
plt.title("Pose: из чего складывается итоговый ранг", loc="left")
plt.xlabel("")
plt.ylabel("")
plt.tight_layout()
plt.show()
"""
    ),
    code(
        r"""
fig, ax = plt.subplots(figsize=(9, 7))
ax.scatter(
    pre["translation_rank"], pre["rotation_rank"],
    c=family_colors(pre), s=75, alpha=0.9, edgecolor="white",
)
to_label = pd.concat([
    pre.nsmallest(5, "pose_rank"),
    pre.nsmallest(3, "translation_rank"),
    pre.nsmallest(3, "rotation_rank"),
]).drop_duplicates("method")
for _, row in to_label.iterrows():
    ax.annotate(row["label"], (row["translation_rank"], row["rotation_rank"]),
                xytext=(5, 4), textcoords="offset points", fontsize=8)
ax.set_xlabel("Translation rank (1 = лучше)")
ax.set_ylabel("Rotation θ rank (1 = лучше)")
ax.set_title("Pose: компромисс translation ↔ rotation", loc="left")
ax.set_xlim(0, n_methods + 1)
ax.set_ylim(0, n_methods + 1)
ax.invert_yaxis()
plt.tight_layout()
plt.show()
"""
    ),
    md(
        r"""
---

# 3. Object classification

Object classification — наиболее глобальная задача: для неё особенно важно объединить
контакты по всей руке в одно представление.
"""
    ),
    code(
        r"""
object_cols = ["object_rank", "label", "family", "object_acc"]
object_top = pre.nsmallest(12, "object_rank")[object_cols].rename(columns={
    "object_rank": "Место",
    "label": "Метод",
    "family": "Семейство",
    "object_acc": "Accuracy",
})
display(styled_table(
    object_top,
    formats={"Место": "{:.0f}", "Accuracy": "{:.3%}"},
    gradient_cols=["Место"],
))

rank_dotplot(pre, "object_rank", "object_acc", "Object classification: лидеры", "acc={:.3f}", top_n=15)
plt.show()
"""
    ),
    md(
        r"""
---

# 4. От локального к глобальному

Используем четыре последовательных уровня:

1. `Force` — локальный контакт;
2. `Pose XY` — взаимное положение и распределение контакта;
3. `Pose θ` — глобальная ориентация;
4. `Object` — глобальная семантика объекта.

Чем ниже линия, тем лучше место. Плоская низкая линия означает универсальный претрейн;
резкий наклон показывает специализацию.
"""
    ),
    code(
        r"""
profile_cols = ["force_rank", "translation_rank", "rotation_rank", "object_rank"]
profile_names = ["Force", "Pose XY", "Pose θ", "Object"]
profile_methods = pre.nsmallest(10, "overall_rank").copy()

fig, ax = plt.subplots(figsize=(12, 7))
for _, row in profile_methods.iterrows():
    values = row[profile_cols].to_numpy(float)
    color = COLORS[row["family"]]
    linewidth = 3 if row["overall_rank"] <= 3 else 1.7
    alpha = 1.0 if row["overall_rank"] <= 5 else 0.65
    ax.plot(profile_names, values, marker="o", linewidth=linewidth, alpha=alpha,
            color=color, label=f'{int(row["overall_rank"])}. {row["label"]}')
ax.invert_yaxis()
ax.set_ylabel("Ранг (выше на графике = лучше)")
ax.set_title("Профиль лучших претрейнов: локальная → глобальная информация", loc="left")
ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False, fontsize=8.5)
ax.grid(axis="x", visible=False)
plt.tight_layout()
plt.show()
"""
    ),
    code(
        r"""
fig, ax = plt.subplots(figsize=(10, 7.5))
ax.scatter(
    pre["force_rank"], pre["global_mean_rank"],
    c=family_colors(pre), s=85, alpha=0.9, edgecolor="white",
)

front = pre[pareto_mask(pre, ["force_rank", "global_mean_rank"])].copy()
to_label = pd.concat([
    front,
    pre.nsmallest(7, "overall_rank"),
]).drop_duplicates("method")
for _, row in to_label.iterrows():
    ax.annotate(row["label"], (row["force_rank"], row["global_mean_rank"]),
                xytext=(5, 4), textcoords="offset points", fontsize=8)

ax.set_xlabel("Локальный ранг: Force (1 = лучше)")
ax.set_ylabel("Глобальный ранг: среднее Pose θ + Object (1 = лучше)")
ax.set_title("Локальное и глобальное качество: нижний левый угол — универсальные методы", loc="left")
ax.set_xlim(0, n_methods + 1)
ax.set_ylim(0, n_methods + 1)
plt.tight_layout()
plt.show()

display(Markdown("**Парето-фронт local/global:** " + ", ".join(front.sort_values("force_rank")["label"])))
"""
    ),
    code(
        r"""
rank_corr = pre[["force_rank", "pose_rank", "object_rank"]].corr(method="spearman")
rank_corr.index = ["Force", "Pose", "Object"]
rank_corr.columns = ["Force", "Pose", "Object"]

plt.figure(figsize=(5.5, 4.5))
sns.heatmap(rank_corr, annot=True, fmt=".2f", cmap="vlag", center=0, vmin=-1, vmax=1)
plt.title("Корреляция рангов между downstream-задачами", loc="left")
plt.tight_layout()
plt.show()
"""
    ),
    md(
        r"""
---

# 5. Общий топ

Основной итог — **средний ранг по трём downstream-задачам с равными весами**.
Рядом показан worst-task rank: насколько плохо метод может выступить на своей слабейшей задаче.
"""
    ),
    code(
        r"""
overall_cols = [
    "overall_rank", "label", "family", "force_rank", "pose_rank", "object_rank",
    "overall_mean_rank", "worst_task_rank", "overall_score_100",
]
overall = pre.sort_values(["overall_rank", "worst_task_rank"])[overall_cols].rename(columns={
    "overall_rank": "Общее место",
    "label": "Метод",
    "family": "Семейство",
    "force_rank": "Force",
    "pose_rank": "Pose",
    "object_rank": "Object",
    "overall_mean_rank": "Средний ранг",
    "worst_task_rank": "Худший ранг",
    "overall_score_100": "Rank score / 100",
})
display(styled_table(
    overall.head(15),
    formats={
        "Общее место": "{:.0f}", "Force": "{:.0f}", "Pose": "{:.0f}", "Object": "{:.0f}",
        "Средний ранг": "{:.2f}", "Худший ранг": "{:.0f}", "Rank score / 100": "{:.1f}",
    },
    gradient_cols=["Общее место", "Force", "Pose", "Object", "Худший ранг"],
))
"""
    ),
    code(
        r"""
top = pre.nsmallest(15, "overall_rank").sort_values("overall_rank", ascending=False)
fig, ax = plt.subplots(figsize=(10, 6.5))
ax.hlines(top["label"], 1, top["overall_mean_rank"], color="#CBD5E1", linewidth=2.5)
ax.scatter(
    top["overall_mean_rank"], top["label"],
    c=family_colors(top), s=85, edgecolor="white",
)
for _, row in top.iterrows():
    ax.text(row["overall_mean_rank"] + 0.2, row["label"],
            f'#{int(row["overall_rank"])} · worst #{int(row["worst_task_rank"])}',
            va="center", fontsize=8.5)
ax.set_xlabel("Средний downstream-ранг (меньше = лучше)")
ax.set_ylabel("")
ax.set_title("Общий топ претрейнов", loc="left")
ax.grid(axis="y", visible=False)
plt.tight_layout()
plt.show()
"""
    ),
    code(
        r"""
top_heat = (
    pre.nsmallest(15, "overall_rank")
    .set_index("label")[["force_rank", "pose_rank", "object_rank", "worst_task_rank"]]
    .rename(columns={
        "force_rank": "Force",
        "pose_rank": "Pose",
        "object_rank": "Object",
        "worst_task_rank": "Worst",
    })
)
plt.figure(figsize=(8, 7))
sns.heatmap(top_heat, annot=True, fmt=".0f", cmap="Purples_r", vmin=1, vmax=n_methods, cbar_kws={"label": "Ранг"})
plt.title("Профиль общего топа: где у каждого метода слабое место", loc="left")
plt.xlabel("")
plt.ylabel("")
plt.tight_layout()
plt.show()
"""
    ),
    code(
        r"""
winner = pre.nsmallest(1, "overall_rank").iloc[0]
force_winner = pre.nsmallest(1, "force_rank").iloc[0]
pose_winner = pre.nsmallest(1, "pose_rank").iloc[0]
object_winner = pre.nsmallest(1, "object_rank").iloc[0]
robust = pre.sort_values(["worst_task_rank", "overall_mean_rank"]).iloc[0]
pareto3 = pre[pareto_mask(pre, ["force_rank", "pose_rank", "object_rank"])].sort_values("overall_rank")

display(Markdown(f'''
## Автоматически рассчитанные выводы

- **Общий победитель по среднему рангу:** `{winner.label}` —
  средний ранг {winner.overall_mean_rank:.2f}, худшее место #{winner.worst_task_rank:.0f}.
- **Лучший Force:** `{force_winner.label}` — RMSE {force_winner.force_rmse:.4f}.
- **Лучший Pose:** `{pose_winner.label}` — translation #{pose_winner.translation_rank:.0f},
  rotation #{pose_winner.rotation_rank:.0f}.
- **Лучший Object:** `{object_winner.label}` — accuracy {object_winner.object_acc:.3%}.
- **Minimax-победитель:** `{robust.label}` — его худшая задача имеет ранг
  #{robust.worst_task_rank:.0f}.
- **Трёхмерный Парето-фронт Force/Pose/Object:**  
  {", ".join(pareto3.label)}

Ранговый топ отвечает на вопрос «кто универсальнее в текущем наборе экспериментов».
Он не показывает статистическую значимость небольших различий и должен обновляться после
добавления новых seed или новых претрейнов.
'''))
"""
    ),
    md(
        r"""
## Ограничения анализа

- Сейчас большинство строк представлены одним seed; ранг не равен статистической уверенности.
- Ранги зависят от состава таблицы: добавление новых методов может изменить номера мест,
  даже если абсолютные метрики старых методов не изменились.
- Overall rank специально даёт одинаковый вес downstream-задачам. Если бизнес-приоритеты
  другие, веса следует менять явно, а не скрыто через число метрик.
- Для финального выбора разумно повторить top-3 минимум на трёх seed и ранжировать уже
  средние значения вместе с неопределённостью.
"""
    ),
]

nbf.write(nb, OUTPUT)
print(OUTPUT)
