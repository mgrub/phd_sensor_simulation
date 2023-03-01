import glob
import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats.mstats as scm


scenarios = [
    "01a_static_input",
    "01b_sinusoidal_input",
    "01c_jumping_input",
    "02a_static_input_noisy",
    "02b_sinusoidal_input_noisy",
    "02c_jumping_input_noisy",
    "03a_variable_blockwise",
    "03b_smaller_blocks",
    "03c_larger_blocks",
    "04a_dropouts",
    "04b_outliers",
    "05a_better_references",
    "05b_equal_references",
    "05c_worse_references",
]

methods_to_include = {
    "gibbs_base": {"marker": "*", "color": "b"},
    "gibbs_minimal": {"marker": "^", "color": "b"},
    "gibbs_no_EIV": {"marker": "v", "color": "b"},
    "gibbs_known_sigma_y": {"marker": "<", "color": "b"},
    "joint_posterior": {"marker": ">", "color": "c"},
    "joint_posterior_agrid": {"marker": "o", "color": "c"},
    "stankovic_base": {"marker": "s", "color": "k"},
    "stankovic_enhanced_unc": {"marker": "P", "color": "k"},
    "stankovic_base_unc": {"marker": "D", "color": "k"},
}

metrics_to_include_summary = [
    {"path": ["summary", "a_true"], "tex": r"$a$"},
    {"path": ["summary", "b_true"], "tex": r"$b$"},
    {"path": ["summary", "sigma_y_true"], "tex": r"$\sigma_y$"},
    {"path": ["summary", "a"], "tex": r"$\hat{a}$"},
    {"path": ["summary", "b"], "tex": r"$\hat{b}$"},
    {"path": ["summary", "sigma"], "tex": r"$\hat{\sigma}_y$"},
    {"path": ["summary", "ua"], "tex": r"$u_{\hat{a}}$"},
    {"path": ["summary", "ub"], "tex": r"$u_{\hat{b}}$"},
]

metrics_to_include_consistency = [
    {"path": ["computation_duration"], "tex": "$\Delta t_{run}$"},
    {"path": ["consistency", "a_normalized_error"], "tex": "$\Delta_a$"},
    {"path": ["consistency", "b_normalized_error"], "tex": "$\Delta_b$"},
    {"path": ["consistency", "sigma_y_normalized_error"], "tex": "$\Delta_\sigma$"},
    {"path": ["consistency", "normalized_model_error_mean"], "tex": "$m_{mean}$"},
    {"path": ["consistency", "normalized_model_error_std"], "tex": "$m_{std}$"},
]

metrics_to_include_convergence = [
    {
        "path": ["convergence", "a", "band_shrinks"],
        "tex": r"\rotatebox{90}{$s_a(4s) > s_a(16s)$}",
    },
    {
        "path": ["convergence", "b", "band_shrinks"],
        "tex": r"\rotatebox{90}{$s_b(4s) > s_b(16s)$}",
    },
    {
        "path": ["convergence", "sigma_y", "band_shrinks"],
        "tex": r"\rotatebox{90}{$s_\sigma(4s) > s_\sigma(16s)$}",
    },
    {"path": ["convergence", "a", "unc_first_below_0.1"], "tex": r"$t_a(0.1)$"},
    {"path": ["convergence", "a", "unc_stays_below_0.1"], "tex": r"$t^*_a(0.1)$"},
    {"path": ["convergence", "b", "unc_first_below_0.1"], "tex": r"$t_b(0.1)$"},
    {"path": ["convergence", "b", "unc_stays_below_0.1"], "tex": r"$t^*_b(0.1)$"},
    {
        "path": ["convergence", "sigma_y", "unc_first_below_0.1"],
        "tex": r"$t_\sigma(0.1)$",
    },
    {
        "path": ["convergence", "sigma_y", "unc_stays_below_0.1"],
        "tex": r"$t^*_\sigma(0.1)$",
    },
]

metrics_to_include_in_tables = (
    metrics_to_include_summary
    + metrics_to_include_consistency
    + metrics_to_include_convergence
)
metrics_to_include_in_graphics = (
    metrics_to_include_summary[3:] + metrics_to_include_consistency
)  # + metrics_to_include_convergence[3:]

# relevant columns
columns_all = [item["tex"] for item in metrics_to_include_in_tables]
columns_summary = [item["tex"] for item in metrics_to_include_summary]
columns_consistency = [item["tex"] for item in metrics_to_include_consistency]
columns_convergence = [item["tex"] for item in metrics_to_include_convergence]

table_template = """
\\begin{{table}}[H]
    \\tiny
    {TABULAR}
    \\caption{{{CAPTION}}}
    \\label{{{LABEL}}}
\\end{{table}}
"""

# where to search
working_directory = os.path.join(os.getcwd())


# some functions that are needed later
formatter = lambda s: s if isinstance(s, str) else f"{s:.2e}"


def address_dict(d, dict_path, key_error_fill="-"):
    tmp = d
    try:
        for key in dict_path:
            tmp = tmp[key]
    except KeyError:
        tmp = key_error_fill
    return tmp


tables = []
refs = []
for scenario in scenarios:
    print(scenario)
    df_scenario = pd.DataFrame(index=methods_to_include.keys(), columns=columns_all)

    metrics_path = os.path.join(
        working_directory, "experiments", scenario, "metrics.json"
    )
    if not os.path.exists(metrics_path):
        print("")
        continue
    metrics_log = json.load(open(metrics_path, "r"))

    for method in methods_to_include.keys():
        fill_value = None
        if method not in metrics_log.keys():
            fill_value = "/"
            print("    ", method, "not found")

        for metric in metrics_to_include_in_tables:
            metric_tex = metric["tex"]
            if fill_value == None:
                metric_value = address_dict(metrics_log[method], metric["path"])
            else:
                metric_value = fill_value
            df_scenario.loc[method, metric_tex] = metric_value

    scenario_escaped = scenario.replace("_", "\\_")
    caption_summary = f"Result summary of scenario \\texttt{{{scenario_escaped}}}."
    caption_consistency = (
        f"Runtime and consistency metrics of scenario \\texttt{{{scenario_escaped}}}."
    )
    caption_convergence = (
        f"Convergence metrics of scenario \\texttt{{{scenario_escaped}}}."
    )
    label_summary = f"tab:evaluation_summary_{scenario}"
    label_consistency = f"tab:evaluation_consistency_metrics_{scenario}"
    label_convergence = f"tab:evaluation_convergence_metrics_{scenario}"

    # generate LaTeX string from table
    # make index prettier
    df_scenario.index = [
        f"\\texttt{{{val}}}".replace("_", "\\_") for val in df_scenario.index
    ]

    ## tabulars
    tabular_summary = (
        df_scenario[columns_summary].style.format(formatter).to_latex(hrules=True)
    )
    tabular_consistency = (
        df_scenario[columns_consistency].style.format(formatter).to_latex(hrules=True)
    )
    tabular_convergence = (
        df_scenario[columns_convergence].style.format(formatter).to_latex(hrules=True)
    )

    ## tables
    table_summary = table_template.format(
        CAPTION=caption_summary,
        LABEL=label_summary,
        TABULAR=tabular_summary,
    )
    table_consistency = table_template.format(
        CAPTION=caption_consistency,
        LABEL=label_consistency,
        TABULAR=tabular_consistency,
    )
    table_convergence = table_template.format(
        CAPTION=caption_convergence,
        LABEL=label_convergence,
        TABULAR=tabular_convergence,
    )

    ref = f"{label_summary},{label_consistency},{label_convergence}"

    ## merge tables into section
    tables.append(
        f"\\newpage\n\n\\section{{Scenario \\texttt{{{scenario}}}}}\n\n".replace(
            "_", "\\_"
        )
        + table_summary
        + table_consistency
        + table_convergence
    )
    ## tables.append(table)
    refs.append(ref)

table_strings = "\n\n\n".join(tables)
ref_strings = ",".join(refs)

# write to file
f = open("evaluation_metrics_tables.tex", "w")
f.write(
    f"\\chapter{{Evaluation Results per Scenario}}\n\label{{chap:evaluation_result_tables}}\n\nIn tables \\cref{{{ref_strings}}} values coming from a method that did not run or did not run successfully are denoted by \\texttt{{/}}, while values that are not applicable are denoted by \\texttt{{-}}.\n\n"
)
f.write(table_strings)
f.close()


# figure LaTeX
figure_template = """
\\begin{{figure}}[tbph]
    \\centering
    \\includegraphics[width=\\textwidth]{{{FILEPATH}}}
    \\caption{{{CAPTION}}}
    \\label{{{LABEL}}}
\\end{{figure}}
"""

figures = []
refs = []
if not os.path.exists("evaluation_metrics"):
    os.mkdir("evaluation_metrics")

for metric in metrics_to_include_in_graphics:
    metric_name = "_".join(metric["path"])
    print(metric_name)

    df_metric = pd.DataFrame(index=scenarios, columns=methods_to_include.keys())

    for scenario in scenarios:
        metrics_path = os.path.join(
            working_directory, "experiments", scenario, "metrics.json"
        )
        if not os.path.exists(metrics_path):
            print("")
            continue
        metrics_log = json.load(open(metrics_path, "r"))

        for method in methods_to_include.keys():
            fill_value = None
            if method not in metrics_log.keys():
                fill_value = np.nan
                print("    ", method, "not found")

            # extract relevant metric
            metric_tex = metric["tex"]
            if fill_value == None:
                metric_value = address_dict(metrics_log[method], metric["path"], np.nan)
            else:
                metric_value = fill_value
            df_metric.loc[scenario, method] = metric_value

    # generate plot
    img_path = f"evaluation_metrics/{metric_name}.pdf"

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    pos = np.arange(len(df_metric.index))
    width = 1 / (len(df_metric.index) + 2)
    labels = [s.split("_")[0] for s in df_metric.index]
    use_log_scale = True
    if np.any(df_metric < 0):
        use_log_scale = False

    # define outlier limits (in lin or log scale)
    flattened_values = df_metric.to_numpy(dtype=np.float64).flatten()
    flattened_values = flattened_values[np.logical_not(np.isnan(flattened_values))]
    if use_log_scale:
        q1, q2 = scm.mquantiles(np.log(flattened_values), [0.1, 0.9])
        dq = q2 - q1
    else:
        q1, q2 = scm.mquantiles(flattened_values, [0.1, 0.9])
    dq = q2 - q1
    outlier_limit_low = q1 - 1.5 * dq
    outlier_limit_high = q2 + 1.5 * dq

    for i, (method_name, scenario_values) in enumerate(df_metric.items()):
        marker = methods_to_include[method_name]["marker"]
        color = methods_to_include[method_name]["color"]
        offset = width * np.random.randn()

        # remove outliers from plot, but still plot
        if use_log_scale:
            log_scenario_values = np.log(scenario_values.to_numpy(dtype=np.float64))
            outliers_above = log_scenario_values > outlier_limit_high
            outliers_below = log_scenario_values < outlier_limit_low
        else:
            outliers_above = scenario_values > outlier_limit_high
            outliers_below = scenario_values < outlier_limit_low

        outliers = np.logical_or(outliers_above, outliers_below)
        include = np.logical_not(outliers)

        ax.plot(
            pos[include] + offset,
            scenario_values[include],
            marker=marker,
            markersize=10,
            color=color,
            linewidth=0,
            alpha=0.5,
            label=method_name,
        )

        # marker outliers in the plot
        if np.any(outliers_above):
            # axis2data = ax.transAxes + ax.transData.inverted()

            ax.plot(
                pos[outliers_above],
                np.full(outliers_above.sum(), 1.05),
                marker="$\\uparrow !$",
                markersize=20,
                color="k",
                linewidth=0,
                #alpha=0.7,
                transform=ax.get_xaxis_transform(),
                clip_on=False,
            )
            ax.plot(
                pos[outliers_above] + offset,
                np.full(outliers_above.sum(), 1.1),
                marker=marker,
                markersize=10,
                color=color,
                linewidth=0,
                alpha=0.3,
                transform=ax.get_xaxis_transform(),
                clip_on=False,
            )

        if np.any(outliers_below):
            ax.plot(
                pos[outliers_below],
                np.full(outliers_below.sum(), -0.1),
                marker="$\\downarrow !$",
                markersize=20,
                color="k",
                linewidth=0,
                #alpha=0.7,
                transform=ax.get_xaxis_transform(),
                clip_on=False,
            )
            ax.plot(
                pos[outliers_below] + offset,
                np.full(outliers_below.sum(), -0.15),
                marker=marker,
                markersize=10,
                color=color,
                linewidth=0,
                alpha=0.3,
                transform=ax.get_xaxis_transform(),
                clip_on=False,
            )

    ax.set_xticks(ticks=pos)
    ax.set_xticklabels(labels, rotation=0)
    if use_log_scale:
        ax.set_yscale("log")
    ax.grid(visible=True, which="both", axis="both", alpha=0.4, zorder=10000)
    ax.legend(
        loc="center right",
        bbox_to_anchor=(1.4, 0.5),
        ncol=1,
        fancybox=True,
    )
    #fig.tight_layout()
    fig.savefig(img_path, bbox_inches="tight")

    # provide tex code
    metric_name = "_".join(metric["path"])
    tex_label = f"fig:{metric_name}"
    tex_caption = (
        f"Overview of {metric['tex']} for multiple methods in different scenarios."
    )
    ref = f"\cref{{{tex_label}}}"
    figure = figure_template.format(
        FILEPATH=img_path, CAPTION=tex_caption, LABEL=tex_label
    )

    figures.append(figure)
    refs.append(ref)

figure_strings = "\n".join(figures)
ref_strings = "\n".join(refs)

# write to file
f = open("evaluation_metrics_figures.tex", "w")
f.write(ref_strings)
f.write("\n\n")
f.write(figure_strings)
f.close()
