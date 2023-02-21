import glob
import json
import os
import numpy as np
import pandas as pd


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

methods_to_include = [
    "gibbs_base",
    "gibbs_minimal",
    "gibbs_no_EIV",
    "gibbs_known_sigma_y",
    "joint_posterior",
    "joint_posterior_agrid",
    "stankovic_base",
    "stankovic_enhanced_unc",
    "stankovic_base_unc",
]

metrics_to_include_summary = [
    {"path": ["summary", "a_true"], "tex": r"$a$"},
    {"path": ["summary", "b_true"], "tex": r"$b$"},
    {"path": ["summary", "sigma_y_true"], "tex": r"$\sigma_y$"},
    {"path": ["summary", "a"], "tex": r"$\hat{a}$"},
    {"path": ["summary", "b"], "tex": r"$\hat{b}$"},
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
    {"path": ["convergence", "a", "band_shrinks"], "tex": r"\rotatebox{90}{$s_a(4s) > s_a(16s)$}"},
    {"path": ["convergence", "b", "band_shrinks"], "tex": r"\rotatebox{90}{$s_b(4s) > s_b(16s)$}"},
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
    metrics_to_include_summary[0:1]
    + metrics_to_include_consistency
    + metrics_to_include_convergence[:4]
)

# relevant columns
columns_all = [item["tex"] for item in metrics_to_include_in_tables]
columns_summary = [item["tex"] for item in metrics_to_include_summary]
columns_consistency = [item["tex"] for item in metrics_to_include_consistency]
columns_convergence = [item["tex"] for item in metrics_to_include_convergence]
columns_graphics = [item["tex"] for item in metrics_to_include_in_graphics]

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

def address_dict(d, dict_path):
    tmp = d
    try:
        for key in dict_path:
            tmp = tmp[key]
    except KeyError:
        tmp = "-"
    return tmp


tables = []
refs = []
for scenario in scenarios:
    print(scenario)
    df_scenario = pd.DataFrame(index=methods_to_include, columns=columns_all)

    metrics_path = os.path.join(
        working_directory, "experiments", scenario, "metrics.json"
    )
    if not os.path.exists(metrics_path):
        print("")
        continue
    metrics_log = json.load(open(metrics_path, "r"))

    for method in methods_to_include:
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
    caption_summary = f"Result summary of scenario \\texttt{{{scenario_escaped}}}. (\\texttt{{/}}: method not (successfully) run, \\texttt{{-}}: not applicable)"
    caption_consistency = f"Runtime and consistency metrics of scenario \\texttt{{{scenario_escaped}}}. (\\texttt{{/}}: method not (successfully) run, \\texttt{{-}}: not applicable)"
    caption_convergence = f"Convergence metrics of scenario \\texttt{{{scenario_escaped}}}. (\\texttt{{/}}: method not (successfully) run, \\texttt{{-}}: not applicable)"
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

    ref = f"\\cref{{{label_summary},{label_consistency},{label_convergence}}}"

    ## merge tables into section
    tables.append(
        f"\\newpage\n\n\\section{{Scenario \\texttt{{{scenario}}}}}\n\n".replace("_", "\\_")
        + table_summary
        + table_consistency
        + table_convergence
    )
    ## tables.append(table)
    refs.append(ref)

table_strings = "\n\n\n".join(tables)
ref_strings = "\n".join(refs)

# write to file
f = open("evaluation_metrics_tables.tex", "w")
f.write("\\chapter{Evaluation Results per Scenario}\n")
f.write(ref_strings)
f.write("\n\n\n\n")
f.write(table_strings)
f.close()
