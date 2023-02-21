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

metrics_to_include_in_graphics = [
    {
        "path": ["computation_duration"],
        "tex": "$\Delta t_{run}$",
    },
    {
        "path": ["consistency", "a_normalized_error"],
        "tex": "$\Delta_a$",
    },
    {
        "path": ["consistency", "b_normalized_error"],
        "tex": "$\Delta_b$",
    },
    {
        "path": ["consistency", "sigma_y_normalized_error"],
        "tex": "$\Delta_\sigma$",
    },
    {
        "path": ["consistency", "normalized_model_error_mean"],
        "tex": "$m_{mean}$",
    },
    {
        "path": ["consistency", "normalized_model_error_std"],
        "tex": "$m_{std}$",
    },
    {
        "path": ["convergence", "a", "normalized_model_error_std"],
        "tex": "$m_{std}$",
    },
]

metrics_to_include_in_tables = metrics_to_include_in_graphics + [
    {"path": ["summary", "a"], "tex": r"$\hat{a}$"},
    {"path": ["summary", "b"], "tex": r"$\hat{b}$"},
    {"path": ["summary", "ua"], "tex": r"$u_{\hat{a}}$"},
    {"path": ["summary", "ub"], "tex": r"$u_{\hat{b}}$"},
    {"path": ["convergence", "a", "band_shrinks"], "tex": r"$s_a(4s) > s_a(16s)$"},
    {"path": ["convergence", "b", "band_shrinks"], "tex": r"$s_b(4s) > s_b(16s)$"},
    {
        "path": ["convergence", "sigma_y", "band_shrinks"],
        "tex": r"$s_\sigma(4s) > s_\sigma(16s)$",
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

caption_info = [
    {"path": ["summary", "a_true"], "tex": "$a$"},
    {"path": ["summary", "b_true"], "tex": "$b$"},
]


# where to search
working_directory = os.path.join(os.getcwd())


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
    df_scenario = pd.DataFrame(
        index=methods_to_include,
        columns=[item["tex"] for item in metrics_to_include_in_tables],
    )

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
            fill_value = "--"
            print("    ", method, "not found")

        for metric in metrics_to_include_in_tables:
            metric_tex = metric["tex"]
            if fill_value == None:
                metric_value = address_dict(metrics_log[method], metric["path"])
            else:
                metric_value = fill_value
            df_scenario.loc[method, metric_tex] = metric_value

    caption = f"Calculated metrics for scenario \\texttt{{{scenario}}}. True values are $a=2$, $b=1$. (\\texttt{{--}}: method not run / not successful, \\texttt{{-}}: not applicable)".replace('_', '\\_')
    label = f"tab:evaluation_metrics_{scenario}"

    # make index prettier
    df_scenario.index = [f"\\texttt{{{val}}}".replace('_', '\\_') for val in df_scenario.index]
    
    # generate LaTeX string from table
    table = df_scenario.style.format(precision=2).to_latex(
        caption=caption,
        label=label,
        hrules=True,
        environment="sidewaystable",
        position="p",
    )
    ref = f"\\cref{{{label}}}"

    tables.append(f"\\section{{Scenario \\texttt{{{scenario}}}}}\n\n".replace('_', '\\_') + table)
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
