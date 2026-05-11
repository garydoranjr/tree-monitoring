# Observability methodology pipeline.
# Mirrors the Reproduce blocks in docs/observability_methodology.md (Steps 5–7).
# Steps 1–4 upstream and plot_planet_image_fraction_monthly.py are out of scope
# (marked TODO in the methodology); results/assessment.npz is treated as an
# external input produced by Step 4.

DATA = "data"
RESULTS = "results"
FIGS = "figs"
SCRIPTS = "scripts"

ASSESSMENT = f"{RESULTS}/assessment.npz"
TRAP_RAW = f"{DATA}/BCI_TRAP200_20241002_spcorrected.txt"
LEAF_COVER = f"{DATA}/df_LeafCoverTimeSeries_byTags_all_2024.csv"
RAD_EAST = f"{DATA}/radiation/bci_lutz48m_sre_elect.csv"
RAD_WEST = f"{DATA}/radiation/bci_lutz48m_srw_elect.csv"


rule all:
    input:
        f"{FIGS}/assessed_cadence.pdf",
        f"{FIGS}/avg_assessed_cadence.pdf",
        f"{RESULTS}/avg_assessed_cadence.npz",
        f"{FIGS}/solar_radiation_comparison.pdf",
        f"{FIGS}/sp_flower_counts_annual_stats.pdf",
        f"{FIGS}/sp_fruit_counts_annual_stats.pdf",
        f"{FIGS}/decid_summary.pdf",
        f"{RESULTS}/flower_summary_stats.csv",
        f"{RESULTS}/fruit_summary_stats.csv",
        f"{RESULTS}/decid_summary_stats.csv",


# ---------------------------------------------------------------------------
# Step 5: Cadence
# ---------------------------------------------------------------------------

rule plot_assessed_cadence:
    input:
        ASSESSMENT,
    output:
        f"{FIGS}/assessed_cadence.pdf",
    shell:
        "python {SCRIPTS}/plot_assessed_cadence.py {input} {output}"


rule plot_avg_assessed_cadence:
    input:
        ASSESSMENT,
    output:
        plot=f"{FIGS}/avg_assessed_cadence.pdf",
        npz=f"{RESULTS}/avg_assessed_cadence.npz",
    shell:
        "python {SCRIPTS}/plot_avg_assessed_cadence.py "
        "{input} {output.plot} {output.npz}"


# ---------------------------------------------------------------------------
# Step 6: Solar radiation validation
# ---------------------------------------------------------------------------

rule windowed_obs_counts:
    input:
        ASSESSMENT,
    output:
        f"{RESULTS}/windowed_obs_counts_05d.npz",
    shell:
        "python {SCRIPTS}/windowed_obs_counts.py {input} {output}"


rule illumination:
    input:
        counts=f"{RESULTS}/windowed_obs_counts_05d.npz",
        east=RAD_EAST,
        west=RAD_WEST,
    output:
        f"{FIGS}/solar_radiation_comparison.pdf",
    shell:
        "python {SCRIPTS}/illumination.py "
        "{input.counts} {input.east} {input.west} {output}"


# ---------------------------------------------------------------------------
# Step 7: Phenological event observability
# ---------------------------------------------------------------------------

rule fit_empirical_count_models:
    input:
        ASSESSMENT,
    output:
        f"{RESULTS}/empirical_model.npz",
    shell:
        "python {SCRIPTS}/fit_empirical_count_models.py {input} {output}"


rule get_annual_trap_data_flower:
    input:
        TRAP_RAW,
    output:
        f"{RESULTS}/sp_flower_counts_annual.npz",
    shell:
        "python {SCRIPTS}/get_annual_trap_data.py {input} {output}"


rule get_annual_trap_data_fruit:
    input:
        TRAP_RAW,
    output:
        f"{RESULTS}/sp_fruit_counts_annual.npz",
    shell:
        "python {SCRIPTS}/get_annual_trap_data.py -f {input} {output}"


rule individual_trap_analysis:
    input:
        f"{RESULTS}/sp_{{kind}}_counts_annual.npz",
    output:
        f"{RESULTS}/sp_{{kind}}_counts_annual_stats.csv",
    wildcard_constraints:
        kind="(flower|fruit)",
    shell:
        "python {SCRIPTS}/individual_trap_analysis.py {input} {output}"


rule individual_decid_analysis:
    input:
        LEAF_COVER,
    output:
        f"{RESULTS}/decid_summary.csv",
    shell:
        "python {SCRIPTS}/individual_decid_analysis.py {input} {output}"


rule event_summary_stats_flower:
    input:
        model=f"{RESULTS}/empirical_model.npz",
        events=f"{RESULTS}/sp_flower_counts_annual_stats.csv",
    output:
        f"{RESULTS}/flower_summary_stats.csv",
    shell:
        "python {SCRIPTS}/event_summary_stats.py "
        "{input.model} {input.events} {output}"


rule event_summary_stats_fruit:
    input:
        model=f"{RESULTS}/empirical_model.npz",
        events=f"{RESULTS}/sp_fruit_counts_annual_stats.csv",
    output:
        f"{RESULTS}/fruit_summary_stats.csv",
    shell:
        "python {SCRIPTS}/event_summary_stats.py "
        "{input.model} {input.events} {output}"


rule event_summary_stats_decid:
    input:
        model=f"{RESULTS}/empirical_model.npz",
        events=f"{RESULTS}/decid_summary.csv",
    output:
        f"{RESULTS}/decid_summary_stats.csv",
    shell:
        "python {SCRIPTS}/event_summary_stats.py "
        "{input.model} {input.events} {output}"


rule plot_trap_summary_flower:
    input:
        model=f"{RESULTS}/empirical_model.npz",
        events=f"{RESULTS}/sp_flower_counts_annual_stats.csv",
    output:
        f"{FIGS}/sp_flower_counts_annual_stats.pdf",
    shell:
        "python {SCRIPTS}/plot_trap_summary.py "
        "{input.model} {input.events} {output}"


rule plot_trap_summary_fruit:
    input:
        model=f"{RESULTS}/empirical_model.npz",
        events=f"{RESULTS}/sp_fruit_counts_annual_stats.csv",
    output:
        f"{FIGS}/sp_fruit_counts_annual_stats.pdf",
    shell:
        "python {SCRIPTS}/plot_trap_summary.py "
        "{input.model} {input.events} {output}"


rule plot_trap_summary_decid:
    input:
        model=f"{RESULTS}/empirical_model.npz",
        events=f"{RESULTS}/decid_summary.csv",
    output:
        f"{FIGS}/decid_summary.pdf",
    shell:
        "python {SCRIPTS}/plot_trap_summary.py "
        "{input.model} {input.events} {output}"
