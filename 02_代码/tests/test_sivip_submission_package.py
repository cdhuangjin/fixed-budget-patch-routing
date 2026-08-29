from pathlib import Path
from zipfile import ZipFile


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SUBMISSION = PROJECT_ROOT / "07_paper" / "submission" / "sivip_submission"
ARCHIVE = SUBMISSION / "SIViP_Experiment027_Manuscript_LaTeX_20260829.zip"


def test_sivip_submission_package_identifies_only_the_selected_journal():
    required_files = {
        "main.tex",
        "supplementary.tex",
        "cover_letter.tex",
        "README_submission.md",
        "reference_verification_report.md",
    }
    assert SUBMISSION.is_dir()

    texts = []
    for name in required_files:
        path = SUBMISSION / name
        assert path.is_file(), f"missing submission source: {path}"
        texts.append(path.read_text(encoding="utf-8"))

    package_text = "\n".join(texts)
    assert "Signal, Image and Video Processing" in package_text
    assert "Pattern Analysis and Applications" not in package_text
    assert "PAAA" not in package_text
    assert "journal/10044" not in package_text


def test_sivip_readme_identifies_the_submission_release_branch():
    readme = (SUBMISSION / "README_submission.md").read_text(encoding="utf-8")
    assert "5f5e027" not in readme
    assert "sivip_rebuild" in readme
    assert "immutable revision" in readme.lower()


def test_sivip_manuscript_describes_the_public_code_release():
    manuscript = (SUBMISSION / "main.tex").read_text(encoding="utf-8")
    assert "will be made available" not in manuscript
    assert "evaluation scripts, configuration files and route definitions are available" in manuscript


def test_sivip_package_does_not_describe_the_initial_release_as_its_revision():
    author_actions = (SUBMISSION / "AUTHOR_ACTIONS_027.md").read_text(encoding="utf-8")
    assert "5f5e027" not in author_actions


def test_sivip_archive_contains_editable_sources_and_compiled_pdfs():
    assert ARCHIVE.is_file()
    with ZipFile(ARCHIVE) as archive:
        names = set(archive.namelist())

    assert {
        "main.tex",
        "supplementary.tex",
        "cover_letter.tex",
        "README_submission.md",
        "build/main.pdf",
        "build/supplementary.pdf",
        "build/cover_letter.pdf",
    } <= names
    assert not any("PAAA" in name for name in names)
    with ZipFile(ARCHIVE) as archive:
        for name in ("main.tex", "README_submission.md", "AUTHOR_ACTIONS_027.md"):
            assert archive.read(name).decode("utf-8") == (SUBMISSION / name).read_text(
                encoding="utf-8"
            )


def test_project_readme_points_to_the_sivip_submission_package():
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    assert "07_paper/submission/sivip_submission/" in readme


def test_sivip_sources_use_the_recommended_two_column_basic_springer_style():
    manuscript = (SUBMISSION / "main.tex").read_text(encoding="utf-8")
    supplement = (SUBMISSION / "supplementary.tex").read_text(encoding="utf-8")

    assert r"\documentclass[iicol,pdflatex,sn-basic,Numbered]{sn-jnl}" in manuscript
    assert r"\documentclass[pdflatex,sn-basic,Numbered]{sn-jnl}" in supplement
    assert "sn-vancouver-num" not in manuscript + supplement
    assert (SUBMISSION / "sn-basic.bst").is_file()


def test_sivip_main_uses_wide_floats_for_full_width_figures():
    manuscript = (SUBMISSION / "main.tex").read_text(encoding="utf-8")

    assert manuscript.count(r"\begin{figure*}") == 3
    assert manuscript.count(r"\end{figure*}") == 3
    assert r"\begin{figure}" not in manuscript
    assert r"\section*{Statements and Declarations}" in manuscript


def test_sivip_front_matter_and_supplement_are_self_explanatory():
    manuscript = (SUBMISSION / "main.tex").read_text(encoding="utf-8")
    supplement = (SUBMISSION / "supplementary.tex").read_text(encoding="utf-8")

    assert "area under the receiver operating characteristic curve (AUROC)" in manuscript
    assert "95th-percentile (P95)" in manuscript
    assert "Supplementary Information for" in supplement
    assert "Submitted to Signal, Image and Video Processing" not in supplement
    assert r"\email{614938561@qq.com\\ORCID" not in manuscript + supplement


def test_sivip_conclusion_is_the_last_numbered_main_section():
    manuscript = (SUBMISSION / "main.tex").read_text(encoding="utf-8")

    assert r"\section{Methods}" not in manuscript
    assert manuscript.index(r"\section{Experimental setup}") < manuscript.index(
        r"\subsection{Implementation details}"
    )
    assert manuscript.index(r"\subsection{Ethics and reproducibility}") < manuscript.index(
        r"\section{Results}"
    )
    assert manuscript.index(r"\section{Discussion}") < manuscript.index(
        r"\section{Conclusion}"
    ) < manuscript.index(r"\backmatter")


def test_sivip_public_repository_language_matches_the_published_branch():
    manuscript = (SUBMISSION / "main.tex").read_text(encoding="utf-8")
    supplement = (SUBMISSION / "supplementary.tex").read_text(encoding="utf-8")

    assert "submission-matched" not in manuscript
    assert "are available in the project repository" in manuscript
    assert "contains the evaluation protocol" in manuscript
    assert r"\caption{Primary paired effects at the 25\% fallback budget}" in supplement
    assert "canonical 25\\% strong-routing result" not in supplement


def test_sivip_figures_do_not_embed_panel_titles_and_add_non_colour_cues():
    strong_script = (PROJECT_ROOT / "02_代码" / "make_strong_routing_figures.py").read_text(
        encoding="utf-8"
    )
    latency_script = (PROJECT_ROOT / "02_代码" / "make_canonical_v2_figures.py").read_text(
        encoding="utf-8"
    )
    budget_script = (
        PROJECT_ROOT / "07_paper" / "figures" / "figures" / "plot_budget_sensitivity.py"
    ).read_text(encoding="utf-8")

    assert "ax.set_title" not in strong_script + latency_script + budget_script
    assert "HATCHES" in strong_script
    assert "hatch=" in strong_script + latency_script
    assert "linestyle=" in budget_script


def test_sivip_archive_contains_basic_style_and_named_supplement():
    assert ARCHIVE.is_file()
    with ZipFile(ARCHIVE) as archive:
        names = set(archive.namelist())

    assert "sn-basic.bst" in names
    assert "build/ESM_1.pdf" in names
    assert "data/canonical_v3/raw_results.csv" in names
    assert "data/canonical_v3/main_results.csv" in names
    assert "data/canonical_v3_budget/budget_sensitivity.csv" in names
    assert "data/upstream/strong_routing_canonical_v1.json" in names
    assert "data/upstream/strong_routing_budget_canonical_v1.json" in names
