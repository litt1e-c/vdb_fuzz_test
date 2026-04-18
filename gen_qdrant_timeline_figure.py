#!/usr/bin/env python3
"""
Generate publication-style Qdrant coverage figures without Python plotting deps.

Why this script exists:
- local env currently lacks matplotlib/seaborn
- the paper still needs reproducible PDF/SVG/PNG assets
- Qdrant coverage artifacts already contain structured JSON/CSV summaries

This script therefore uses:
- Python stdlib for data extraction/validation
- PGFPlots via LaTeX to produce vector PDF
- latex + dvisvgm to produce vector SVG
- ImageMagick `convert` to rasterize SVG into PNG

Supported figure modes:
1. `timeline`:
   Dual-axis line chart from `coverage_timeline.csv`
   - left axis: line coverage (%)
   - right axis: covered lines
   Main paper use: scalar target coverage growth
2. `summary-groups`:
   Static horizontal bar chart from `summary.json`
   - x axis: line coverage (%)
   - y axis: scalar coverage groups
   - annotation: covered/total lines
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = Path.home() / "qdrant_artifacts" / "figures" / "qdrant"

SERIF_PREAMBLE = r"""
\usepackage{newtxtext,newtxmath}
\usepackage{pgfplots}
\usetikzlibrary{calc}
\pgfplotsset{compat=1.18}
\definecolor{timelinecoverage}{HTML}{E76F51}
\definecolor{timelinelines}{HTML}{264653}
\definecolor{chartgrid}{HTML}{D9E2E8}
\definecolor{charttext}{HTML}{2C2C2C}
\definecolor{chartsubtle}{HTML}{5A5A5A}
\definecolor{summarybar}{HTML}{4C78A8}
\definecolor{summaryaccent}{HTML}{7F7F7F}
"""

TIMELINE_COLOR_COVERAGE = "timelinecoverage"
TIMELINE_COLOR_LINES = "timelinelines"
SUMMARY_BAR_COLOR = "summarybar"
SUMMARY_ACCENT_COLOR = "summaryaccent"


@dataclass(frozen=True)
class FigureOutputs:
    tex_path: Path
    data_path: Path
    pdf_path: Path
    svg_path: Path
    png_path: Path


def ensure_tool(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise SystemExit(f"required tool not found: {name}")
    return path


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def run_command(cmd: list[str], cwd: Path) -> None:
    proc = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)
    if proc.returncode != 0:
        raise SystemExit(
            f"command failed ({proc.returncode}): {' '.join(cmd)}\n"
            f"stdout:\n{proc.stdout}\n\nstderr:\n{proc.stderr}"
        )


def latex_escape(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    output = value
    for src, dst in replacements.items():
        output = output.replace(src, dst)
    return output


def pretty_group_label(name: str) -> str:
    mapping = {
        "scalar_target": "Scalar Target",
        "field_index": "Field Index",
        "payload_storage": "Payload Storage",
        "payload_mutation_update": "Payload Mutation",
        "query_optimization": "Query Optimization",
        "payload_index_core": "Payload Index Core",
        "query_api_stack": "Query API Stack",
        "facet_related": "Facet-Related",
        "full_text_index": "Full-Text Index",
        "formula_rescore": "Formula Rescore",
        "state_recovery": "State Recovery",
    }
    if name in mapping:
        return mapping[name]
    return name.replace("_", " ").title()


def compile_figure(outputs: FigureOutputs, build_dir: Path) -> None:
    pdflatex = ensure_tool("pdflatex")
    latex = ensure_tool("latex")
    dvisvgm = ensure_tool("dvisvgm")
    convert = shutil.which("convert")
    pdftoppm = shutil.which("pdftoppm")

    build_dir.mkdir(parents=True, exist_ok=True)

    run_command(
        [pdflatex, "-interaction=nonstopmode", "-halt-on-error", "-output-directory", str(build_dir), str(outputs.tex_path)],
        cwd=outputs.tex_path.parent,
    )
    pdf_from_build = build_dir / f"{outputs.tex_path.stem}.pdf"
    outputs.pdf_path.write_bytes(pdf_from_build.read_bytes())

    run_command(
        [latex, "-interaction=nonstopmode", "-halt-on-error", "-output-directory", str(build_dir), str(outputs.tex_path)],
        cwd=outputs.tex_path.parent,
    )
    dvi_path = build_dir / f"{outputs.tex_path.stem}.dvi"
    run_command(
        [dvisvgm, str(dvi_path), "-n", "-o", str(outputs.svg_path)],
        cwd=outputs.tex_path.parent,
    )
    if pdftoppm:
        raster_root = build_dir / f"{outputs.tex_path.stem}_raster"
        run_command(
            [pdftoppm, "-png", "-singlefile", "-r", "320", str(outputs.pdf_path), str(raster_root)],
            cwd=outputs.tex_path.parent,
        )
        outputs.png_path.write_bytes((raster_root.with_suffix(".png")).read_bytes())
    else:
        if convert is None:
            raise SystemExit("required tool not found: convert")
        run_command(
            [
                convert,
                "-background",
                "white",
                "-alpha",
                "remove",
                "-alpha",
                "off",
                "-density",
                "320",
                str(outputs.svg_path),
                "-resize",
                "2400x",
                str(outputs.png_path),
            ],
            cwd=outputs.tex_path.parent,
        )


def load_timeline_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fin:
        reader = csv.DictReader(fin)
        rows = list(reader)
    if not rows:
        raise SystemExit(f"timeline csv is empty: {path}")
    return rows


def float_or_zero(value: str | None) -> float:
    if value is None:
        return 0.0
    text = str(value).strip()
    if not text:
        return 0.0
    try:
        return float(text)
    except ValueError:
        return 0.0


def int_or_zero(value: str | None) -> int:
    if value is None:
        return 0
    text = str(value).strip()
    if not text:
        return 0
    try:
        return int(float(text))
    except ValueError:
        return 0


def choose_timeline_x_field(rows: list[dict[str, str]], requested: str | None) -> tuple[str, str]:
    if requested and requested != "auto":
        if requested not in rows[0]:
            raise SystemExit(f"x field not found in timeline: {requested}")
        label = {
            "elapsed_minutes": "Elapsed Time (minutes)",
            "suite_elapsed_seconds": "Elapsed Time (seconds)",
            "wall_elapsed_seconds": "Wall-Clock Time (seconds)",
            "step_index": "Completed Step Index",
        }.get(requested, requested.replace("_", " ").title())
        return requested, label

    if "elapsed_minutes" in rows[0]:
        return "elapsed_minutes", "Elapsed Time (minutes)"
    if "suite_elapsed_seconds" in rows[0]:
        return "suite_elapsed_seconds", "Elapsed Time (seconds)"
    return "step_index", "Completed Step Index"


def prepare_timeline_rows(
    rows: list[dict[str, str]],
    metric_group: str,
    x_field: str,
    *,
    prepend_origin: bool,
) -> list[dict[str, object]]:
    if metric_group == "overall":
        percent_field = "overall_line_percent"
        covered_field = "overall_covered_lines"
    else:
        percent_field = f"{metric_group}_line_percent"
        covered_field = f"{metric_group}_covered_lines"

    missing = [field for field in (x_field, percent_field, covered_field) if field not in rows[0]]
    if missing:
        raise SystemExit(f"timeline csv missing required fields: {missing}")

    prepared: list[dict[str, object]] = []
    for row in rows:
        coverage_available = str(row.get("coverage_available", "")).strip().lower()
        if coverage_available and coverage_available not in {"yes", "true", "1"}:
            continue
        prepared.append(
            {
                "x": float_or_zero(row.get(x_field)),
                "coverage_percent": float_or_zero(row.get(percent_field)),
                "covered_lines": int_or_zero(row.get(covered_field)),
                "job_name": row.get("job_name", ""),
                "job_result": row.get("job_result", ""),
            }
        )
    prepared.sort(key=lambda item: (float(item["x"]), str(item["job_name"])))
    if prepend_origin and prepared:
        first = prepared[0]
        first_x = float(first["x"])
        first_percent = float(first["coverage_percent"])
        first_lines = int(first["covered_lines"])
        needs_origin = first_x > 0.0 or first_percent > 0.0 or first_lines > 0
        if needs_origin:
            prepared.insert(
                0,
                {
                    "x": 0.0,
                    "coverage_percent": 0.0,
                    "covered_lines": 0,
                    "job_name": "origin",
                    "job_result": "ORIGIN",
                },
            )
    return prepared


def expand_step_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    if not rows:
        return []
    expanded: list[dict[str, object]] = [dict(rows[0])]
    previous = rows[0]
    for current in rows[1:]:
        current_x = float(current["x"])
        previous_x = float(previous["x"])
        if current_x > previous_x:
            expanded.append(
                {
                    "x": current_x,
                    "coverage_percent": previous["coverage_percent"],
                    "covered_lines": previous["covered_lines"],
                    "job_name": current.get("job_name", ""),
                    "job_result": current.get("job_result", ""),
                }
            )
        expanded.append(dict(current))
        previous = current
    return expanded


def write_csv_rows(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def render_timeline_tex(
    outputs: FigureOutputs,
    title: str,
    subtitle: str,
    x_label: str,
    x_min: float,
    x_max: float,
    y_left_max: float,
    y_right_max: int,
    last_x: float,
    last_percent: float,
    last_lines: int,
) -> str:
    title_tex = latex_escape(title)
    subtitle_tex = latex_escape(subtitle)
    x_label_tex = latex_escape(x_label)
    last_lines_label = latex_escape(f"{int(last_lines):,}")
    subtitle_block = ""
    if subtitle.strip():
        subtitle_block = rf"""
\node[
    anchor=north west,
    font=\scriptsize\itshape,
    text=chartsubtle,
    align=left
] at ($(current bounding box.north west)+(0.12,-0.28)$) {{{subtitle_tex}}};
"""
    return rf"""
\documentclass[tikz,border=4pt]{{standalone}}
{SERIF_PREAMBLE}
\begin{{document}}
\begin{{tikzpicture}}
\begin{{axis}}[
    width=13.6cm,
    height=7.8cm,
    axis x line*=bottom,
    axis y line*=left,
    xmin={x_min:.3f},
    xmax={x_max:.3f},
    ymin=0,
    ymax={y_left_max:.2f},
    enlarge x limits={{abs=1.25}},
    enlarge y limits={{upper, value=0.08}},
    xlabel={{{x_label_tex}}},
    ylabel={{Coverage (\%)}},
    xlabel style={{font=\small,text=charttext}},
    ylabel style={{font=\small,text={TIMELINE_COLOR_COVERAGE}}},
    tick label style={{font=\small,text=chartsubtle}},
    title={{{title_tex}}},
    title style={{font=\large\bfseries,text=charttext,yshift=-0.2em}},
    grid=major,
    grid style={{draw=chartgrid}},
    major grid style={{draw=chartgrid}},
    minor tick num=0,
    tick align=outside,
    line width=1.2pt,
    axis line style={{charttext}},
    ymajorgrids=true,
    xmajorgrids=true,
    x grid style={{draw=chartgrid,dash pattern=on 3pt off 4pt}},
    y grid style={{draw=chartgrid}},
    legend columns=2,
    legend cell align=left,
    legend style={{
        at={{(0.02,0.98)}},
        anchor=north west,
        font=\small,
        draw=chartgrid,
        fill=white,
        fill opacity=0.94,
        text opacity=1,
        rounded corners=2pt,
        /tikz/every even column/.style={{column sep=0.8cm}},
    }},
    scaled y ticks=false,
    clip=false,
]
\addplot[
    color={TIMELINE_COLOR_COVERAGE},
    mark=*,
    mark size=2.2pt,
    line width=2.4pt,
    line cap=round,
    line join=round,
] table [x=x, y=coverage_percent, col sep=comma] {{{outputs.data_path.as_posix()}}};
\addlegendentry{{Coverage (\%)}}
\addlegendimage{{color={TIMELINE_COLOR_LINES},line width=2.4pt,mark=*,mark size=2.2pt}}
\addlegendentry{{Covered statements}}
\node[
    anchor=west,
    font=\bfseries\scriptsize,
    text={TIMELINE_COLOR_COVERAGE},
    xshift=8pt,
    yshift=6pt,
    fill=white,
    fill opacity=0.9,
    text opacity=1,
    rounded corners=2pt,
    inner xsep=3pt,
    inner ysep=2pt,
] at (axis cs:{last_x:.6f},{last_percent:.6f}) {{{last_percent:.2f}\%}};
\end{{axis}}

\begin{{axis}}[
    width=13.6cm,
    height=7.8cm,
    axis x line=none,
    axis y line*=right,
    xmin={x_min:.3f},
    xmax={x_max:.3f},
    ymin=0,
    ymax={y_right_max},
    enlarge x limits={{abs=1.25}},
    enlarge y limits={{upper, value=0.08}},
    ylabel={{Covered statements}},
    ylabel style={{font=\small,text={TIMELINE_COLOR_LINES}}},
    tick label style={{font=\small,text=chartsubtle}},
    ymajorgrids=false,
    line width=1.2pt,
    axis line style={{charttext}},
    scaled y ticks=false,
    yticklabel style={{/pgf/number format/fixed,/pgf/number format/1000 sep={{,}}}},
    clip=false,
]
\addplot[
    color={TIMELINE_COLOR_LINES},
    mark=*,
    mark size=2.2pt,
    line width=2.4pt,
    line cap=round,
    line join=round,
] table [x=x, y=covered_lines, col sep=comma] {{{outputs.data_path.as_posix()}}};
\node[
    anchor=west,
    font=\bfseries\scriptsize,
    text={TIMELINE_COLOR_LINES},
    xshift=8pt,
    yshift=-8pt,
    fill=white,
    fill opacity=0.9,
    text opacity=1,
    rounded corners=2pt,
    inner xsep=3pt,
    inner ysep=2pt,
] at (axis cs:{last_x:.6f},{int(last_lines)}) {{{last_lines_label}}};
\end{{axis}}
{subtitle_block}
\end{{tikzpicture}}
\end{{document}}
"""


def generate_timeline_figure(args: argparse.Namespace) -> FigureOutputs:
    timeline_csv = Path(args.timeline_csv).expanduser().resolve()
    rows = load_timeline_rows(timeline_csv)
    if "elapsed_minutes" not in rows[0] and "suite_elapsed_seconds" in rows[0]:
        for row in rows:
            row["elapsed_minutes"] = f"{float_or_zero(row.get('suite_elapsed_seconds')) / 60.0:.6f}"

    x_field, x_label = choose_timeline_x_field(rows, args.x_field)
    prepared = prepare_timeline_rows(
        rows,
        args.metric_group,
        x_field,
        prepend_origin=args.prepend_origin,
    )
    if len(prepared) < 2:
        raise SystemExit(f"not enough timeline points for figure: {timeline_csv}")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    basename = args.basename
    outputs = FigureOutputs(
        tex_path=output_dir / f"{basename}.tex",
        data_path=output_dir / f"{basename}.data.csv",
        pdf_path=output_dir / f"{basename}.pdf",
        svg_path=output_dir / f"{basename}.svg",
        png_path=output_dir / f"{basename}.png",
    )

    plot_rows = expand_step_rows(prepared)
    write_csv_rows(outputs.data_path, ["x", "coverage_percent", "covered_lines", "job_name", "job_result"], plot_rows)
    max_percent = max(float(item["coverage_percent"]) for item in prepared)
    max_lines = max(int(item["covered_lines"]) for item in prepared)
    first_x = float(prepared[0]["x"])
    last_x = float(prepared[-1]["x"])
    if first_x > 0.0 and not args.prepend_origin:
        left_pad = min(5.0, max(1.0, 0.08 * max(last_x - first_x, 1.0)))
        x_min = max(0.0, first_x - left_pad)
    else:
        x_min = 0.0
    right_pad = min(5.0, max(1.0, 0.05 * max(last_x - x_min, 1.0)))
    x_max = last_x + right_pad
    y_left_max = max(5.0, round(max_percent * 1.18 + 0.5, 1))
    y_right_max = max(100, int(max_lines * 1.15) + 1)
    last_point = prepared[-1]
    last_x = float(last_point["x"])
    last_percent = float(last_point["coverage_percent"])
    last_lines = int(last_point["covered_lines"])

    if args.title:
        title = args.title
    elif args.metric_group == "overall":
        title = "Overall Coverage Growth"
    else:
        title = "Scalar Target Coverage Growth"

    if args.subtitle:
        subtitle = args.subtitle
    elif args.metric_group == "overall":
        subtitle = "Overall coverage and covered lines across cumulative completed steps"
    else:
        subtitle = "Scalar target coverage and covered lines across cumulative completed steps"

    tex = render_timeline_tex(
        outputs,
        title=title,
        subtitle=subtitle,
        x_label=x_label,
        x_min=x_min,
        x_max=x_max,
        y_left_max=y_left_max,
        y_right_max=y_right_max,
        last_x=last_x,
        last_percent=last_percent,
        last_lines=last_lines,
    )
    write_text(outputs.tex_path, tex)
    compile_figure(outputs, build_dir=output_dir / ".build" / basename)
    return outputs


def load_summary_groups(summary_json: Path) -> list[dict[str, object]]:
    payload = json.loads(summary_json.read_text(encoding="utf-8"))
    coverage = payload.get("coverage", {})
    scalar_groups = coverage.get("scalar_groups", {})
    if not scalar_groups:
        raise SystemExit(f"summary does not contain scalar_groups: {summary_json}")

    rows: list[dict[str, object]] = []
    preferred_order = [
        "scalar_target",
        "field_index",
        "payload_storage",
        "payload_mutation_update",
        "query_optimization",
        "payload_index_core",
        "query_api_stack",
        "facet_related",
        "full_text_index",
        "formula_rescore",
        "state_recovery",
    ]
    for name in preferred_order:
        group = scalar_groups.get(name)
        if not isinstance(group, dict):
            continue
        rows.append(
            {
                "group": name,
                "line_percent": float(group.get("line_percent") or 0.0),
                "covered_lines": int(group.get("covered_lines") or 0),
                "total_lines": int(group.get("lines") or 0),
            }
        )
    return rows


def render_summary_groups_tex(
    outputs: FigureOutputs,
    title: str,
    group_order: list[str],
    group_labels: dict[int, str],
    count_annotations: dict[int, tuple[float, str]],
    x_max: float,
) -> str:
    title_tex = latex_escape(title)
    ytick_positions = ",".join(str(idx) for idx in range(1, len(group_order) + 1))
    label_lines = []
    for y_index, label in group_labels.items():
        escaped = latex_escape(label)
        label_lines.append(
            rf"\node[anchor=west,font=\scriptsize,text=black,fill=white,fill opacity=0.85,text opacity=1,inner sep=1.2pt] at (axis cs:{x_max*0.01:.2f},{y_index}) {{{escaped}}};"
        )
    count_lines = []
    for y_index, (x_pos, label) in count_annotations.items():
        escaped = latex_escape(label)
        count_lines.append(
            rf"\node[anchor=west,font=\scriptsize,text={SUMMARY_ACCENT_COLOR},fill=white,fill opacity=0.70,text opacity=1,inner sep=1.0pt] at (axis cs:{x_pos:.2f},{y_index}) {{{escaped}}};"
        )
    labels_tex = "\n".join(label_lines)
    counts_tex = "\n".join(count_lines)
    return rf"""
\documentclass[tikz,border=4pt]{{standalone}}
{SERIF_PREAMBLE}
\begin{{document}}
\begin{{tikzpicture}}
\begin{{axis}}[
    width=14.2cm,
    height=8.0cm,
    xbar,
    xmin=0,
    xmax={x_max:.2f},
    enlarge y limits=0.08,
    bar width=6.0pt,
    ytick={{{ytick_positions}}},
    yticklabels={{}},
    y dir=reverse,
    xlabel={{Line Coverage (\%)}},
    xlabel style={{font=\small}},
    tick label style={{font=\small}},
    title={{{title_tex}}},
    title style={{font=\normalsize\bfseries}},
    grid=major,
    grid style={{draw=gray!18}},
    major grid style={{draw=gray!18}},
    axis line style={{black}},
    tick align=outside,
    line width=0.9pt,
    clip=false,
]
\addplot[
    fill={SUMMARY_BAR_COLOR},
    draw={SUMMARY_BAR_COLOR},
] table [x=line_percent, y=y_index, col sep=comma] {{{outputs.data_path.as_posix()}}};
{labels_tex}
{counts_tex}
\end{{axis}}
\end{{tikzpicture}}
\end{{document}}
"""


def generate_summary_groups_figure(args: argparse.Namespace) -> FigureOutputs:
    summary_json = Path(args.summary_json).expanduser().resolve()
    rows = load_summary_groups(summary_json)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    basename = args.basename
    outputs = FigureOutputs(
        tex_path=output_dir / f"{basename}.tex",
        data_path=output_dir / f"{basename}.data.csv",
        pdf_path=output_dir / f"{basename}.pdf",
        svg_path=output_dir / f"{basename}.svg",
        png_path=output_dir / f"{basename}.png",
    )
    figure_rows: list[dict[str, object]] = []
    group_order: list[str] = []
    for idx, row in enumerate(rows, start=1):
        label = pretty_group_label(str(row["group"]))
        line_percent = float(row["line_percent"])
        figure_rows.append(
            {
                "y_index": idx,
                "group": label,
                "line_percent": line_percent,
                "covered_lines": row["covered_lines"],
                "total_lines": row["total_lines"],
            }
        )
        group_order.append(label)
    write_csv_rows(outputs.data_path, ["y_index", "group", "line_percent", "covered_lines", "total_lines"], figure_rows)
    x_max = max(5.0, max(float(row["line_percent"]) for row in figure_rows) * 1.18 + 1.0)
    title = args.title or "Qdrant Scalar Subsystem Coverage Summary"
    group_labels = {}
    count_annotations = {}
    for row in figure_rows:
        idx = int(row["y_index"])
        label = str(row["group"])
        line_percent = float(row["line_percent"])
        group_labels[idx] = label
        count_x = min(max(line_percent + max(1.0, x_max * 0.015), x_max * 0.33), x_max * 0.87)
        count_annotations[idx] = (count_x, f"{int(row['covered_lines'])}/{int(row['total_lines'])}")
    tex = render_summary_groups_tex(
        outputs,
        title=title,
        group_order=group_order,
        group_labels=group_labels,
        count_annotations=count_annotations,
        x_max=x_max,
    )
    write_text(outputs.tex_path, tex)
    compile_figure(outputs, build_dir=output_dir / ".build" / basename)
    return outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate publication-style Qdrant coverage figures")
    sub = parser.add_subparsers(dest="mode", required=True)

    timeline = sub.add_parser("timeline", help="Generate a dual-axis line chart from coverage_timeline.csv")
    timeline.add_argument("--timeline-csv", required=True, help="Path to coverage_timeline.csv")
    timeline.add_argument("--metric-group", default="scalar_target", help="Coverage group prefix, e.g. scalar_target or overall")
    timeline.add_argument("--x-field", default="auto", help="X-axis field; auto prefers elapsed_minutes, then suite_elapsed_seconds, then step_index")
    timeline.add_argument(
        "--prepend-origin",
        action="store_true",
        help="Insert a synthetic origin point when the first timeline snapshot starts later than zero",
    )
    timeline.add_argument("--title", default=None, help="Optional custom title")
    timeline.add_argument("--subtitle", default=None, help="Optional custom subtitle")
    timeline.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for PDF/SVG/PNG/TEX outputs")
    timeline.add_argument("--basename", default="fig_qdrant_scalar_timeline", help="Base filename without extension")

    summary = sub.add_parser("summary-groups", help="Generate a scalar subsystem summary figure from summary.json")
    summary.add_argument("--summary-json", required=True, help="Path to summary.json")
    summary.add_argument("--title", default=None, help="Optional custom title")
    summary.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for PDF/SVG/PNG/TEX outputs")
    summary.add_argument("--basename", default="fig_qdrant_scalar_subsystem_summary", help="Base filename without extension")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.mode == "timeline":
        outputs = generate_timeline_figure(args)
    elif args.mode == "summary-groups":
        outputs = generate_summary_groups_figure(args)
    else:
        raise SystemExit(f"unknown mode: {args.mode}")

    print(f"tex:  {outputs.tex_path}")
    print(f"data: {outputs.data_path}")
    print(f"pdf:  {outputs.pdf_path}")
    print(f"svg:  {outputs.svg_path}")
    print(f"png:  {outputs.png_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
