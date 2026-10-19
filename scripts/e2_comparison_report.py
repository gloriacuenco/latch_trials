"""Add a completed, validated six-condition comparison to the LATCH progress PDF.

Run with the bundled PDF Python environment. The original is preserved. The original timeline
and discussion slides are omitted, so the updated deck ends with the next experiments. The PDF skill's
artifact-operation marker must be called before authoring, and the output must be rendered
and visually checked before delivery. This script refuses incomplete experiment results.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from io import BytesIO
import json
import math
from pathlib import Path
from zoneinfo import ZoneInfo

from pypdf import PdfReader, PdfWriter
from reportlab.lib.colors import HexColor
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph
from reportlab.lib.styles import ParagraphStyle

WIDTH, HEIGHT = 960, 540
NAVY = HexColor("#182743")
ORANGE = HexColor("#bc430b")
GRAY = HexColor("#59657a")
BACKGROUND = HexColor("#fafaf8")
BLUE = HexColor("#e7edf5")
BORDER = HexColor("#bfc2c3")


def fonts():
    base = Path("/System/Library/Fonts/Supplemental")
    for name, filename in (("ReportBody", "Arial.ttf"), ("ReportBold", "Arial Bold.ttf"),
                           ("ReportTitle", "Georgia Bold.ttf")):
        pdfmetrics.registerFont(TTFont(name, str(base / filename)))


def text(c, value, x, y, size=14, font="ReportBody", color=NAVY):
    c.setFillColor(color)
    c.setFont(font, size)
    c.drawString(x, y, value)


def paragraph(c, value, x, top, width, size=14, leading=None, color=NAVY):
    style = ParagraphStyle("body", fontName="ReportBody", fontSize=size,
                           leading=leading or size * 1.4, textColor=color)
    p = Paragraph(value, style)
    _, height = p.wrap(width, HEIGHT)
    p.drawOn(c, x, top - height)
    return height


def start_page(c, kicker, title):
    c.setFillColor(BACKGROUND)
    c.rect(0, 0, WIDTH, HEIGHT, stroke=0, fill=1)
    text(c, kicker.upper(), 64, 464, size=11, color=ORANGE)
    size = 30
    while pdfmetrics.stringWidth(title, "ReportTitle", size) > 832:
        size -= 1
    text(c, title, 64, 425, size=size, font="ReportTitle")


def label(row):
    names = {"random": "Random", "latent": "Latent OT", "anatomy": "Anatomy OT"}
    return names[row["coupling"]] + (", loss off" if row["anatomy_weight"] == 0 else ", loss on")


def validate(comparison, status):
    if status.get("status") != "complete":
        raise ValueError("The six-condition comparison has not completed; no report will be generated")
    rows = comparison["conditions"]
    keys = {(r["coupling"], r["anatomy_weight"]) for r in rows}
    expected = {(kind, weight) for kind in ("random", "latent", "anatomy") for weight in (0, 1)}
    if len(rows) != 6 or keys != expected:
        raise ValueError("Expected all six distinct pairing/anatomy-loss conditions")
    if comparison["protocol"].get("test_used") is not False:
        raise ValueError("Unexpected use of test data")
    for row in rows:
        if row["n_val_source"] != 92:
            raise ValueError("Every condition must contain 92 source-validation subjects")
        for key in ("dice", "cortical_dice", "volume_change", "probe_output_trained", "distance_closed", "psnr"):
            if not math.isfinite(float(row[key])):
                raise ValueError(f"Nonfinite result in {row['condition']}: {key}")
    return rows


def revised_method():
    """Replace the original encoding claim without leaving stale text under an overlay."""
    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=(WIDTH, HEIGHT))
    start_page(c, "The method", "A flow moves each scan's latent toward NYU")
    boxes = [(64, 120, "3D scan, site s", "#edf0f3"),
             (228, 120, "Frozen 3D<br/>encoder", "#dce6f2"),
             (393, 135, "Flow: site s to NYU", "#23548b"),
             (572, 120, "Frozen 3D<br/>decoder", "#dce6f2"),
             (736, 120, "Harmonized<br/>scan", "#edf0f3")]
    for i, (x, width, value, color) in enumerate(boxes):
        c.setFillColor(HexColor(color))
        c.roundRect(x, 336, width, 56, 6, fill=1, stroke=0)
        paragraph(c, value, x + 12, 377, width - 24, size=13, leading=17,
                  color=HexColor("#ffffff") if i == 2 else NAVY)
        if i < 4:
            c.setStrokeColor(GRAY)
            c.setLineWidth(3)
            c.line(x + width + 9, 364, x + width + 28, 364)
            p = c.beginPath()
            p.moveTo(x + width + 34, 364)
            p.lineTo(x + width + 25, 369)
            p.lineTo(x + width + 25, 359)
            p.close()
            c.setFillColor(GRAY)
            c.drawPath(p, fill=1, stroke=0)
    cards = [("Anatomy-matched pairing", "Training pairs a source scan with NYU scans of similar anatomy "
              "(FastSurfer regional volumes, age, sex), reducing anatomical differences within pairs."),
             ("Anatomy consistency", "A frozen anatomy reader checks that the flow's estimated endpoint "
              "retains the source scan's tissue-class layout."),
             ("Fixed-latent, 3D", "The flow is deterministic for a fixed latent. This comparison holds cached "
              "latents fixed; VAE re-encoding samples the posterior.")]
    for i, (title, body) in enumerate(cards):
        x = 64 + i * 283
        c.setFillColor(HexColor("#ffffff"))
        c.setStrokeColor(HexColor("#d8e0e9"))
        c.setLineWidth(0.5)
        c.roundRect(x, 177, 267, 138, 8, fill=1, stroke=1)
        text(c, title, x + 18, 280, size=16, font="ReportBold")
        paragraph(c, body, x + 18, 266, 231, size=13, leading=19, color=GRAY)
    c.save()
    buf.seek(0)
    return PdfReader(buf).pages[0]


def revised_overview(rows, date):
    """Refresh the cover and summary, preserving the original authorship."""
    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=(WIDTH, HEIGHT))
    c.setFillColor(BACKGROUND)
    c.rect(0, 0, WIDTH, HEIGHT, fill=1, stroke=0)
    text(c, "TRDP2 PROGRESS REPORT", 64, 419, size=12, color=ORANGE)
    text(c, "LATCH", 64, 324, size=72, font="ReportTitle")
    text(c, "Latent Anatomy-preserving Transport for Cross-site Harmonization", 64, 283, size=18)
    paragraph(c, "Progress since 1 October, including the completed six-condition pairing comparison", 64, 234, 800,
              size=17, leading=24, color=GRAY)
    text(c, "Muhammad Muqsit Islam | Gloria García Cuenco", 64, 100, size=14)
    text(c, f"Updated {date}", 64, 74, size=13, color=GRAY)
    c.showPage()
    start_page(c, "In one slide", "Anatomy loss helps; site information remains")
    off = [r for r in rows if not r["anatomy_weight"]]
    on = [r for r in rows if r["anatomy_weight"]]
    cards = [
        ("Pipeline ready", "1,112 ABIDE I scans prepared in 3D; two cached VAE representations. "
         "Every decoded comparison uses the same 92 source-validation volumes."),
        ("Six conditions complete", "Random, latent OT and anatomy OT pairing, each with anatomy loss off/on. "
         "10,000 training steps and one seed per condition; no test scans used."),
        ("Anatomy loss matters", f"Dice rises from {min(r['dice'] for r in off):.3f}-{max(r['dice'] for r in off):.3f} "
         f"to {min(r['dice'] for r in on):.3f}-{max(r['dice'] for r in on):.3f} across pairings. "
         "Anatomy-matched pairing does not lead preservation in this pilot."),
        ("Site stays readable", f"Output-trained latent site balanced accuracy remains "
         f"{min(r['probe_output_trained'] for r in rows):.3f}-{max(r['probe_output_trained'] for r in rows):.3f}. "
         "Fooling a frozen probe does not establish removal of site information."),
    ]
    for i, (title, body) in enumerate(cards):
        x, y = 64 + (i % 2) * 430, 239 - (i // 2) * 155
        c.setFillColor(BLUE)
        c.roundRect(x, y, 402, 138, 8, fill=1, stroke=0)
        text(c, title, x + 18, y + 109, size=17, font="ReportBold")
        paragraph(c, body, x + 18, y + 91, 366, size=14, leading=20)
    text(c, "Pilot evidence on validation; the final frozen-audit test evaluation is still pending.", 64, 38, size=11, color=GRAY)
    c.save()
    buf.seek(0)
    return PdfReader(buf)


def revised_reference():
    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=(WIDTH, HEIGHT))
    start_page(c, "Result 2 - site in the latents (E1)", "The site is readable in both VAE representations")
    for i, (number, body) in enumerate([
        ("1.00", "Linear site-probe balanced accuracy on both autoencoders' latents, including a central crop."),
        ("0.58", "Anatomy and demographics alone: an empirical comparison point, not a guaranteed floor."),
        ("0.06", "Chance across 17 sites; shuffled-label controls are near chance."),
    ]):
        x = 64 + i * 283
        text(c, number, x, 321, size=54, font="ReportTitle")
        paragraph(c, body, x, 292, 247, size=16, leading=23)
    paragraph(c, "The aim is to reduce appearance-related site information while preserving each person's anatomy. "
              "Interpret probe scores together with independent anatomy measurements, rather than treating 0.58 as a hard threshold.",
              64, 158, 832, size=16, leading=23)
    paragraph(c, "Probes trained on the training split and evaluated on validation, with balanced accuracy over 17 sites. "
              "CNN probes achieve 0.96-0.97. Anatomy features include FastSurfer volumes, age and sex.",
              64, 65, 832, size=11, leading=15, color=GRAY)
    c.save()
    buf.seek(0)
    return PdfReader(buf).pages[0]


def revised_pilots(comparison, rows):
    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=(WIDTH, HEIGHT))
    start_page(c, "Result 4 - the two original anatomy-OT pilots", "Anatomy consistency improves preservation")
    selected = [comparison["baseline"]] + [next(r for r in rows if r["coupling"] == "anatomy" and r["anatomy_weight"] == w)
                                              for w in (0, 1)]
    columns = [352, 160, 160, 160]
    headers = ["92 source-validation scans", "Autoencoder", "Pilot 1: loss off", "Pilot 2: loss on"]
    x, top = 64, 378
    for width, header in zip(columns, headers):
        c.setFillColor(HexColor("#e9e9e6"))
        c.setStrokeColor(BORDER)
        c.rect(x, top - 34, width, 34, stroke=1, fill=1)
        text(c, header, x + 9, top - 22, size=12, font="ReportBold")
        x += width
    metrics = [("FastSurfer Dice, all structures", "dice", "{:.3f}", 1),
               ("Cortical regions", "cortical_dice", "{:.3f}", 1),
               ("Median absolute regional-volume change", "volume_change", "{:.1f}%", 100),
               ("NYU intensity distance closed", "distance_closed", "{:.1f}%", 100),
               ("Output-trained site balanced accuracy", "probe_output_trained", "{:.3f}", 1),
               ("PSNR against input (dB)", "psnr", "{:.1f}", 1)]
    for i, (name, key, fmt, factor) in enumerate(metrics):
        x, y = 64, top - 34 - i * 32
        values = [name] + [fmt.format(factor * r[key]) for r in selected]
        for j, (width, value) in enumerate(zip(columns, values)):
            c.setFillColor(BLUE if i < 3 else BACKGROUND)
            c.rect(x, y - 32, width, 32, stroke=1, fill=1)
            if j == 0:
                text(c, value, x + 9, y - 21, size=12)
            else:
                c.setFont("ReportBody", 12)
                c.setFillColor(NAVY)
                c.drawRightString(x + width - 10, y - 21, value)
            x += width
    paragraph(c, "The anatomy term raises Dice from 0.581 to 0.798, while NYU intensity distance closed falls "
              "from 24.4% to 16.1%. Site information remains recoverable (output-trained BA 0.926).",
              64, 125, 832, size=14, leading=20)
    paragraph(c, "Original anatomy-OT pilots, seed 0, 10,000 steps. Anatomy metrics compare FastSurfer with the input. "
              "Linear site probes use 110 validation latents across 17 sites. These two runs are reused in the six-condition comparison.",
              64, 55, 832, size=11, leading=15, color=GRAY)
    c.save()
    buf.seek(0)
    return PdfReader(buf).pages[0]


def revised_open_problem(rows):
    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=(WIDTH, HEIGHT))
    c.setFillColor(NAVY)
    c.rect(0, 0, WIDTH, HEIGHT, fill=1, stroke=0)
    text(c, "OPEN PROBLEM", 64, 361, size=11, color=HexColor("#e4a65d"))
    paragraph(c, "Intensity distributions move toward NYU,<br/>but the site stays readable", 64, 330, 832,
              size=29, leading=39, color=HexColor("#ffffff"))
    paragraph(c, f"Across all six conditions, output-trained latent site balanced accuracy remains "
              f"{min(r['probe_output_trained'] for r in rows):.3f}-{max(r['probe_output_trained'] for r in rows):.3f}. "
              "Frozen-probe accuracy falls to 0.000-0.013, yet newly trained probes still recover the site. "
              "The anatomy loss improves preservation but does not directly penalize this residual site information.",
              64, 210, 832, size=17, leading=25, color=HexColor("#d0d8e4"))
    paragraph(c, "Frozen probes use 92 source-validation latents across 16 sites; output-trained probes use "
              "110 validation latents across 17 sites, including unchanged NYU. These cohorts are fixed across conditions.",
              64, 112, 832, size=11, leading=16, color=HexColor("#d0d8e4"))
    text(c, "The next task is to reduce recoverable site information while retaining the preservation gain.",
         64, 42, size=13, color=HexColor("#d0d8e4"))
    c.save()
    buf.seek(0)
    return PdfReader(buf).pages[0]


def revised_next_steps():
    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=(WIDTH, HEIGHT))
    start_page(c, "Next experiments", "Confirm the trade-off, then improve site removal")
    items = [
        ("Longer ablation", "Repeat the completed six-condition pilot with longer training and multiple seeds."),
        ("Anatomy weights", "Compare weights 3 and 10 under a fixed pairing protocol; measure preservation versus alignment."),
        ("Site removal", "Test a constraint targeting site information that retrained probes can still recover."),
        ("E2a", "Run a matched MAISI comparison, including a VAE-specific anatomy reader."),
        ("Repeatability", "Use posterior means or quantify repeated encoding; keep this separate from the fixed-cache ablation."),
        ("Final", "Choose settings on validation, then use the frozen ISBI test once; whole-volume and ABIDE II evaluations follow."),
    ]
    top, step = 377, 47
    for i, (name, body) in enumerate(items):
        y = top - i * step
        c.setFillColor(BLUE if i % 2 == 0 else BACKGROUND)
        c.rect(64, y - step, 832, step, fill=1, stroke=0)
        text(c, name, 74, y - 26, size=14, font="ReportBold")
        paragraph(c, body, 245, y - 9, 635, size=13, leading=17)
    text(c, "The pairing contribution remains a hypothesis; the pilot supports the anatomy-consistency loss more clearly.",
         64, 48, size=11, color=GRAY)
    c.save()
    buf.seek(0)
    return PdfReader(buf).pages[0]


def additions(comparison, rows, date):
    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=(WIDTH, HEIGHT))
    start_page(c, "Result 5 - six-condition E2 pilot ablation", "Pairing and anatomy control, compared")
    text(c, "Same cached NV-Generate-MR latents, 10,000 steps, seed 0; validation only", 64, 388, size=14)
    columns = [222, 78, 92, 112, 112, 116, 100]
    headings = ["Condition", "Dice", "Cortical\nDice", "Volume\nchange", "NYU distance\nclosed",
                "Output-trained\nsite BA", "PSNR\n(dB)"]
    baseline = {**comparison["baseline"], "name": "Autoencoder only"}
    ordered = [baseline] + [{**r, "name": label(r)} for r in sorted(rows, key=lambda r:
               (("random", "latent", "anatomy").index(r["coupling"]), r["anatomy_weight"]))]
    top, header_height, row_height = 364, 38, 31
    x = 64
    for width, header in zip(columns, headings):
        c.setFillColor(HexColor("#e9e9e6"))
        c.setStrokeColor(BORDER)
        c.rect(x, top - header_height, width, header_height, fill=1, stroke=1)
        for i, line in enumerate(header.split("\n")):
            text(c, line, x + 9, top - 15 - i * 13, size=11, font="ReportBold")
        x += width
    for i, row in enumerate(ordered):
        y = top - header_height - i * row_height
        values = [row["name"], f"{row['dice']:.3f}", f"{row['cortical_dice']:.3f}",
                  f"{100 * row['volume_change']:.1f}%", f"{100 * row['distance_closed']:.1f}%",
                  f"{row['probe_output_trained']:.3f}", f"{row['psnr']:.1f}"]
        x = 64
        for j, (width, value) in enumerate(zip(columns, values)):
            c.setFillColor(BLUE if row.get("anatomy_weight") == 1 else BACKGROUND)
            c.rect(x, y - row_height, width, row_height, fill=1, stroke=1)
            if j == 0:
                text(c, value, x + 9, y - 21, size=12, font="ReportBold" if i == 0 else "ReportBody")
            else:
                c.setFillColor(NAVY)
                c.setFont("ReportBody", 12)
                c.drawRightString(x + width - 10, y - 21, value)
            x += width
    paragraph(c, "Dice and regional-volume change compare FastSurfer on output with FastSurfer on input. "
              "NYU alignment uses foreground-intensity quantiles. Higher Dice, lower volume change, "
              "and lower retrained site BA are desirable; greater distance closed means closer intensity distributions.",
              64, 91, 832, size=11, leading=15, color=GRAY)
    text(c, f"Completed {date}. Anatomy OT off/on reuse the two original pilots; four conditions are new.",
         64, 24, size=10, color=GRAY)
    c.showPage()

    start_page(c, "Reading the comparison", "The anatomy loss drives the preservation gain")
    table = {(r["coupling"], r["anatomy_weight"]): r for r in rows}
    for i, coupling in enumerate(("random", "latent", "anatomy")):
        off, on = table[(coupling, 0)], table[(coupling, 1)]
        x = 64 + i * 283
        c.setFillColor(BLUE)
        c.roundRect(x, 238, 266, 145, 8, fill=1, stroke=0)
        text(c, {"random": "Random pairing", "latent": "Latent OT pairing", "anatomy": "Anatomy OT pairing"}[coupling],
             x + 15, 356, size=15, font="ReportBold")
        text(c, "Change when anatomy loss is enabled", x + 15, 332, size=11, color=GRAY)
        text(c, f"Dice: {on['dice'] - off['dice']:+.3f}", x + 15, 308, size=17, font="ReportBold")
        text(c, f"NYU distance closed: {100 * (on['distance_closed'] - off['distance_closed']):+.1f} pp",
             x + 15, 282, size=12)
        text(c, f"Output-trained site BA: {100 * (on['probe_output_trained'] - off['probe_output_trained']):+.1f} pp",
             x + 15, 259, size=12)
    best_anatomy = max(rows, key=lambda r: r["dice"])
    best_site = min(rows, key=lambda r: r["probe_output_trained"])
    paragraph(c, f"<b>No clear anatomy-pairing advantage:</b> all pairings with the loss reach Dice 0.798-0.807; "
              f"{label(best_anatomy)} is highest ({best_anatomy['dice']:.3f}). "
              f"Anatomy OT with loss has the lowest site BA ({best_site['probe_output_trained']:.3f}), "
              "but site remains highly readable. These pilots do not establish successful site removal.",
              64, 217, 832, size=14, leading=20)
    paragraph(c, "<b>Scope:</b> one seed and 10,000 steps per condition; 92 source-validation volumes for image/anatomy "
              "metrics. Output-trained probes: 110 validation latents, 17 sites; frozen probes: 92 source latents, 16 sites. "
              "NYU latents remain unchanged. "
              "No test scans were used. Close rankings need longer runs and multiple seeds.",
              64, 142, 832, size=12, leading=17, color=GRAY)
    paragraph(c, "<b>Reproducibility:</b> the flow is deterministic for a fixed latent; MONAI's encoder samples the VAE posterior. "
              "The existing latent cache was held fixed here. The anatomy/demographics probe's 0.58 score is an empirical "
              "reference, not a proven lower bound. Full provenance and per-subject metrics accompany the report.",
              64, 79, 832, size=11, leading=15, color=GRAY)
    c.showPage()
    c.save()
    buf.seek(0)
    return PdfReader(buf)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    comparison = json.loads(args.comparison.read_text())
    status = json.loads(args.status.read_text())
    rows = validate(comparison, status)
    if args.out.resolve() == args.source.resolve():
        raise ValueError("Output must preserve the original source report")
    fonts()
    date = datetime.fromisoformat(status["completed_at"]).astimezone(ZoneInfo("Europe/Paris")).strftime("%d %B %Y")
    added = additions(comparison, rows, date)
    overview = revised_overview(rows, date)
    original = PdfReader(args.source)
    if len(original.pages) != 16:
        raise ValueError("Expected the original 16-page progress report")
    writer = PdfWriter()
    # Retain the first 14 original slides, with the two comparison slides inserted below.
    # The original final two slides are Timeline and For discussion.
    for i, page in enumerate(original.pages[:14]):
        replacements = {0: overview.pages[0], 1: overview.pages[1], 3: revised_method,
                        6: revised_reference, 10: lambda: revised_pilots(comparison, rows),
                        12: lambda: revised_open_problem(rows),
                        13: revised_next_steps}
        replacement = replacements.get(i, page)
        writer.add_page(replacement() if callable(replacement) else replacement)
        if i == 11:
            for new_page in added.pages:
                writer.add_page(new_page)
    writer.add_metadata({"/Title": "LATCH Progress Report - Updated", "/Subject": "Verified six-condition E2 pairing pilot comparison"})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("wb") as stream:
        writer.write(stream)
    if len(PdfReader(args.out).pages) != 16:
        raise ValueError("Updated report should contain 16 pages")
    print(args.out.resolve())


if __name__ == "__main__":
    main()
