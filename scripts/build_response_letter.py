from pathlib import Path

from docx import Document
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls, qn
from docx.shared import Cm, Inches, Pt, RGBColor

OUT = Path(r"g:\My Drive\LU-STW(Hungarian Algorithm_ CDRS)\Response_to_Reviewers_TITR-2026-0877.docx")
TEXT = RGBColor(0x11, 0x11, 0x11)
HEADER_FILL = "F2F2F2"
BORDER = "222222"

ROWS = [
    ("Comment", "Response"),
    (
        "1. The novelty of combining Hungarian-based latent-factor alignment with cosine-similarity-based learning-rate adaptation is not clearly established. Explain how the method differs from existing cross-domain alignment and personalized transfer methods.",
        "Thank you for this comment. The novelty is rewritten as an ordered three-phase pipeline, and LU-STW is clearly compared with existing alignment and personalized-transfer methods.\n(Section 2 — Related Work, novelty paragraphs)\n(Table 1 — Novelty positioning)",
    ),
    (
        "2. Independently trained MF may differ by rotation. Explain why permutation-based alignment is sufficient and compare it with orthogonal Procrustes.",
        "Thank you for raising this point. Why permutation is sufficient is explained, and an orthogonal Procrustes comparison is added.\n(Section 3.4 — Phase 2, Permutation alignment versus orthogonal Procrustes)\n(Table 3 — Phase 2 alignment variants)",
    ),
    (
        "3. Show whether the confidence score indicates how useful source information is for a user, and explain why the score remains unchanged during fine-tuning.",
        "We appreciate this request. A tercile analysis is added, and the fixed-ωu choice is explained.\n(Section 3.5 — Phase 3, Relation between ωu and transfer benefit)\n(Table 15 — ωu tercile analysis)\n(Section 6 — Limitations and Future Directions)",
    ),
    (
        "4. Explain why the confidence-based learning rate is applied to item factors and biases, and compare with user-only weighting.",
        "Thank you for this suggestion. The choice is explained, and a user-only weighting comparison is added.\n(Section 3.5 — Phase 3, Confidence weighting of item factors and item biases)\n(Table 4 — Full versus user-only confidence weighting)",
    ),
    (
        "5. Report the Romance–Drama movie overlap and repeat the experiments with no common movies.",
        "Thank you for this careful reading. The overlap is reported, and a no-common-movies re-run is added.\n(Section 4.1 — Experimental Setup, Item-overlap quantification)\n(Table 6 — Overlapping-genre vs genre-disjoint)\n(Section 6 — Limitations and Future Directions)",
    ),
    (
        "6. Report Precision@K, Recall@K, NDCG@K, and MRR, and explain the choice of source–target pairs.",
        "Thank you for this suggestion. Ranking metrics are added, and the domain-pair choice is explained.\n(Section 4.2 — Evaluation Metrics)\n(Section 4.1 — Experimental Setup, Rationale for source–target pairs)\n(Table 10 — Ranking metrics)",
    ),
    (
        "7. Ensure fair baseline comparisons and conduct multi-seed experiments with a more suitable statistical analysis.",
        "Thank you for this comment. Fair baseline settings are clarified, and a multi-seed statistical check is added.\n(Section 4.1 — Experimental Setup, Baseline fairness and multi-seed testing)\n(Table 7 — Hyperparameter and reproducibility settings)\n(Table 12 — Multi-seed evaluation)",
    ),
    (
        "8. Repeat the ablation at all three training densities, check the small Amazon improvements, and report training and inference costs separately.",
        "Thank you for these points. Ablation is repeated at all three densities, the Amazon gaps are checked against fold variation, and training and inference costs are reported separately.\n(Section 3.6 — Computational Complexity Analysis, Training versus inference)\n(Table 13 — Ablation at 50%)\n(Table 14 — Ablation at 25% and 75%)",
    ),
    (
        "9. Include more recent work (2023–2026) on domain adaptation, interest alignment, selective knowledge transfer, and negative transfer, and compare LU-STW with them.",
        "Thank you for this suggestion. Recent work is discussed and cited, and LU-STW is compared with these approaches.\n(Section 2 — Related Work, Personalized Cross-Domain Transfer)\n(Table 2 — Summary of previous models)",
    ),
]


def set_run(run, *, size=11, bold=False):
    run.font.name = "Times New Roman"
    run._element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = TEXT


def add_para(doc, text="", *, size=11, bold=False, center=False, space_after=6):
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.space_after = Pt(space_after)
    pf.space_before = Pt(0)
    pf.line_spacing = 1.15
    if center:
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    if text:
        set_run(p.add_run(text), size=size, bold=bold)
    return p


def add_labeled(doc, label, value):
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.space_after = Pt(2)
    pf.space_before = Pt(0)
    pf.line_spacing = 1.15
    set_run(p.add_run(label), size=11, bold=True)
    set_run(p.add_run(value), size=11, bold=False)
    return p


def shade_cell(cell, hex_color):
    tc_pr = cell._tc.get_or_add_tcPr()
    tc_pr.append(parse_xml(f'<w:shd {nsdecls("w")} w:fill="{hex_color}" w:val="clear"/>'))


def set_cell_borders(cell):
    tc_pr = cell._tc.get_or_add_tcPr()
    tc_pr.append(
        parse_xml(
            f'<w:tcBorders {nsdecls("w")}>'
            f'<w:top w:val="single" w:sz="8" w:space="0" w:color="{BORDER}"/>'
            f'<w:left w:val="single" w:sz="8" w:space="0" w:color="{BORDER}"/>'
            f'<w:bottom w:val="single" w:sz="8" w:space="0" w:color="{BORDER}"/>'
            f'<w:right w:val="single" w:sz="8" w:space="0" w:color="{BORDER}"/>'
            f"</w:tcBorders>"
        )
    )


def set_cell_text(cell, text, *, bold=False):
    cell.text = ""
    cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.TOP
    for i, line in enumerate(text.split("\n")):
        p = cell.paragraphs[0] if i == 0 else cell.add_paragraph()
        pf = p.paragraph_format
        pf.space_after = Pt(2)
        pf.space_before = Pt(0)
        pf.line_spacing = 1.1
        set_run(p.add_run(line), size=11, bold=bold)
    set_cell_borders(cell)


def main():
    doc = Document()
    for section in doc.sections:
        section.page_width = Cm(21.0)
        section.page_height = Cm(29.7)
        section.left_margin = Cm(1.75)
        section.right_margin = Cm(1.65)
        section.top_margin = Cm(1.8)
        section.bottom_margin = Cm(1.8)

    style = doc.styles["Normal"]
    style.font.name = "Times New Roman"
    style.font.size = Pt(11)
    style._element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")

    add_para(doc, "Response to Reviewers", size=16, bold=True, center=True, space_after=12)
    add_labeled(doc, "Manuscript ID:", " TITR-2026-0877")
    add_labeled(
        doc,
        "Title:",
        " Confidence-Aware Learning of User-Specific Transfer Weights in Cross-Domain Recommendation Systems",
    )
    add_labeled(doc, "Journal:", " IETE Technical Review")
    add_para(doc, "", space_after=6)
    add_para(doc, "Dear Editor,", space_after=6)
    add_para(
        doc,
        "Thank you for the reviews. Every comment is answered in the table below. "
        "The location of each change in the revised manuscript is given in brackets. "
        "Revised text is marked in blue.",
        space_after=6,
    )
    add_para(doc, "Sincerely,", space_after=2)
    add_para(doc, "Vibhor Kant", space_after=0)
    add_para(doc, "Corresponding Author", space_after=6)
    add_labeled(doc, "Reviewer:", " 1")
    add_para(doc, "", space_after=6)

    table = doc.add_table(rows=len(ROWS), cols=2)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    w0, w1 = Inches(2.5), Inches(4.45)
    for i, (c, r) in enumerate(ROWS):
        cell0, cell1 = table.rows[i].cells
        cell0.width = w0
        cell1.width = w1
        set_cell_text(cell0, c, bold=(i == 0))
        set_cell_text(cell1, r, bold=(i == 0))
        if i == 0:
            shade_cell(cell0, HEADER_FILL)
            shade_cell(cell1, HEADER_FILL)

    tbl_grid = table._tbl.tblGrid
    for i, w in enumerate([w0, w1]):
        if i < len(tbl_grid.gridCol_lst):
            tbl_grid.gridCol_lst[i].set(qn("w:w"), str(int(w.twips)))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc.save(OUT)
    print("saved", OUT)


if __name__ == "__main__":
    main()
