"""Human audit of the ReXErr sentence-level labels by a radiologist.

make:    draw a stratified sample of sentences from the test split and write a blinded Excel workbook for the reader
         plus an answer key (identifiers and assigned labels only, no text).
analyze: compare one or two filled workbooks with the key.

Strata (one sentence per study, so every row comes from a different report):
  error      sentences labeled erroneous, the same number for each of the 11 error types;
  changed    sentences labeled unchanged although their text differs from the original sentence;
  neutral    sentences labeled neutral (references a prior or no clinically relevant content) whose text differs;
  identical  sentences labeled unchanged and identical to the original (attention checks).
The reader sees the original report, the aligned original sentence and the modified sentence, never the label.
The workbook contains report text, so it is not distributed with the code.

usage: python scripts/label_audit.py make --data_dir DATA --out audit.xlsx --key results/label_audit_key.csv
       python scripts/label_audit.py analyze --key results/label_audit_key.csv --sheets filled_A.xlsx [filled_B.xlsx]
"""
import argparse
import json
import os
import sys
from difflib import SequenceMatcher

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rexerr.data import load_reports, load_sentences, normalize_text  # noqa: E402
from rexerr.labels import NATIVE, NATIVE_DISPLAY, normalize_type  # noqa: E402

DEFINITIONS = {
    "add_medical_device": "a medical device (line, tube, catheter, pacemaker...) is mentioned that is not in the original report",
    "false_prediction": "a finding is reported that is not in the original report",
    "false_negation": "a finding is wrongly negated (stated as absent)",
    "add_repetitions": "a statement is repeated unnecessarily",
    "add_contradictions": "two statements in the report contradict each other",
    "change_name_of_device": "a device is named incorrectly",
    "change_position_of_device": "the position of a device is described incorrectly",
    "change_severity": "the severity or size of a finding is misstated",
    "change_location": "the anatomical location or laterality of a finding is wrong",
    "change_measurement": "a measurement or number is wrong",
    "change_to_homophone": "a word is replaced by a similar-sounding wrong word (speech-recognition error)",
    "add_typo": "a typographical error",
}
TYPE_CHOICES = [NATIVE_DISPLAY[c] for c in NATIVE] + ["Other"]
SIZES = {"error": 18, "changed": 26, "neutral": 16, "identical": 10}


def sample(data_dir, seed):
    sent = load_sentences(data_dir, "test")
    rep = load_reports(data_dir, "test").set_index("study_id")
    sent = sent.assign(row=np.arange(len(sent)), o=sent["original_sentence"].map(normalize_text),
                       e=sent["error_sentence"].map(normalize_text),
                       t=[normalize_type(x) if p == 1 else "" for x, p in zip(sent["error_type"], sent["error_present"])])
    sent = sent[sent["e"] != ""]
    strata = [(f"error:{c}", sent[(sent["error_present"] == 1) & (sent["t"] == c)], SIZES["error"])
              for c in NATIVE if ((sent["error_present"] == 1) & (sent["t"] == c)).any()]
    strata += [("changed", sent[(sent["error_present"] == 0) & (sent["o"] != sent["e"])], SIZES["changed"]),
               ("neutral", sent[(sent["error_present"] == 2) & (sent["o"] != sent["e"])], SIZES["neutral"]),
               ("identical", sent[(sent["error_present"] == 0) & (sent["o"] == sent["e"])], SIZES["identical"])]
    rng = np.random.default_rng(seed)
    used, rows = set(), []
    for name, cand, n in strata:  # smallest strata first would also work; every stratum has far more candidates than n
        picked = 0
        for i in rng.permutation(len(cand)):
            r = cand.iloc[i]
            if r["study_id"] in used:
                continue
            used.add(r["study_id"])
            rows.append({"stratum": name.split(":")[0], "row": int(r["row"]), "study_id": int(r["study_id"]),
                         "subject_id": int(r["subject_id"]), "error_present": int(r["error_present"]),
                         "error_type": r["t"], "stratum_size": len(cand), "original_sentence": r["o"],
                         "modified_sentence": r["e"],
                         "original_report": normalize_text(rep.loc[r["study_id"], "original_report"])})
            picked += 1
            if picked == n:
                break
    df = pd.DataFrame(rows).sample(frac=1, random_state=seed).reset_index(drop=True)
    df.insert(0, "item", [f"S{i + 1:03d}" for i in range(len(df))])
    return df


def rich_diff(original, modified):
    """Modified sentence with words not in the aligned original sentence in bold red."""
    from openpyxl.cell.rich_text import CellRichText, TextBlock
    from openpyxl.cell.text import InlineFont
    if not original:
        return modified
    a, b = original.split(), modified.split()
    normal, changed = InlineFont(rFont="Arial", sz=10), InlineFont(rFont="Arial", sz=10, b=True, color="C00000")
    parts = []
    for tag, _, _, j1, j2 in SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if j1 < j2:
            parts.append((" ".join(b[j1:j2]), tag == "equal"))
    blocks = []
    for k, (txt, same) in enumerate(parts):
        blocks.append(TextBlock(normal if same else changed, txt + (" " if k < len(parts) - 1 else "")))
    return CellRichText(*blocks) if any(not s for _, s in parts) else modified


def lines(text, width):
    return sum(max(1, int(np.ceil(len(p) / width))) for p in str(text).split("\n"))


def write_workbook(df, path):
    from openpyxl import Workbook
    from openpyxl.comments import Comment
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
    from openpyxl.worksheet.datavalidation import DataValidation

    wb = Workbook()
    base, bold, title = Font(name="Arial", size=10), Font(name="Arial", size=10, bold=True), Font(name="Arial", size=14, bold=True)
    head_fill, input_fill = PatternFill("solid", fgColor="D9E1F2"), PatternFill("solid", fgColor="FFF2CC")
    thin = Side(style="thin", color="BFBFBF")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)
    wrap = Alignment(wrap_text=True, vertical="top")

    ins = wb.active
    ins.title = "Instructions"
    ins.column_dimensions["A"].width = 30
    ins.column_dimensions["B"].width = 95
    text = [
        ("Radiology report error audit", None, title),
        ("Purpose", "We are checking whether the automatically assigned error labels of ReXErr, a public dataset of "
                    "de-identified chest X-ray reports (MIMIC-CXR) into which errors were inserted by a language model, "
                    "are correct. Your answers will be compared with these labels; you will not see the labels.", None),
        ("What each row shows", "The original report (the reference, assumed to be correct), the sentence of the "
                                "original report that was modified, if any, and one sentence from the modified report. "
                                "Words that differ from the original sentence are shown in bold red. \"(new sentence)\" "
                                "means that the sentence was added to the report.", None),
        ("Your task", "Judge only the modified sentence, compared with the original report. Fill in the yellow cells "
                      "of the sheet \"Review\" (columns E–H) and please leave the other cells unchanged.", None),
        ("Error present? (column E)", "Yes: the sentence states something wrong or misleading compared with the "
                                      "original report (changed finding, negation, location, severity, measurement or "
                                      "device; spelling or speech-recognition error; unnecessary repetition). "
                                      "No: it conveys the same information as the original (rewording without a change "
                                      "of meaning is No). Unsure: you cannot decide.", None),
        ("Error type (column F)", "Only if Yes: the type that fits best (definitions below). If several apply, choose "
                                  "the clinically most important one; choose Other if none fits and explain in Comments.",
         None),
        ("Clinical significance (column G)", "Only if Yes. Minor: unlikely to change patient management (for example "
                                             "a typo whose meaning is obvious). Major: could change patient management "
                                             "or cause harm if not corrected.", None),
        ("Comments (column H)", "Optional, for example why you were unsure.", None),
        ("Practical notes", f"{len(df)} rows, about 1 minute each (3–4 hours in total); you can stop at any time and "
                            "continue later; save the file to keep your answers. Please work independently and do not "
                            "discuss the rows with anyone until you have finished.", None),
        ("Error types", None, bold),
    ] + [(NATIVE_DISPLAY[c], DEFINITIONS[c], None) for c in NATIVE] + [
        ("Other", "an error that fits none of the types above", None),
        ("Example (not part of the review)", None, bold),
        ("Original report", "Findings: Heart size is normal. No pleural effusion or pneumothorax. Impression: No acute "
                            "cardiopulmonary process.", None),
        ("Original sentence", "No pleural effusion or pneumothorax.", None),
        ("Modified sentence", "Small left pleural effusion. No pneumothorax.", None),
        ("Answers", "Error present: Yes · Error type: False prediction · Clinical significance: Major · Comments: "
                    "effusion added", None),
        ("بالعربي", "الهدف: التأكد من صحة تصنيف الأخطاء الآلي في مجموعة بيانات ReXErr. كل صف فيه التقرير الأصلي (المرجع "
                    "الصحيح) وجملة واحدة من التقرير المعدّل، والكلمات المختلفة باللون الأحمر. المطلوب الحكم على الجملة "
                    "المعدّلة مقارنة بالتقرير الأصلي فقط، وتعبئة الخلايا الصفراء في ورقة Review: هل فيها خطأ (Yes/No/Unsure)، "
                    "نوع الخطأ، وأهميته السريرية (Minor/Major)، وملاحظات اختيارية. إعادة الصياغة بدون تغيير المعنى = No. "
                    "حوالي دقيقة لكل صف، ويمكن الحفظ والإكمال لاحقاً. الرجاء العمل بشكل مستقل.", None),
    ]
    r = 1
    for a, b, font in text:
        ins.cell(r, 1, a).font = font or bold
        if b is not None:
            c = ins.cell(r, 2, b)
            c.font, c.alignment = base, wrap
            ins.row_dimensions[r].height = 14 * lines(b, 105) + 4
        ins.cell(r, 1).alignment = wrap
        r += 1 if font is not title else 2
    ins.sheet_view.showGridLines = False

    rv = wb.create_sheet("Review")
    cols = [("Item", 7), ("Original report (reference)", 60), ("Original sentence", 34), ("Modified sentence", 34),
            ("Error present?", 12), ("Error type", 22), ("Clinical significance", 13), ("Comments", 28)]
    for j, (name, w) in enumerate(cols, start=1):
        c = rv.cell(1, j, name)
        c.font, c.fill, c.alignment, c.border = bold, head_fill, Alignment(wrap_text=True, vertical="center"), border
        rv.column_dimensions[c.column_letter].width = w
    rv.cell(1, 5).comment = Comment("Yes / No / Unsure", "audit")
    rv.cell(1, 6).comment = Comment("Only if Yes; see the sheet Instructions for definitions", "audit")
    rv.cell(1, 7).comment = Comment("Only if Yes: Minor or Major", "audit")
    for i, x in enumerate(df.itertuples(), start=2):
        vals = [x.item, x.original_report, x.original_sentence or "(new sentence)",
                rich_diff(x.original_sentence, x.modified_sentence)]
        for j, v in enumerate(vals, start=1):
            c = rv.cell(i, j, v)
            c.font, c.alignment, c.border = base, wrap, border
        for j in range(5, 9):
            c = rv.cell(i, j)
            c.fill, c.border, c.font, c.alignment = input_fill, border, base, wrap
        n = max(lines(x.original_report, 66), lines(x.original_sentence, 36), lines(x.modified_sentence, 36))
        rv.row_dimensions[i].height = 13 * n + 6
    last = len(df) + 1
    for rng_, opts in ((f"E2:E{last}", ["Yes", "No", "Unsure"]), (f"F2:F{last}", TYPE_CHOICES),
                       (f"G2:G{last}", ["Minor", "Major"])):
        dv = DataValidation(type="list", formula1='"' + ",".join(opts) + '"', allow_blank=True,
                            error="Please choose a value from the list", showErrorMessage=True)
        rv.add_data_validation(dv)
        dv.add(rng_)
    rv.freeze_panes = "B2"
    rv.auto_filter.ref = f"A1:H{last}"
    for ws in (ins, rv):  # printable: one page wide, landscape for the review
        ws.page_setup.orientation = "landscape" if ws is rv else "portrait"
        ws.page_setup.paperSize = ws.PAPERSIZE_A4
        ws.page_setup.fitToWidth, ws.page_setup.fitToHeight = 1, 0
        ws.sheet_properties.pageSetUpPr.fitToPage = True
        ws.page_margins.left = ws.page_margins.right = 0.4
    rv.print_title_rows = "1:1"

    pr = wb.create_sheet("Progress")
    pr.column_dimensions["A"].width = 34
    pr.column_dimensions["B"].width = 14
    items = [("Rows", f"=COUNTA(Review!A2:A{last})", "0"),
             ("Rows answered (column E)", f"=COUNTA(Review!E2:E{last})", "0"),
             ("Share answered", "=IF(B1=0,0,B2/B1)", "0%"),
             ("Yes", f'=COUNTIF(Review!E2:E{last},"Yes")', "0"),
             ("No", f'=COUNTIF(Review!E2:E{last},"No")', "0"),
             ("Unsure", f'=COUNTIF(Review!E2:E{last},"Unsure")', "0"),
             ("Yes without an error type", f'=COUNTIFS(Review!E2:E{last},"Yes",Review!F2:F{last},"")', "0")]
    for k, (a, f, fmt) in enumerate(items, start=1):
        pr.cell(k, 1, a).font = bold
        c = pr.cell(k, 2, f)
        c.font, c.number_format = base, fmt
    pr.cell(9, 1, "Rater name").font = bold
    pr.cell(9, 2).fill = input_fill
    pr.cell(10, 1, "Read-only summary; fill in only the yellow cell.").font = Font(name="Arial", size=9, italic=True)
    wb.active = 1
    wb.save(path)


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, float(c - h)), min(1.0, float(c + h)))


def kappa(a, b):
    a, b = np.asarray(a), np.asarray(b)
    cats = sorted(set(a) | set(b))
    po = float(np.mean(a == b))
    pe = sum(float(np.mean(a == c)) * float(np.mean(b == c)) for c in cats)
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def read_sheet(path):
    df = pd.read_excel(path, sheet_name="Review", dtype=str).fillna("")
    df = df.rename(columns={"Item": "item", "Error present?": "present", "Error type": "type",
                            "Clinical significance": "significance", "Comments": "comments"})
    return df[["item", "present", "type", "significance", "comments"]]


def analyze(key, sheets):
    k = pd.read_csv(key, keep_default_na=False)
    disp = {NATIVE_DISPLAY[c]: c for c in NATIVE}
    out, raters = {}, []
    for s in sheets:
        d = k.merge(read_sheet(s), on="item", how="left").fillna("")
        d["type_key"] = d["type"].map(lambda t: disp.get(t, "other" if t else ""))
        raters.append(d)
        ans = d[d["present"].isin(["Yes", "No"])]
        res = {"answered": int(d["present"].ne("").sum()), "unsure": int((d["present"] == "Unsure").sum())}
        err = ans[ans["stratum"] == "error"]
        yes = int((err["present"] == "Yes").sum())
        res["error_labels_confirmed"] = {"k": yes, "n": len(err), "share": yes / max(1, len(err)),
                                         "ci95": wilson(yes, len(err))}
        # weighted by the number of test sentences of each type, since every type was sampled equally
        w = err.assign(yes=err["present"] == "Yes").groupby("error_type").agg(share=("yes", "mean"),
                                                                             size=("stratum_size", "first"))
        res["error_labels_confirmed_weighted"] = float((w["share"] * w["size"]).sum() / w["size"].sum()) if len(w) else None
        res["per_type_confirmed"] = {c: {"k": int((g["present"] == "Yes").sum()), "n": len(g)}
                                     for c, g in err.groupby("error_type")}
        conf = err[err["present"] == "Yes"]
        res["type_agreement"] = {"k": int((conf["type_key"] == conf["error_type"]).sum()), "n": len(conf),
                                 "kappa": kappa(conf["error_type"], conf["type_key"]) if len(conf) else None}
        res["significance_of_confirmed"] = conf["significance"].replace("", "missing").value_counts().to_dict()
        for st in ("changed", "neutral", "identical"):
            g = ans[ans["stratum"] == st]
            y = int((g["present"] == "Yes").sum())
            res[f"{st}_judged_erroneous"] = {"k": y, "n": len(g), "ci95": wilson(y, len(g))}
        out[os.path.basename(s)] = res
    if len(raters) == 2:
        a, b = raters
        both = a["present"].isin(["Yes", "No"]) & b["present"].isin(["Yes", "No"])
        out["inter_rater"] = {"n": int(both.sum()), "agreement": float((a["present"][both] == b["present"][both]).mean()),
                              "kappa_error_present": kappa(a["present"][both], b["present"][both])}
    print(json.dumps(out, indent=1, default=float))
    return out


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make")
    m.add_argument("--data_dir", required=True)
    m.add_argument("--out", required=True)
    m.add_argument("--key", required=True)
    m.add_argument("--seed", type=int, default=0)
    a = sub.add_parser("analyze")
    a.add_argument("--key", required=True)
    a.add_argument("--sheets", nargs="+", required=True)
    a.add_argument("--summary", default=None)
    args = ap.parse_args()
    if args.cmd == "make":
        df = sample(args.data_dir, args.seed)
        write_workbook(df, args.out)
        df[["item", "stratum", "row", "study_id", "subject_id", "error_present", "error_type", "stratum_size"]].to_csv(
            args.key, index=False)
        print(f"wrote {args.out} and {args.key}: {len(df)} rows;", df["stratum"].value_counts().to_dict())
    else:
        res = analyze(args.key, args.sheets)
        if args.summary:
            with open(args.summary, "w") as f:
                json.dump(res, f, indent=1, default=float)


if __name__ == "__main__":
    main()
