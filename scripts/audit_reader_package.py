"""Reader packages for the human audit: a plain-language Arabic guide (Word and PDF) and each reader's two workbooks.

The answer keys and sampling seeds are never copied. The examples in the guide are invented and do not appear in
the workbooks, so they cannot bias the answers.

usage: python scripts/audit_reader_package.py --audit_dir AUDIT --out_dir OUT [--author "Ahmed Alsaqqa"]
  AUDIT holds ReXErr_label_audit_Reader_{A,B}.xlsx and ReXErr_rewording_audit_Reader_{A,B}.xlsx
"""
import argparse
import datetime
import os
import re
import shutil
import subprocess
import zipfile

import openpyxl
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.worksheet.datavalidation import DataValidation

FONT, FONT_CS = "Arial", "Noto Sans Arabic"  # Arabic glyphs in the PDF; Word substitutes if the font is missing
YELLOW, GREY, BLUE = "FFF2CC", "F2F2F2", "DDEBF7"
MAIN, SHORT = "1_Main_review_250_rows_Reader_{}.xlsx", "2_Short_review_63_rows_Reader_{}.xlsx"
GUIDE = "0_Guide_read_first"

# ---------------------------------------------------------------- guide text (Arabic, English terms as in the files)
SUMMARY = [
    "ستصلك ملفّا Excel: الملف 1 (250 سطراً، حوالي 3–4 ساعات) والملف 2 (63 سطراً، حوالي ساعة).",
    "في كل سطر: تقرير أشعة صدر أصلي (نعتبره صحيحاً)، وجملة واحدة منه بعد تعديلها.",
    "المطلوب: هل الجملة المعدّلة فيها خطأ مقارنة بالتقرير الأصلي؟ وما نوعه؟ وما أهميته؟ كل ذلك بالاختيار من قوائم جاهزة.",
    "يمكن التوقف في أي وقت والإكمال لاحقاً، بشرط حفظ الملف.",
    "يُرجى العمل بشكل مستقل وعدم مناقشة الأسطر مع أي شخص حتى الانتهاء.",
    "عند الانتهاء: إعادة الملفين إلى من أرسلهما.",
]
DEADLINE_LINE = "آخر موعد لإعادة الملفين: قبل {deadline}. يكفي حوالي ساعة يومياً."
ABOUT = (
    "ندرس برامج حاسوب تكتشف الأخطاء في تقارير أشعة الصدر. استخدمنا مجموعة بيانات عامة للباحثين اسمها ReXErr، فيها "
    "تقارير حقيقية مجهولة الهوية من مستشفى أمريكي (MIMIC-CXR)، أُدخلت عليها أخطاء بشكل مقصود بواسطة برنامج ذكاء "
    "اصطناعي، ثم صنّف برنامج آخر كل جملة: فيها خطأ أم لا، وما نوعه. نريد أن يتحقق طبيب بنفسه من هذا التصنيف. "
    "ستُقارَن إجاباتك بتصنيف البرنامج، لكنك لن ترى تصنيف البرنامج، وهذا مقصود."
)
NO_IMAGES = "لا توجد صور أشعة. المقارنة دائماً بين نصّين: التقرير الأصلي والجملة المعدّلة."
FILES = [
    ("0_Guide_read_first.pdf", "—", "10 دقائق", "هذا الدليل"),
    (MAIN.format("X"), "250", "3–4 ساعات", "جمل من تقارير أُدخلت عليها أخطاء، وجمل بلا خطأ. ابدأ بهذا الملف."),
    (SHORT.format("X"), "63", "حوالي ساعة", "جمل أُعيدت صياغتها أو أُدخل عليها خطأ بطرق مختلفة. فيه عمود إضافي (I)."),
]
STEPS = [
    "افتح الملف على كمبيوتر باستخدام Excel (أو LibreOffice أو Google Sheets). الهاتف غير مناسب.",
    "إذا ظهر شريط أصفر في الأعلى مكتوب فيه Enable Editing، اضغط عليه.",
    "انتقل إلى الورقة المسمّاة Review (أسماء الأوراق في أسفل الشاشة).",
    "كل سطر = جملة واحدة. اقرأ بالترتيب: العمود B ثم C ثم D.",
    "عبّئ الخلايا الصفراء فقط (الأعمدة E إلى H، وفي الملف 2 حتى I). اضغط على الخلية فيظهر سهم صغير، واختر من القائمة.",
    "احفظ كثيراً (Ctrl+S). يمكنك إغلاق الملف والعودة إليه لاحقاً.",
    "تابع تقدّمك في الورقة المسمّاة Progress.",
]
COLUMNS = [
    ("A", "رقم", "رقم السطر. لا تغيّره."),
    ("B", "التقرير الأصلي (صحيح، للمقارنة فقط)", "التقرير الأصلي كاملاً كما كتبه الطبيب. هو المرجع ونعتبره صحيحاً."),
    ("C", "الجملة قبل التعديل", "الجملة كما كانت في التقرير الأصلي. إذا كُتب (new sentence) فلا توجد جملة أصلية: الجملة في "
                               "العمود D جديدة أُضيفت إلى التقرير."),
    ("D", "الجملة بعد التعديل: احكم عليها", "هذه الجملة التي تحكم عليها. الكلمات المكتوبة بالأحمر هي التي تغيّرت عن العمود C. "
                                          "الأحمر يدل على التغيير فقط وليس على وجود خطأ: كثير من التغييرات إعادة صياغة "
                                          "بنفس المعنى، وجوابها No."),
    ("E–H", "إجاباتك", "الخلايا الصفراء: الأسئلة الثلاثة في القسم التالي، وملاحظات اختيارية."),
]
Q1 = [
    ("Yes", "الجملة تقول شيئاً خاطئاً أو مضلّلاً مقارنة بالتقرير الأصلي: موجودة غير موجودة، نفي خاطئ، جهة خاطئة، شدّة أو "
            "قياس خاطئ، جهاز خاطئ أو في مكان خاطئ، خطأ إملائي، كلمة بصوت مشابه، أو تكرار غير ضروري."),
    ("No", "الجملة تنقل نفس المعلومة. إعادة الصياغة بكلمات مختلفة دون تغيير المعنى = No."),
    ("Unsure", "لا يمكنك الحسم. اكتب السبب باختصار في عمود Comments."),
]
NEW_SENTENCE = ("إذا كانت الجملة جديدة (new sentence): هل ما تقوله موجود أو مدعوم في التقرير الأصلي؟ إذا كان غير موجود فيه، "
                "فالإجابة Yes. وإذا كانت تكرّر جملة موجودة أصلاً، فالإجابة Yes ونوعها Add repetitions.")
TYPES = [
    ("Add medical device", "ذِكر جهاز طبي (أنبوب، قسطرة، منظم ضربات...) غير موجود في التقرير الأصلي",
     "(new sentence) → A right chest tube is in place."),
    ("False prediction", "ذِكر موجودة مرضية غير موجودة في التقرير الأصلي",
     "No pleural effusion. → Small left pleural effusion."),
    ("False negation", "نفي موجودة مرضية موجودة فعلاً", "Mild pulmonary edema. → No pulmonary edema."),
    ("Add repetitions", "تكرار جملة أو معلومة بلا داعٍ", "تكرار جملة موجودة في التقرير مرة ثانية"),
    ("Add contradictions", "جملتان في التقرير تناقض إحداهما الأخرى", "Heart size is normal. ... Marked cardiomegaly."),
    ("Change name of device", "تسمية الجهاز باسم خاطئ", "Feeding tube → Chest tube"),
    ("Change position of device", "وصف خاطئ لمكان الجهاز",
     "The feeding tube ends in the stomach. → ... ends in the distal esophagus."),
    ("Change severity", "تغيير شدّة أو حجم الموجودة", "Small left pleural effusion. → Large left pleural effusion."),
    ("Change location", "تغيير المكان التشريحي أو الجهة (يمين/يسار)",
     "Opacity in the right upper lobe. → Opacity in the left upper lobe."),
    ("Change measurement", "رقم أو قياس خاطئ", "Nodule measuring 8 mm → Nodule measuring 18 mm"),
    ("Change to homophone", "كلمة بدلاً من كلمة لها صوت مشابه (خطأ برامج الإملاء الصوتي)",
     "Small right pleural effusion. → Small right plural effusion."),
    ("Add typo", "خطأ إملائي في الكتابة", "The lungs are well expanded. → The lungs are well exapnded."),
    ("Other", "خطأ لا يناسب أي نوع مما سبق. اشرحه في Comments", "—"),
]
SEVERITY = [
    ("Minor", "غالباً لن يغيّر تشخيص المريض أو علاجه أو متابعته (مثل خطأ إملائي معناه واضح)."),
    ("Major", "قد يغيّر التشخيص أو العلاج، أو يسبب ضرراً إذا لم يُصحَّح."),
]
COL_I = [
    ("Yes", "طبيب يقرأ التقرير المعدّل وحده، دون الأصلي ودون الصورة، سيلاحظ الخطأ. مثل: خطأ إملائي، تكرار، أو جملة تناقض بقية "
            "التقرير."),
    ("No", "الجملة تبدو طبيعية ومقبولة، ولا يُكتشف الخطأ إلا بالرجوع إلى التقرير الأصلي أو إلى الصورة. مثل: تبديل اليمين "
           "باليسار."),
    ("Unsure", "لا يمكنك الحسم."),
]
EXAMPLES = [  # original sentence, modified sentence, E, F, G, why
    ("No pleural effusion or pneumothorax.", "Small left pleural effusion. No pneumothorax.", "Yes", "False prediction",
     "Major", "أُضيف انصباب جنبي غير موجود في الأصل."),
    ("Heart size is normal.", "The heart is normal in size.", "No", "", "", "نفس المعنى بكلمات مختلفة."),
    ("The lungs are well expanded.", "The lungs are well exapnded.", "Yes", "Add typo", "Minor",
     "خطأ إملائي والمعنى واضح."),
    ("Small right pleural effusion.", "Small right plural effusion.", "Yes", "Change to homophone", "Minor",
     "كلمة بصوت مشابه، والمعنى ما زال مفهوماً."),
    ("Opacity in the right upper lobe.", "Opacity in the left upper lobe.", "Yes", "Change location", "Major",
     "تغيّرت الجهة."),
    ("The feeding tube ends in the stomach.", "The feeding tube ends in the distal esophagus.", "Yes",
     "Change position of device", "Major", "مكان الأنبوب الجديد يستدعي تدخلاً."),
    ("(new sentence)", "A right chest tube is in place.", "Yes", "Add medical device", "Major",
     "التقرير الأصلي لا يذكر أنبوب صدر."),
    ("Blunting of the left costophrenic angle, which may reflect a small effusion or pleural thickening.",
     "Small left pleural effusion.", "Unsure", "", "", "الأصل يذكر احتمالين والمعدّلة جزمت بأحدهما؛ يُكتب السبب في Comments."),
]
RULES = [
    "احكم على الجملة المعدّلة فقط، مقارنةً بالتقرير الأصلي.",
    "التقرير الأصلي يُعتبر صحيحاً؛ لا داعي للحكم على صحته هو.",
    "لا توجد صور؛ لا تحاول تخمين ما في الصورة.",
    "إعادة الصياغة دون تغيير المعنى ليست خطأ (No).",
    "إذا وُجد أكثر من خطأ: اختر الأهم سريرياً، واذكر الآخر في Comments.",
    "لا تغيّر الأعمدة A إلى D، ولا ترتّب الأسطر (Sort)، ولا تحذف أو تضيف أسطراً.",
    "حوالي دقيقة للسطر تكفي. الحكم الأول المعقول غالباً هو الصحيح.",
    "مراجِع آخر يعمل على نفس الأسطر بشكل مستقل، ونقيس مدى اتفاقكما؛ لذلك يُرجى عدم مناقشة الأسطر معه أو مع غيره قبل أن "
    "ينتهي كلاكما.",
    "البيانات مجهولة الهوية ومتاحة للباحثين، لكن يُرجى عدم مشاركة الملفات مع أحد.",
]
DONE = [
    "افتح الورقة Progress وتأكد أن Share answered = 100%، وأن Yes without an error type = 0.",
    "عبّئ الورقة Reader info في الملف 1 (الاسم، التخصص، سنوات الخبرة). سنذكر في البحث التخصص وسنوات الخبرة فقط، "
    "ولن نذكر الاسم إلا بموافقتك.",
    "احفظ الملفين وأعِدهما إلى من أرسلهما، دون تغيير اسميهما.",
]
FAQ = [
    ("لا تظهر القائمة المنسدلة؟", "اضغط على الخلية الصفراء نفسها، فيظهر سهم صغير على يمينها. إذا لم يظهر، فاضغط Enable "
                                 "Editing أولاً."),
    ("هل يمكن استخدام Google Sheets؟", "نعم، والقوائم تعمل فيه. عند الانتهاء: File ← Download ← Microsoft Excel (.xlsx)، ثم "
                                     "أرسل الملف الناتج."),
    ("هل يمكن الكتابة بدل الاختيار من القائمة؟", "يُفضّل الاختيار من القائمة حتى تكون الإجابات موحّدة."),
    ("الجملة فيها خطأ موجود أصلاً في التقرير الأصلي؟", "احكم فقط على ما تغيّر. إذا كانت المعدّلة بنفس معنى الأصلية فالإجابة No."),
    ("النص طويل وصعب القراءة؟", "كبّر العرض (Zoom) أو وسّع العمود B بسحب حدّه."),
    ("الوقت لا يكفي دفعة واحدة؟", "قسّم العمل على جلسات، مثلاً 50 سطراً في الجلسة، واحفظ في كل مرة."),
]


# ---------------------------------------------------------------- docx helpers (right-to-left Arabic with English runs)
LATIN = re.compile(r"[A-Za-z0-9][A-Za-z0-9 .,:;()\-–/+%&'\"?=→←…]*[A-Za-z0-9)%\".?]|[A-Za-z0-9]")


def _rpr(run, size, bold, color, rtl):
    rpr = run._r.get_or_add_rPr()
    fonts = rpr.find(qn("w:rFonts"))
    if fonts is None:
        fonts = OxmlElement("w:rFonts")
        rpr.insert(0, fonts)
    for k in ("w:ascii", "w:hAnsi"):
        fonts.set(qn(k), FONT)
    fonts.set(qn("w:cs"), FONT_CS)
    run.font.size = Pt(size)
    szcs = OxmlElement("w:szCs")
    szcs.set(qn("w:val"), str(int(size * 2)))
    rpr.append(szcs)
    if bold:
        run.bold = True
        rpr.append(OxmlElement("w:bCs"))
    if color:
        run.font.color.rgb = RGBColor.from_string(color)
    if rtl:
        rpr.append(OxmlElement("w:rtl"))


def add_text(par, text, size=11, bold=False, color=None):
    """Arabic runs right-to-left, embedded English runs left-to-right."""
    pos = 0
    for m in LATIN.finditer(text):
        if m.start() > pos:
            _rpr(par.add_run(text[pos:m.start()]), size, bold, color, True)
        _rpr(par.add_run(m.group(0)), size, bold, color, False)
        pos = m.end()
    if pos < len(text):
        _rpr(par.add_run(text[pos:]), size, bold, color, True)
    return par


def rtl(par, space_after=4):
    ppr = par._p.get_or_add_pPr()
    ppr.insert(0, OxmlElement("w:bidi"))  # alignment stays "start" = right margin; jc="right" would mean the end
    par.paragraph_format.space_after = Pt(space_after)
    return par


def para(doc_or_cell, text, size=11, bold=False, color=None, ltr=False, space_after=4):
    p = doc_or_cell.add_paragraph()
    if ltr:
        p.paragraph_format.space_after = Pt(space_after)
        _rpr(p.add_run(text), size, bold, color, False)
        return p
    return add_text(rtl(p, space_after), text, size, bold, color)


def heading(doc, text):
    p = para(doc, text, size=14, bold=True, color="1F4E79", space_after=3)
    p.paragraph_format.space_before = Pt(10)
    p.paragraph_format.keep_with_next = True
    return p


def bullets(doc, items, numbered=False):
    for i, t in enumerate(items, 1):
        para(doc, (f"{i}. " if numbered else "• ") + t, space_after=2)


def shade(cell, hex_color):
    tcpr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_color)
    tcpr.append(shd)


def table(doc, header, rows, widths, rtl_table=True, ltr_cols=(), size=9.5, fills=None, bold_first=False):
    t = doc.add_table(rows=1, cols=len(header))
    t.style = "Table Grid"
    t.autofit = False
    layout = OxmlElement("w:tblLayout")
    layout.set(qn("w:type"), "fixed")
    t._tbl.tblPr.append(layout)
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    if rtl_table:
        t._tbl.tblPr.append(OxmlElement("w:bidiVisual"))
    for j, h in enumerate(header):
        c = t.rows[0].cells[j]
        c.paragraphs[0]._p.getparent().remove(c.paragraphs[0]._p)
        para(c, h, size=size, bold=True, ltr=j in ltr_cols or not rtl_table, space_after=0)
        shade(c, "D9E2F3")
    for r in rows:
        cells = t.add_row().cells
        for j, v in enumerate(r):
            c = cells[j]
            c.paragraphs[0]._p.getparent().remove(c.paragraphs[0]._p)
            para(c, str(v), size=size, ltr=j in ltr_cols or not rtl_table, space_after=0,
                 bold=(j == 0 and bold_first))
            if fills and fills[j]:
                shade(c, fills[j])
    for j, w in enumerate(widths):
        t.columns[j].width = Cm(w)
    for row in t.rows:
        for j, w in enumerate(widths):
            row.cells[j].width = Cm(w)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    return t


def box(doc, title, items):
    t = doc.add_table(rows=1, cols=1)
    t.style = "Table Grid"
    t._tbl.tblPr.append(OxmlElement("w:bidiVisual"))
    c = t.rows[0].cells[0]
    shade(c, "E2EFDA")
    c.paragraphs[0]._p.getparent().remove(c.paragraphs[0]._p)
    para(c, title, size=12, bold=True, space_after=3)
    for it in items:
        para(c, "• " + it, size=10.5, space_after=2)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)


def build_guide(path_docx, author, deadline=None):
    doc = Document()
    sec = doc.sections[0]
    sec.page_height, sec.page_width = Cm(29.7), Cm(21.0)
    sec.left_margin = sec.right_margin = Cm(1.8)
    sec.top_margin = sec.bottom_margin = Cm(1.6)
    doc.styles["Normal"].font.name = FONT
    p = para(doc, "دليل مراجعة جُمل تقارير الأشعة", size=20, bold=True, color="1F4E79", space_after=2)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p = para(doc, "يُرجى قراءة هذا الدليل أولاً (حوالي 10 دقائق)", size=11, color="595959", space_after=8)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    box(doc, "باختصار", SUMMARY + ([DEADLINE_LINE.format(deadline=deadline)] if deadline else []))

    heading(doc, "1. ما هذا المشروع؟")
    para(doc, ABOUT)
    para(doc, NO_IMAGES, bold=True)

    heading(doc, "2. الملفات")
    table(doc, ["الملف", "عدد الأسطر", "الوقت", "المحتوى"], FILES, [7.6, 1.5, 2.0, 6.3], ltr_cols=(0,))
    para(doc, "حرف X في اسم الملف هو A أو B حسب المراجِع.", size=9.5, color="595959")

    heading(doc, "3. خطوات العمل")
    bullets(doc, STEPS, numbered=True)

    heading(doc, "4. كيف يبدو السطر في الورقة Review")
    para(doc, "مثال مخترع للتوضيح، بنفس ترتيب الأعمدة في Excel (من اليسار إلى اليمين):", size=10)
    table(doc, ["A Item", "B Original report (correct)", "C Sentence BEFORE the change",
                "D Sentence AFTER the change", "E Error present?", "F Error type", "G Clinical significance",
                "H Comments"],
          [["S000", "Findings: Heart size is normal. No pleural effusion or pneumothorax. Impression: No acute "
                    "cardiopulmonary process.", "No pleural effusion or pneumothorax.",
            "Small left pleural effusion. No pneumothorax.", "Yes", "False prediction", "Major", "effusion added"]],
          [1.1, 3.8, 2.4, 2.6, 1.7, 1.9, 2.3, 1.6], rtl_table=False, size=8.5,
          fills=[GREY, GREY, GREY, GREY, YELLOW, YELLOW, YELLOW, YELLOW])
    para(doc, "عناوين الأعمدة في الملف مكتوبة بالإنجليزية والعربية:", size=10)
    table(doc, ["العمود", "العنوان بالعربية", "المعنى"], COLUMNS, [1.4, 4.6, 11.4], ltr_cols=(0,))

    heading(doc, "5. الأسئلة الثلاثة لكل سطر")
    para(doc, "السؤال 1 — هل في الجملة المعدّلة خطأ؟ (العمود E)", bold=True, color="1F4E79")
    table(doc, ["الاختيار", "متى"], Q1, [2.2, 15.2], ltr_cols=(0,), bold_first=True)
    para(doc, NEW_SENTENCE, size=10.5)
    para(doc, "السؤال 2 — ما نوع الخطأ؟ (العمود F) — فقط إذا كانت الإجابة Yes", bold=True, color="1F4E79")
    table(doc, ["الاختيار في القائمة", "المعنى", "مثال (الأصلية ← المعدّلة)"], TYPES, [4.2, 6.4, 6.8],
          ltr_cols=(0, 2), size=9, bold_first=True)
    para(doc, "السؤال 3 — ما أهميته السريرية؟ (العمود G) — فقط إذا كانت الإجابة Yes", bold=True, color="1F4E79")
    table(doc, ["الاختيار", "المعنى"], SEVERITY, [2.2, 15.2], ltr_cols=(0,), bold_first=True)
    para(doc, "العمود H (Comments): اختياري، مثلاً لسبب اختيار Unsure أو لذكر خطأ ثانٍ.", size=10.5)
    para(doc, "الملف 2 فقط — العمود I: هل يمكن ملاحظة الخطأ دون التقرير الأصلي؟ (فقط إذا كانت الإجابة Yes في E)", bold=True,
         color="1F4E79")
    table(doc, ["الاختيار", "المعنى"], COL_I, [2.2, 15.2], ltr_cols=(0,), bold_first=True)

    heading(doc, "6. أمثلة محلولة")
    para(doc, "الأمثلة مخترعة للتوضيح وليست من الملفات.", size=10, color="595959")
    table(doc, ["الجملة الأصلية", "الجملة المعدّلة", "E", "F", "G", "لماذا"], EXAMPLES, [3.8, 3.7, 1.5, 2.9, 1.4, 4.1],
          ltr_cols=(0, 1, 2, 3, 4), size=8.5)

    heading(doc, "7. قواعد مهمة")
    bullets(doc, RULES)
    heading(doc, "8. عند الانتهاء")
    bullets(doc, DONE, numbered=True)
    heading(doc, "9. أسئلة شائعة")
    for q, a in FAQ:
        para(doc, q, bold=True, size=10.5, space_after=0)
        para(doc, a, size=10.5, space_after=5)
    para(doc, "شكراً جزيلاً لوقتك ومساعدتك.", bold=True, color="1F4E79").alignment = WD_ALIGN_PARAGRAPH.CENTER

    cp = doc.core_properties
    cp.author = cp.last_modified_by = author
    cp.title = "دليل مراجعة جُمل تقارير الأشعة"
    cp.comments = cp.subject = cp.keywords = cp.category = ""
    cp.created = cp.modified = datetime.datetime.now()
    cp.revision = 1
    doc.save(path_docx)


# ---------------------------------------------------------------- workbooks
def reader_info_sheet(wb):
    ws = wb.create_sheet("Reader info")
    ws["A1"], ws["A1"].font = "Reader information / بيانات المراجِع", Font(bold=True, size=13)
    ws["A2"] = ("Please fill in the yellow cells after finishing. The paper reports specialty and experience only; your "
                "name is used only if you agree. / يُرجى تعبئة الخلايا الصفراء بعد الانتهاء. يذكر البحث التخصص والخبرة "
                "فقط، ولا يُذكر الاسم إلا بموافقتك.")
    ws["A2"].alignment = Alignment(wrap_text=True, vertical="top")
    ws.merge_cells("A2:B2")
    ws.row_dimensions[2].height = 48
    fields = [("Name / الاسم", None),
              ("Specialty / التخصص", '"Radiologist (consultant/specialist),Radiology resident,Other physician,Other"'),
              ("Years reading chest X-rays / سنوات الخبرة في قراءة صور الصدر", None),
              ("Workplace (optional) / مكان العمل (اختياري)", None),
              ("Date finished / تاريخ الانتهاء", None),
              ("May we name you in the acknowledgments? / هل توافق على ذكر اسمك في الشكر؟", '"Yes,No"')]
    for i, (label, options) in enumerate(fields, start=4):
        ws.cell(i, 1, label).alignment = Alignment(wrap_text=True, vertical="center")
        c = ws.cell(i, 2)
        c.fill = PatternFill("solid", fgColor=YELLOW)
        ws.row_dimensions[i].height = 30
        if options:
            dv = DataValidation(type="list", formula1=options, allow_blank=True)
            ws.add_data_validation(dv)
            dv.add(c)
    ws.column_dimensions["A"].width, ws.column_dimensions["B"].width = 62, 40


HEADERS = {  # column -> header in the reader copies (English as in the instructions, then Arabic)
    "A": "Item\nرقم",
    "B": "Original report (correct, for comparison)\nالتقرير الأصلي (صحيح، للمقارنة فقط)",
    "C": "Sentence BEFORE the change\nالجملة قبل التعديل",
    "D": "Sentence AFTER the change: judge this one\nالجملة بعد التعديل: احكم عليها",
    "E": "Error present?\nهل فيها خطأ؟",
    "F": "Error type (if Yes)\nنوع الخطأ (إذا Yes)",
    "G": "Clinical significance (if Yes)\nالأهمية (إذا Yes)",
    "H": "Comments (optional)\nملاحظات (اختياري)",
    "I": "Recognizable without original? (if Yes)\nيُلاحَظ دون التقرير الأصلي؟ (إذا Yes)",
}


def clear_headers(ws):
    """Self-explanatory bilingual headers; no filter buttons, so the rows cannot be sorted by accident."""
    ws.auto_filter.ref = None
    for col in range(1, ws.max_column + 1):
        c = ws.cell(1, col)
        letter = c.column_letter
        if letter in HEADERS and c.value:
            c.value = HEADERS[letter]
            c.alignment = Alignment(wrap_text=True, vertical="center", horizontal="center")
            c.font = Font(bold=True, size=10)
    ws.row_dimensions[1].height = 60
    for letter, width in (("A", 8), ("E", 14), ("G", 17), ("I", 19)):
        ws.column_dimensions[letter].width = max(ws.column_dimensions[letter].width or 0, width)


def workbook(src, dst, author, add_info):
    wb = openpyxl.load_workbook(src, rich_text=True)  # keeps the bold red differences in column D
    clear_headers(wb["Review"])
    if add_info and "Reader info" not in wb.sheetnames:
        reader_info_sheet(wb)
    wb.active = wb.sheetnames.index("Instructions")
    p = wb.properties
    p.creator = p.lastModifiedBy = author
    p.title, p.description, p.keywords, p.subject = "Radiology report audit", None, None, None
    p.created = p.modified = datetime.datetime.now()
    wb.save(dst)


def to_pdf(path_docx, out_dir):
    subprocess.run(["soffice", "--headless", "--convert-to", "pdf", "--outdir", out_dir, path_docx], check=True,
                   capture_output=True, timeout=240)


MESSAGE = """السلام عليكم دكتور/ة،
شكراً لموافقتك على المساعدة في البحث. أرسلت لك ثلاثة ملفات:
0) دليل قصير (PDF) — يُرجى قراءته أولاً، حوالي 10 دقائق.
1) الملف الأول: 250 سطراً (حوالي 3–4 ساعات، ويمكن تقسيمها على عدة جلسات).
2) الملف الثاني: 63 سطراً (حوالي ساعة).
في كل سطر تقرير أشعة صدر أصلي وجملة معدّلة منه، والمطلوب اختيار: هل فيها خطأ، نوعه، وأهميته، من قوائم جاهزة داخل الملف.
يُرجى العمل بشكل مستقل وعدم مناقشة الأسطر مع أحد حتى الانتهاء، وتعبئة ورقة Reader info في الملف الأول.
يُرجى إعادة الملفين لي قبل {deadline} (يكفي حوالي ساعة يومياً)، ولأي سؤال أنا موجود.
شكراً جزيلاً.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--author", default="Ahmed Alsaqqa")
    ap.add_argument("--deadline", default="[التاريخ]", help="return date as it should appear in the guide and message")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    guide_docx = os.path.join(args.out_dir, GUIDE + ".docx")
    build_guide(guide_docx, args.author, args.deadline)
    to_pdf(guide_docx, args.out_dir)
    guide_pdf = os.path.join(args.out_dir, GUIDE + ".pdf")
    for reader in ("A", "B"):
        d = os.path.join(args.out_dir, f"Reader_{reader}")
        os.makedirs(d, exist_ok=True)
        workbook(os.path.join(args.audit_dir, f"ReXErr_label_audit_Reader_{reader}.xlsx"),
                 os.path.join(d, MAIN.format(reader)), args.author, add_info=True)
        workbook(os.path.join(args.audit_dir, f"ReXErr_rewording_audit_Reader_{reader}.xlsx"),
                 os.path.join(d, SHORT.format(reader)), args.author, add_info=False)
        shutil.copy(guide_pdf, os.path.join(d, GUIDE + ".pdf"))
        with zipfile.ZipFile(os.path.join(args.out_dir, f"Reader_{reader}.zip"), "w", zipfile.ZIP_DEFLATED) as z:
            for f in sorted(os.listdir(d)):
                z.write(os.path.join(d, f), f)
    with open(os.path.join(args.out_dir, "Message_to_reader.txt"), "w", encoding="utf-8") as f:
        f.write(MESSAGE.format(deadline=args.deadline))
    print("wrote", sorted(os.listdir(args.out_dir)))


if __name__ == "__main__":
    main()
