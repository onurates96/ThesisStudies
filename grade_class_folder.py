"""
Class Diagram Grading Runner (Folder-based)

Folder layout expected (per case):
thesis/benchmark_results/Class/case_0/
  ground_truth.md              -> expected Mermaid
  input_prompt.txt             -> ignored
  actual_*.md (or any .md)     -> treated as model outputs (except ground_truth.md)
  ... any other files

Output (per case folder):
  grading_results.csv          -> one CSV per case (one row per model output file)

Output (Class folder):
  grading_overall.csv          -> aggregated rows across all cases
  grading_model_summary.csv    -> mean/std/min/max per model output file name
"""

import os
import glob
import re
import json
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional
import pandas as pd

# ============================================================
# 1) Mermaid classDiagram parser (tolerant)
# ============================================================

@dataclass
class ClassIR:
    name: str
    attributes: List[str] = field(default_factory=list)
    operations: List[str] = field(default_factory=list)
    stereotypes: Set[str] = field(default_factory=set)

@dataclass
class EnumIR:
    name: str
    literals: List[str] = field(default_factory=list)

@dataclass
class RelationIR:
    src: str
    dst: str
    rel: str
    src_mult: Optional[str] = None
    dst_mult: Optional[str] = None
    label: Optional[str] = None

@dataclass
class ParseResult:
    diagram_type: str
    classes: Dict[str, ClassIR] = field(default_factory=dict)
    enums: Dict[str, EnumIR] = field(default_factory=dict)
    relations: List[RelationIR] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

REL_MAP = {
    "<|--": "inheritance",
    "--|>": "inheritance",
    "<|..": "realization",
    "..|>": "realization",
    "*--": "composition",
    "--*": "composition",
    "o--": "aggregation",
    "--o": "aggregation",
    "..>": "dependency",
    "<..": "dependency",
    "-->": "association",
    "<--": "association",
    "--":  "association",
    "..":  "dependency",
}

RE_HEADER = re.compile(r"^\s*classdiagram\b", re.IGNORECASE | re.MULTILINE)
RE_CLASS_BLOCK = re.compile(
    r"(?is)\bclass\s+([A-Za-z_]\w*)\s*\{(.*?)\n\s*\}",
    re.IGNORECASE
)
RE_CLASS_DECL = re.compile(r"^\s*class\s+([A-Za-z_]\w*)\s*$", re.IGNORECASE)
RE_MEMBER_LINE = re.compile(r"^\s*([A-Za-z_]\w*)\s*:\s*(.+?)\s*$")
RE_REL_LINE = re.compile(
    r"""^\s*
        ([A-Za-z_]\w*)
        (?:\s*"([^"]+)"\s*)?
        \s*([.<>\-\*o|]{2,5})\s*    
        (?:\s*"([^"]+)"\s*)?
        ([A-Za-z_]\w*)
        (?:\s*:\s*(.+?))?
        \s*$""",
    re.VERBOSE
)
RE_STEREOTYPE = re.compile(r"<<\s*([A-Za-z_]\w*)\s*>>")

def _strip_comments(code: str) -> str:
    lines = []
    for line in code.splitlines():
        if line.strip().startswith("%%") and not line.strip().startswith("%%{"):
            continue
        if "%%" in line and not "%%{" in line:
            line = line.split("%%", 1)[0]
        lines.append(line)
    return "\n".join(lines)

def strip_invisible_chars(s: str) -> str:
    """
    Removes common invisible Unicode chars that break regex matching.
    """
    if not s:
        return ""
    invis = [
        "\ufeff",  # BOM
        "\u200b",  # zero-width space
        "\u200c",  # zero-width non-joiner
        "\u200d",  # zero-width joiner
        "\u2060",  # word joiner
        "\u200e",  # LTR mark
        "\u200f",  # RTL mark
    ]
    for ch in invis:
        s = s.replace(ch, "")
    return s

def _normalize_ws(code: str) -> str:
    return str(code).replace("\r\n", "\n").replace("\r", "\n")

def _clean_ident(x: str) -> str:
    return x.strip().strip('"').strip("'").strip("`")

def _ensure_class(res: ParseResult, name: str) -> ClassIR:
    if name not in res.classes:
        res.classes[name] = ClassIR(name=name)
    return res.classes[name]

def _ensure_enum(res: ParseResult, name: str) -> EnumIR:
    if name not in res.enums:
        res.enums[name] = EnumIR(name=name)
    return res.enums[name]

def count_expected_members(exp_pr: ParseResult) -> int:
    """Counts expected attributes + operations across all expected classes."""
    total = 0
    for _, c in exp_pr.classes.items():
        total += len([x for x in c.attributes if (x or "").strip()])
        total += len([x for x in c.operations if (x or "").strip()])
    return total

def extract_mermaid_block(text: str) -> str:
    t = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    # Strictly capture content BETWEEN fences
    m = re.search(r"```mermaid\s*\n(.*?)\n```", t, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()

    # Otherwise, try any fenced block, pick one containing classDiagram
    blocks = re.findall(r"```(?:\w+)?\s*\n(.*?)\n```", t, flags=re.DOTALL)
    for b in blocks:
        if "classdiagram" in b.lower():
            return b.strip()

    # Finally, slice from classDiagram line
    lines = t.splitlines()
    for i, line in enumerate(lines):
        if "classdiagram" in line.lower():
            return "\n".join(lines[i:]).strip()

    return t.strip()

def parse_mermaid_class_diagram(code: str) -> ParseResult:
    code = (code or "").lstrip("\ufeff")
    code = strip_invisible_chars(code)
    code = _normalize_ws(code)
    code = _strip_comments(code)
    code = fix_common_llm_mermaid_typos(code) 

    res = ParseResult(diagram_type="Class")
    if not RE_HEADER.search(code):
        res.warnings.append("No 'classDiagram' header found (still attempting to parse as class diagram).")

    # Pass 1: class blocks
    blocks = []
    for m in RE_CLASS_BLOCK.finditer(code):
        cls_name = _clean_ident(m.group(1))
        body = m.group(2) or ""
        blocks.append((m.start(), m.end(), cls_name, body))

        cls_ir = _ensure_class(res, cls_name)

        for st in RE_STEREOTYPE.findall(body):
            cls_ir.stereotypes.add(st.lower())

        for raw_line in body.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if RE_STEREOTYPE.search(line):
                continue
            is_op = "(" in line and ")" in line
            if is_op:
                cls_ir.operations.append(line)
            else:
                cls_ir.attributes.append(line)

    # Remove class blocks from code
    if blocks:
        parts = []
        last = 0
        for start, end, _, _ in sorted(blocks, key=lambda x: x[0]):
            parts.append(code[last:start])
            last = end
        parts.append(code[last:])
        code_wo_blocks = "\n".join(parts)
    else:
        code_wo_blocks = code

    # Pass 2: line-by-line
    for raw_line in code_wo_blocks.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        low = line.lower()

        if low.startswith("classdiagram"):
            continue

        m_decl = RE_CLASS_DECL.match(line)
        if m_decl:
            _ensure_class(res, _clean_ident(m_decl.group(1)))
            continue

        m_mem = RE_MEMBER_LINE.match(line)
        if m_mem:
            cls_name = _clean_ident(m_mem.group(1))
            member = m_mem.group(2).strip()

            cls_ir = _ensure_class(res, cls_name)
            for st in RE_STEREOTYPE.findall(member):
                cls_ir.stereotypes.add(st.lower())

            # Enum entry as member line heuristic
            if "enumeration" in cls_ir.stereotypes:
                lits = re.split(r"[,\s]+", member.replace("{", " ").replace("}", " "))
                lits = [t.strip() for t in lits if t.strip()]
                if "(" not in member and lits:
                    enum_ir = _ensure_enum(res, cls_name)
                    for lit in lits:
                        if lit.lower() not in [x.lower() for x in enum_ir.literals]:
                            enum_ir.literals.append(lit)
                    continue

            is_op = "(" in member and ")" in member
            if is_op:
                cls_ir.operations.append(member)
            else:
                cls_ir.attributes.append(member)
            continue

        m_rel = RE_REL_LINE.match(line)
        if m_rel:
            src = _clean_ident(m_rel.group(1))
            src_mult = m_rel.group(2)
            token = (m_rel.group(3) or "").strip()
            dst_mult = m_rel.group(4)
            dst = _clean_ident(m_rel.group(5))
            label = (m_rel.group(6) or "").strip() or None

            rel_type = REL_MAP.get(token)
            if not rel_type:
                tok2 = token.replace("-.","..").replace(".-","..")
                rel_type = REL_MAP.get(tok2, "association")

            _ensure_class(res, src)
            _ensure_class(res, dst)

            res.relations.append(RelationIR(
                src=src, dst=dst, rel=rel_type,
                src_mult=src_mult, dst_mult=dst_mult, label=label
            ))
            continue

        # LLMs sometimes output "enum X { ... }"
        if low.startswith("enum "):
            m_enum = re.match(r"^\s*enum\s+([A-Za-z_]\w*)\s*(\{.*\})?\s*$", line, re.IGNORECASE)
            if m_enum:
                ename = _clean_ident(m_enum.group(1))
                enum_ir = _ensure_enum(res, ename)
                _ensure_class(res, ename).stereotypes.add("enumeration")
                body = m_enum.group(2)
                if body:
                    inside = body.strip().lstrip("{").rstrip("}")
                    lits = [t.strip() for t in re.split(r"[,\s]+", inside) if t.strip()]
                    for lit in lits:
                        if lit.lower() not in [x.lower() for x in enum_ir.literals]:
                            enum_ir.literals.append(lit)
                continue

        if not (low.startswith("direction") or low.startswith("linkstyle") or low.startswith("style")):
            res.warnings.append(f"Unparsed line: {line}")

    # Post: enum stereotypes but empty enum list
    for cname, cir in res.classes.items():
        if "enumeration" in cir.stereotypes and cname not in res.enums:
            res.enums[cname] = EnumIR(name=cname)

    return res

def fix_common_llm_mermaid_typos(code: str) -> str:
    """
    Fix common Mermaid formatting issues without breaking 'classDiagram'.
    - 'classCustomer' -> 'class Customer'
    - 'Class Restaurant' -> 'class Restaurant'
    - 'class Account{' -> 'class Account {'
    """
    c = (code or "").lstrip("\ufeff")  # remove BOM if any

    # Normalize keyword 'Class' to 'class' at line start
    c = re.sub(r"(?im)^\s*Class\s+", "class ", c)

    # Fix missing space after 'class' keyword but do NOT touch 'classDiagram'
    c = re.sub(r"(?im)^\s*class(?!diagram\b)([A-Za-z_]\w+)\b", r"class \1", c)

    # Normalize 'class Name{' -> 'class Name {'
    c = re.sub(r"(?im)^\s*class\s+([A-Za-z_]\w*)\s*\{", r"class \1 {", c)

    return c

# ============================================================
# 2) Algorithm 1: Compare Classes
# ============================================================

_CAMEL_SPLIT = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")

def normalize_name(name: str) -> str:
    name = (name or "").strip()
    name = name.replace("_", " ").replace("-", " ")
    name = _CAMEL_SPLIT.sub(" ", name)
    name = re.sub(r"\s+", " ", name).strip().lower()
    return name

def levenshtein(a: str, b: str) -> int:
    a, b = a or "", b or ""
    if a == b: return 0
    if not a: return len(b)
    if not b: return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            ins = cur[j-1] + 1
            dele = prev[j] + 1
            sub = prev[j-1] + (0 if ca == cb else 1)
            cur.append(min(ins, dele, sub))
        prev = cur
    return prev[-1]

def syntactic_match(cs_name: str, ci_name: str, ratio: float = 0.4) -> bool:
    a = normalize_name(cs_name)
    b = normalize_name(ci_name)
    if not a or not b: return False
    d = levenshtein(a, b)
    max_len = max(len(a), len(b))
    return d <= ratio * max_len

def syntactic_score(cs_name: str, ci_name: str) -> float:
    a = normalize_name(cs_name)
    b = normalize_name(ci_name)
    if not a or not b: return 0.0
    d = levenshtein(a, b)
    max_len = max(len(a), len(b))
    return 1.0 - (d / max_len) if max_len else 0.0

def semantic_score(cs_name: str, ci_name: str) -> float:
    # Semantic match hook (WordNet/WS4J equivalent can be added later).
    return 0.0

def semantic_match(cs_name: str, ci_name: str, threshold: float = 0.75) -> bool:
    return semantic_score(cs_name, ci_name) >= threshold

def _norm_member_name(m: str) -> str:
    s = (m or "").strip()
    s = re.sub(r"^[\+\-\#~]\s*", "", s).strip()

    # If nothing remains after stripping visibility, return empty key
    if not s:
        return ""

    if "(" in s:
        name = s.split("(", 1)[0].strip()
    else:
        parts = s.split()

        # ✅ Guard against empty split results
        if not parts:
            return ""

        if len(parts) == 1:
            name = parts[0]
        else:
            if parts[0].lower() in {"int","integer","string","bool","boolean","float","double","long","date","datetime","uuid"}:
                # Also guard index
                name = parts[1] if len(parts) > 1 else parts[0]
            else:
                name = parts[0]

    name = re.sub(r"\W+", "", name).lower()
    return name

def props_of_class(cls_ir: ClassIR) -> Set[str]:
    out = set()
    for x in (cls_ir.attributes + cls_ir.operations):
        if not (x or "").strip():
            continue
        k = _norm_member_name(x)
        if k:          # ✅ ignore empty keys
            out.add(k)
    return out

def content_similarity(ci_cls: ClassIR, cs_cls: ClassIR) -> float:
    ci_set = props_of_class(ci_cls)
    cs_set = props_of_class(cs_cls)
    if not ci_set and not cs_set: return 1.0
    if not ci_set or not cs_set: return 0.0
    return len(ci_set & cs_set) / len(ci_set | cs_set)

def content_match(ci_cls: ClassIR, cs_cls: ClassIR, threshold: float = 0.30) -> bool:
    return content_similarity(ci_cls, cs_cls) >= threshold

def association_ends(pr: ParseResult, class_name: str) -> Set[Tuple[str, str]]:
    ends = set()
    for r in pr.relations:
        if r.src == class_name:
            ends.add((normalize_name(r.dst), r.rel))
        elif r.dst == class_name:
            ends.add((normalize_name(r.src), r.rel))
    return ends

def assoc_match(list_s: Set[Tuple[str,str]], list_i: Set[Tuple[str,str]], threshold: float = 0.40) -> bool:
    if not list_s and not list_i:
        return True
    if not list_s or not list_i:
        return False
    j = len(list_s & list_i) / len(list_s | list_i)
    return j >= threshold

@dataclass
class Candidate:
    ci: str
    cs: str
    name_score: float
    content_score: float
    total_score: float

def grade_candidate(ci_name: str, ci_cls: ClassIR,
                    cs_name: str, cs_cls: ClassIR,
                    w_name: float = 0.5, w_content: float = 0.5) -> Candidate:
    ns = max(syntactic_score(cs_name, ci_name), semantic_score(cs_name, ci_name))
    cs = content_similarity(ci_cls, cs_cls)
    total = w_name * ns + w_content * cs
    return Candidate(ci=ci_name, cs=cs_name, name_score=ns, content_score=cs, total_score=total)

def compare_class_algorithm1(instructor_pr: ParseResult, student_pr: ParseResult,
                             synt_ratio: float = 0.4,
                             semantic_th: float = 0.75,
                             content_th: float = 0.30,
                             assoc_th: float = 0.40,
                             w_name: float = 0.5,
                             w_content: float = 0.5) -> Tuple[Dict[str,str], List[str]]:
    instList = list(instructor_pr.classes.keys())
    studList = list(student_pr.classes.keys())

    possible: Dict[str, List[Candidate]] = {ci: [] for ci in instList}

    for ci in instList:
        ci_cls = instructor_pr.classes[ci]
        for cs in studList:
            cs_cls = student_pr.classes[cs]

            if (syntactic_match(cs, ci, synt_ratio)
                or semantic_match(cs, ci, semantic_th)
                or content_match(ci_cls, cs_cls, content_th)):
                possible[ci].append(grade_candidate(ci, ci_cls, cs, cs_cls, w_name, w_content))

        possible[ci].sort(key=lambda x: x.total_score, reverse=True)

    classMatchMap: Dict[str, str] = {}
    used_student: Set[str] = set()
    missClassList: List[str] = []

    for ci in instList:
        cand_list = possible.get(ci, [])
        chosen = None
        for c in cand_list:
            if c.cs not in used_student:
                chosen = c
                break
        if chosen:
            classMatchMap[ci] = chosen.cs
            used_student.add(chosen.cs)
        else:
            missClassList.append(ci)

    # Association-based salvage matching for missed classes
    for ci in list(missClassList):
        list_i = association_ends(instructor_pr, ci)
        best_cs = None
        for cs in studList:
            if cs in used_student:
                continue
            list_s = association_ends(student_pr, cs)
            if assoc_match(list_s, list_i, assoc_th):
                best_cs = cs
                break
        if best_cs:
            classMatchMap[ci] = best_cs
            used_student.add(best_cs)
            missClassList.remove(ci)

    return classMatchMap, missClassList


# ============================================================
# 3) Algorithm 3: Split detection
# ============================================================

def _norm_mult(m: Optional[str]) -> Optional[str]:
    if m is None:
        return None
    return m.strip().lower().replace(" ", "")

def _is_many(m: Optional[str]) -> bool:
    s = _norm_mult(m)
    if s is None:
        return False
    return s in {"*", "0..*", "1..*", "many"} or s.endswith("..*")

def _is_one(m: Optional[str]) -> bool:
    s = _norm_mult(m)
    if s is None:
        return False
    return s in {"1", "0..1", "1..1"}

def has_one_to_many_relation(student_pr: ParseResult, a: str, b: str) -> bool:
    for r in student_pr.relations:
        if r.src == a and r.dst == b:
            if (_is_one(r.src_mult) and _is_many(r.dst_mult)) or (_is_many(r.src_mult) and _is_one(r.dst_mult)):
                return True
        if r.src == b and r.dst == a:
            if (_is_one(r.src_mult) and _is_many(r.dst_mult)) or (_is_many(r.src_mult) and _is_one(r.dst_mult)):
                return True
    return False

def class_split_match_algorithm3(instructor_pr: ParseResult, student_pr: ParseResult,
                                 classMatchMap: Dict[str,str],
                                 coverage_th: float = 0.70,
                                 require_one_to_many: bool = True) -> Dict[str, Tuple[str,str]]:
    instList = list(instructor_pr.classes.keys())
    studList = list(student_pr.classes.keys())
    splitClassMap: Dict[str, Tuple[str,str]] = {}

    # Try all unordered pairs (Cs0, Cs1)
    for i in range(len(studList)):
        for j in range(i+1, len(studList)):
            cs0 = studList[i]
            cs1 = studList[j]

            if require_one_to_many and not has_one_to_many_relation(student_pr, cs0, cs1):
                continue

            cs0_props = props_of_class(student_pr.classes[cs0])
            cs1_props = props_of_class(student_pr.classes[cs1])
            if not cs0_props and not cs1_props:
                continue

            for ci in instList:
                ci_props = props_of_class(instructor_pr.classes[ci])
                if not ci_props:
                    continue
                overlap0 = len(ci_props & cs0_props)
                overlap1 = len(ci_props & cs1_props)
                coverage = len(ci_props & (cs0_props | cs1_props)) / len(ci_props)

                if coverage >= coverage_th and overlap0 > 0 and overlap1 > 0:
                    splitClassMap[ci] = (cs0, cs1)
                    break

    return splitClassMap


# ============================================================
# 4) Algorithm 4: Merge detection
# ============================================================

def has_association_between(instructor_pr: ParseResult, ci1: str, ci2: str) -> bool:
    for r in instructor_pr.relations:
        if (r.src == ci1 and r.dst == ci2) or (r.src == ci2 and r.dst == ci1):
            return True
    return False

def content_misplaced_into(ci1_props: Set[str], cs_props: Set[str], misplaced_th: float = 0.70) -> bool:
    if not ci1_props:
        return False
    coverage = len(ci1_props & cs_props) / len(ci1_props)
    return coverage >= misplaced_th

def class_merge_match_algorithm4(instructor_pr: ParseResult, student_pr: ParseResult,
                                 classMatchMap: Dict[str,str],
                                 misplaced_th: float = 0.70) -> Dict[str, Tuple[str,str]]:
    # Invert expected->student map to student->expected (unique expected per student)
    inv_map: Dict[str, str] = {}
    for ci, cs in classMatchMap.items():
        inv_map[cs] = ci

    inst_classes = list(instructor_pr.classes.keys())
    mergeClassMap: Dict[str, Tuple[str,str]] = {}

    for cs, ci2 in inv_map.items():
        if cs not in student_pr.classes:
            continue

        cs_props = props_of_class(student_pr.classes[cs])

        candidates_ci1: List[str] = []
        for ci1 in inst_classes:
            if ci1 == ci2:
                continue
            ci1_props = props_of_class(instructor_pr.classes[ci1])
            if content_misplaced_into(ci1_props, cs_props, misplaced_th):
                candidates_ci1.append(ci1)

        chosen_ci1 = None
        for ci1 in candidates_ci1:
            if has_association_between(instructor_pr, ci1, ci2):
                chosen_ci1 = ci1
                break

        if chosen_ci1:
            mergeClassMap[cs] = (chosen_ci1, ci2)

    return mergeClassMap


# ============================================================
# 5) Mark-based scoring with detailed summary fields
# Rules:
# - Misplaced attribute/operation -> half mark
# - Derived association -> half mark
# - Missing element -> zero
# - Enum entry as attribute/class -> average mark (1/N)
# ============================================================

def _member_set(cls_ir: ClassIR) -> Set[str]:
    return {_norm_member_name(x) for x in (cls_ir.attributes + cls_ir.operations) if (x or "").strip()}

def score_members_with_details(exp_pr: ParseResult, act_pr: ParseResult, class_map: Dict[str, str],
                               full_mark: float = 1.0,
                               misplaced_ratio: float = 0.5,
                               top_k: int = 5) -> Tuple[float, float, Dict[str, object]]:
    # Global inverted index: member_name -> student classes containing it
    member_index: Dict[str, Set[str]] = {}
    for s_name, s_cls in act_pr.classes.items():
        for mn in _member_set(s_cls):
            member_index.setdefault(mn, set()).add(s_name)

    earned = 0.0
    total = 0.0
    correct = misplaced = missing = 0
    missing_list: List[str] = []

    for exp_c, exp_cls in exp_pr.classes.items():
        exp_members = _member_set(exp_cls)
        if not exp_members:
            continue

        stud_c = class_map.get(exp_c)
        stud_members = _member_set(act_pr.classes[stud_c]) if stud_c and stud_c in act_pr.classes else set()

        for m in exp_members:
            total += full_mark
            if m in stud_members:
                earned += full_mark
                correct += 1
            else:
                found_in = member_index.get(m, set())
                if found_in:
                    earned += full_mark * misplaced_ratio
                    misplaced += 1
                else:
                    missing += 1
                    missing_list.append(m)

    return earned, total, {
        "correct": correct,
        "misplaced": misplaced,
        "missing": missing,
        "top_missing": ",".join(missing_list[:top_k])
    }

def score_relations_with_details(exp_pr: ParseResult, act_pr: ParseResult, class_map: Dict[str, str],
                                 full_mark: float = 1.0,
                                 derived_ratio: float = 0.5,
                                 top_k: int = 5) -> Tuple[float, float, Dict[str, object]]:
    # Index student relations by endpoints (direction-sensitive)
    stud_by_endpoints: Dict[Tuple[str, str], List[str]] = {}
    for r in act_pr.relations:
        stud_by_endpoints.setdefault((r.src, r.dst), []).append(r.rel)

    earned = 0.0
    total = 0.0
    exact = derived = missing = 0
    missing_list: List[str] = []

    for r in exp_pr.relations:
        if r.src not in class_map or r.dst not in class_map:
            continue
        src = class_map[r.src]
        dst = class_map[r.dst]
        total += full_mark

        rel_types = stud_by_endpoints.get((src, dst), [])
        if not rel_types:
            missing += 1
            missing_list.append(f"{src}->{dst}:{r.rel}")
            continue

        if r.rel in rel_types:
            earned += full_mark
            exact += 1
        else:
            earned += full_mark * derived_ratio
            derived += 1

    return earned, total, {
        "exact": exact,
        "derived": derived,
        "missing": missing,
        "top_missing": ",".join(missing_list[:top_k])
    }

def score_enums_with_details(exp_pr: ParseResult, act_pr: ParseResult, class_map: Dict[str, str],
                             enum_total_mark: float = 1.0,
                             top_k: int = 5) -> Tuple[float, float, Dict[str, object]]:
    # Student enum literals
    student_enum_entries: Set[Tuple[str, str]] = set()
    for ename, e in act_pr.enums.items():
        for lit in e.literals:
            student_enum_entries.add((ename, lit.strip().lower()))

    # Student class names for "class representing an enum entry"
    student_class_names = {c.lower() for c in act_pr.classes.keys()}

    # Student attributes per class for "attribute representing an enum entry"
    student_attr_names: Set[Tuple[str, str]] = set()
    for cname, c in act_pr.classes.items():
        for a in c.attributes:
            student_attr_names.add((cname, _norm_member_name(a)))

    earned = 0.0
    total = 0.0
    found = missing = 0
    missing_list: List[str] = []

    for ename, e in exp_pr.enums.items():
        mapped_enum = class_map.get(ename)
        if not mapped_enum:
            continue

        entries = [x.strip().lower() for x in e.literals if x.strip()]
        if not entries:
            continue

        total += enum_total_mark
        per_entry = enum_total_mark / len(entries)

        for ent in entries:
            if (mapped_enum, ent) in student_enum_entries:
                earned += per_entry
                found += 1
            elif (mapped_enum, ent) in student_attr_names:
                earned += per_entry
                found += 1
            elif ent in student_class_names:
                earned += per_entry
                found += 1
            else:
                missing += 1
                missing_list.append(f"{mapped_enum}.{ent}")

    return earned, total, {
        "found": found,
        "missing": missing,
        "top_missing": ",".join(missing_list[:top_k])
    }


# ============================================================
# 6) Folder-based runner: one CSV per case + overall summaries
# ============================================================

EXCLUDE_FILES = {"ground_truth.md", "input_prompt.txt"}

def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def grade_one_case(case_dir: str, csv_name: str = "grading_results.csv") -> pd.DataFrame:
    gt_path = os.path.join(case_dir, "ground_truth.md")
    if not os.path.exists(gt_path):
        print(f"[SKIP] Missing ground_truth.md: {case_dir}")
        return pd.DataFrame()

    expected_text = extract_mermaid_block(read_text(gt_path))
    exp_pr = parse_mermaid_class_diagram(expected_text)

    md_files = sorted(glob.glob(os.path.join(case_dir, "*.md")))
    actual_files = [p for p in md_files if os.path.basename(p) not in EXCLUDE_FILES]

    if not actual_files:
        print(f"[SKIP] No model outputs (.md) found: {case_dir}")
        return pd.DataFrame()

    rows = []
    case_name = os.path.basename(case_dir)

    for ap in actual_files:
        model_file = os.path.basename(ap)  # generic model id
        actual_text = extract_mermaid_block(read_text(ap))
        act_pr = parse_mermaid_class_diagram(actual_text)

        parse_ok = (len(act_pr.classes) > 0) or (len(act_pr.relations) > 0)

        # Algorithm 1
        class_map, miss = compare_class_algorithm1(exp_pr, act_pr)

        # Algorithm 3/4 (reported only)
        split_map = class_split_match_algorithm3(exp_pr, act_pr, class_map)
        merge_map = class_merge_match_algorithm4(exp_pr, act_pr, class_map)

        # Mark-based scoring + summary details
        mem_earned, mem_total, mem_info = score_members_with_details(exp_pr, act_pr, class_map)
        rel_earned, rel_total, rel_info = score_relations_with_details(exp_pr, act_pr, class_map)
        enum_earned, enum_total, enum_info = score_enums_with_details(exp_pr, act_pr, class_map)
        class_earned = len(class_map)
        class_max = len(exp_pr.classes)
        
        # --- Class points (2 points per expected class) ---
        EXPECTED_CLASS_POINTS = 2
        SPLIT_PENALTY = 1  # if split, deduct 1 point (so 2 -> 1)

        expected_class_count = len(exp_pr.classes)
        class_max = expected_class_count * EXPECTED_CLASS_POINTS

        # Base: 2 points for each matched expected class
        class_earned = len(class_map) * EXPECTED_CLASS_POINTS

        # If merge detected, count merged instructor classes as matched too (if they were not in class_map)
        merged_expected = set()
        for _, pair in merge_map.items():  # pair is (Ci1, Ci2)
            merged_expected.update(pair)

        # Add points for merged classes not already counted
        for ci in merged_expected:
            if ci not in class_map and ci in exp_pr.classes:
                class_earned += EXPECTED_CLASS_POINTS

        # Apply split penalty: for each split expected class, reduce earned by 1 (2 -> 1)
        # (Only if that class exists in expected model.)
        for ci in split_map.keys():
            if ci in exp_pr.classes:
                class_earned -= SPLIT_PENALTY

        # Guard: keep in [0, class_max]
        class_earned = max(0, min(class_earned, class_max))

        # --- Expected total points (classes + members + relations) ---
        expected_member_count = count_expected_members(exp_pr)
        expected_relation_count = len(exp_pr.relations)

        expected_points = (expected_class_count * EXPECTED_CLASS_POINTS) + expected_member_count + expected_relation_count

        # --- Achieved points ---
        achieved_points = class_earned + mem_earned + rel_earned

        percent_covered = (achieved_points / expected_points) if expected_points > 0 else 0.0

        # Note: percent_covered is based on expected points (classes+members+relations).
        # You can still keep enum separately in the CSV if you want.
        total_earned = achieved_points
        total_max = expected_points
        total_score = percent_covered

        rows.append({
            "case": case_name,
            "model_output_file": model_file,

            "total_score": total_score,
            "total_earned": total_earned,
            "total_max": total_max,
            
            "class_earned": class_earned,
            "class_max": class_max,
            
            "expected_points": expected_points,
            "achieved_points": achieved_points,
            "percent_covered": percent_covered,
            
            "expected_member_count": expected_member_count,
            "expected_relation_count": expected_relation_count,

            "member_earned": mem_earned,
            "member_max": mem_total,
            "member_correct_count": mem_info["correct"],
            "member_misplaced_count": mem_info["misplaced"],
            "member_missing_count": mem_info["missing"],
            "member_top_missing": mem_info["top_missing"],

            "relation_earned": rel_earned,
            "relation_max": rel_total,
            "rel_exact_count": rel_info["exact"],
            "rel_derived_count": rel_info["derived"],
            "rel_missing_count": rel_info["missing"],
            "rel_top_missing": rel_info["top_missing"],

            "enum_earned": enum_earned,
            "enum_max": enum_total,
            "enum_entry_found_count": enum_info["found"],
            "enum_entry_missing_count": enum_info["missing"],
            "enum_top_missing_entries": enum_info["top_missing"],

            "expected_class_count": len(exp_pr.classes),
            "actual_class_count": len(act_pr.classes),
            "matched_classes": len(class_map),
            "missing_expected_classes": len(miss),

            "split_found": len(split_map),
            "merge_found": len(merge_map),

            "parse_ok": parse_ok,
            "warnings_head": " | ".join(act_pr.warnings[:3]),
        })
        details_dir = os.path.join(case_dir, "details")
        os.makedirs(details_dir, exist_ok=True)

        detail_obj = {
            "case": case_name,
            "model_output_file": model_file,
            "parse_ok": parse_ok,
            "warnings": act_pr.warnings,

            "class_match_map": class_map,          # expected -> actual
            "missing_expected_classes": miss,      # list

            "split_map": {k: list(v) for k, v in split_map.items()},   # expected -> [cs0, cs1]
            "merge_map": {k: list(v) for k, v in merge_map.items()},   # student -> [ci1, ci2]

            "scores": {
                "class": {"earned": class_earned, "max": class_max},
                "members": {"earned": mem_earned, "max": mem_total},
                "relations": {"earned": rel_earned, "max": rel_total},
                "enums": {"earned": enum_earned, "max": enum_total},
                "total": {"earned": total_earned, "max": total_max, "score": total_score},
            }
        }

        detail_name = f"detail_{model_file.replace('.', '_')}.json"
        detail_path = os.path.join(details_dir, detail_name)

        with open(detail_path, "w", encoding="utf-8") as f:
            json.dump(detail_obj, f, ensure_ascii=False, indent=2)

    df = pd.DataFrame(rows).sort_values(by="total_score", ascending=False)
    out_path = os.path.join(case_dir, csv_name)
    df.to_csv(out_path, index=False, encoding="utf-8-sig", sep=";", decimal=",")
    print(f"[OK] Wrote: {out_path}")
    return df

def grade_all_cases_in_class_folder(root: str = r"benchmark_results\Class",
                                   csv_name: str = "grading_results.csv") -> pd.DataFrame:
    case_dirs = sorted([p for p in glob.glob(os.path.join(root, "case_*")) if os.path.isdir(p)])
    if not case_dirs:
        print(f"[ERROR] No case_* directories found under: {root}")
        return pd.DataFrame()

    all_rows = []
    for case_dir in case_dirs:
        df_case = grade_one_case(case_dir, csv_name=csv_name)
        if not df_case.empty:
            all_rows.append(df_case)

    if not all_rows:
        return pd.DataFrame()

    agg = pd.concat(all_rows, ignore_index=True)

    # Overall per-example-per-model table
    overall_path = os.path.join(root, "grading_overall.csv")
    agg.to_csv(overall_path, index=False, encoding="utf-8-sig", sep=";", decimal=",")
    print(f"[OK] Wrote: {overall_path}")

    # Model-level summary (based on output file name)
    model_summary = (agg.groupby("model_output_file")["total_score"]
                     .agg(["count", "mean", "std", "min", "max"])
                     .reset_index()
                     .sort_values("mean", ascending=False))
    model_summary_path = os.path.join(root, "grading_model_summary.csv")
    model_summary.to_csv(model_summary_path, index=False, encoding="utf-8", sep=";", decimal=",")
    print(f"[OK] Wrote: {model_summary_path}")

    return agg

if __name__ == "__main__":
    # Run from the thesis folder:
    #   python grade_class_folder.py
    grade_all_cases_in_class_folder(
        root=r"benchmark_results\Class",
        csv_name="grading_results.csv"
    )