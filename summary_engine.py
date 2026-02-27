"""
MANOBHAV — Structured Deterministic Summary Engine
Accurate Section-wise Sentiment Classification
"""

import re
from collections import Counter, defaultdict
from typing import Dict, List


class HybridSummaryEngine:

    POS_WORDS = {
        "improve", "improved", "helpful", "smooth", "efficient", "success",
        "resolved", "reliable", "fast", "responsive", "benefit",
        "beneficial", "good", "great", "excellent", "clear"
    }

    NEG_WORDS = {
        "breach", "violation", "risk", "delay", "failure",
        "deficiency", "sanction", "crash", "error", "slow",
        "problem", "issue", "deny", "blocked", "penalty",
        "harsh", "difficult", "confusing", "unclear"
    }

    def __init__(self, batch_size: int = 500):
        self.batch_size = batch_size

    # ---------------------------------------------------
    # CLEANING
    # ---------------------------------------------------
    def _clean(self, comments: List[str]) -> List[str]:
        cleaned = []
        seen = set()

        for c in comments:
            if not c or not isinstance(c, str):
                continue

            t = re.sub(r"http\S+", "", c)
            t = re.sub(r"\s+", " ", t).strip()

            if not t:
                continue

            key = t.lower()
            if key not in seen:
                seen.add(key)
                cleaned.append(t)

        return cleaned

    # ---------------------------------------------------
    # STRICT SENTIMENT CLASSIFICATION
    # ---------------------------------------------------
    def _classify_comments(self, comments: List[str]) -> dict:

        pos_bucket = []
        neg_bucket = []
        neu_bucket = []

        for c in comments:
            low = c.lower()

            neg_hits = sum(
                1 for kw in self.NEG_WORDS
                if re.search(rf"\b{kw}\b", low)
            )

            pos_hits = sum(
                1 for kw in self.POS_WORDS
                if re.search(rf"\b{kw}\b", low)
            )

            # Strict rule:
            # Any negative keyword → negative
            if neg_hits > 0 and neg_hits >= pos_hits:
                neg_bucket.append(c)

            elif pos_hits > 0 and pos_hits > neg_hits:
                pos_bucket.append(c)

            else:
                neu_bucket.append(c)

        return {
            "positive": pos_bucket,
            "negative": neg_bucket,
            "neutral": neu_bucket,
            "total": len(comments)
        }

    # ---------------------------------------------------
    # FORMAT SECTION OUTPUT
    # ---------------------------------------------------
    def _top_k(self, items: List[str], k: int):
        if not items:
            return ["None"]

        freq = Counter(items)
        return [t for t, _ in freq.most_common(k)]

    def _format_section(self, title: str, data: dict) -> str:

        lines = []
        lines.append(f"{title}:")
        lines.append("Positive:")

        for p in self._top_k(data["positive"], 4):
            lines.append(f"- {p}")

        lines.append("")
        lines.append("Negative:")

        for n in self._top_k(data["negative"], 6):
            lines.append(f"- {n}")

        lines.append("")
        lines.append("Neutral:")

        for n in self._top_k(data["neutral"], 3):
            lines.append(f"- {n}")

        lines.append("")
        return "\n".join(lines)

    # ---------------------------------------------------
    # CROSS-SECTION DEDUPE
    # ---------------------------------------------------
    def _dedupe_cross_section(self, text: str) -> str:
        try:
            biz_neg = re.search(
                r"BUSINESS:.*?Negative:(.*?)(Neutral:|$)",
                text,
                re.DOTALL | re.IGNORECASE
            )
            ind_neg = re.search(
                r"INDIVIDUAL:.*?Negative:(.*?)(Neutral:|$)",
                text,
                re.DOTALL | re.IGNORECASE
            )

            if not biz_neg or not ind_neg:
                return text

            biz_block = biz_neg.group(1).strip()
            ind_block = ind_neg.group(1).strip()

            if biz_block and biz_block[:120] in ind_block:
                text = text.replace(
                    ind_block,
                    "\n- Similar compliance-related issues observed at individual level.\n"
                )

            return text

        except Exception:
            return text

    # ---------------------------------------------------
    # PUBLIC GENERATE METHOD
    # ---------------------------------------------------
    def generate(self, data: Dict[str, List[str]], output_format: str = "narrative") -> dict:

        if not isinstance(data, dict):
            return {"summary": "Invalid input format.", "source": "error"}

        biz_raw = data.get("business", [])
        ind_raw = data.get("individual", [])

        biz_clean = self._clean(biz_raw)
        ind_clean = self._clean(ind_raw)

        biz_classified = self._classify_comments(biz_clean)
        ind_classified = self._classify_comments(ind_clean)

        # Build a narrative summary from actual comment content
        final_summary = self._build_narrative(biz_classified, ind_classified)
        final_summary = self._dedupe_cross_section(final_summary)

        stats = {
            "business_total": biz_classified["total"],
            "business_positive": len(biz_classified["positive"]),
            "business_negative": len(biz_classified["negative"]),
            "business_neutral": len(biz_classified["neutral"]),
            "individual_total": ind_classified["total"],
            "individual_positive": len(ind_classified["positive"]),
            "individual_negative": len(ind_classified["negative"]),
            "individual_neutral": len(ind_classified["neutral"]),
        }

        return {
            "summary": final_summary,
            "source": "deterministic-structured",
            "stats": stats
        }

    # ---------------------------------------------------
    # NARRATIVE BUILDER  (uses real comment text)
    # ---------------------------------------------------
    def _build_narrative(self, biz: dict, ind: dict) -> str:
        """
        Build a readable structured narrative from classified comment buckets.
        Quotes up to 3 real comments per bucket so the output reflects the
        actual uploaded data, not canned phrases.
        """
        def excerpt(comments, n=3):
            """Return up to n short excerpts from real comments."""
            items = comments[:n]
            return ["  - " + (c[:120] + "…" if len(c) > 120 else c) for c in items]

        def sentiment_label(pos, neg, total):
            if total == 0:
                return "no data"
            pos_pct = pos / total
            neg_pct = neg / total
            if pos_pct >= 0.6:
                return "predominantly positive"
            elif neg_pct >= 0.6:
                return "predominantly critical"
            elif pos_pct > neg_pct:
                return "mixed, leaning positive"
            elif neg_pct > pos_pct:
                return "mixed, leaning critical"
            return "mixed"

        biz_total = biz["total"]
        ind_total = ind["total"]
        overall_total = biz_total + ind_total

        biz_label = sentiment_label(
            len(biz["positive"]), len(biz["negative"]), biz_total
        )
        ind_label = sentiment_label(
            len(ind["positive"]), len(ind["negative"]), ind_total
        )

        lines = []

        # ── Overall header ──
        lines.append(
            f"OVERALL SUMMARY ({overall_total} submissions: "
            f"{biz_total} business, {ind_total} individual)\n"
        )
        lines.append(
            f"Business sentiment: {biz_label}  |  "
            f"Individual sentiment: {ind_label}\n"
        )
        lines.append("─" * 60)

        # ── Business section ──
        lines.append("\nBUSINESS STAKEHOLDERS:\n")

        if biz["positive"]:
            lines.append(
                f"Positive ({len(biz['positive'])} comments):"
            )
            lines.extend(excerpt(biz["positive"]))
            lines.append("")

        if biz["negative"]:
            lines.append(
                f"Key Concerns ({len(biz['negative'])} comments):"
            )
            lines.extend(excerpt(biz["negative"]))
            lines.append("")

        if biz["neutral"]:
            lines.append(
                f"Neutral / Observational ({len(biz['neutral'])} comments):"
            )
            lines.extend(excerpt(biz["neutral"]))
            lines.append("")

        if biz_total == 0:
            lines.append("  No business submissions found.\n")

        lines.append("─" * 60)

        # ── Individual section ──
        lines.append("\nINDIVIDUAL / CITIZEN FEEDBACK:\n")

        if ind["positive"]:
            lines.append(
                f"Positive ({len(ind['positive'])} comments):"
            )
            lines.extend(excerpt(ind["positive"]))
            lines.append("")

        if ind["negative"]:
            lines.append(
                f"Key Concerns ({len(ind['negative'])} comments):"
            )
            lines.extend(excerpt(ind["negative"]))
            lines.append("")

        if ind["neutral"]:
            lines.append(
                f"Neutral / Observational ({len(ind['neutral'])} comments):"
            )
            lines.extend(excerpt(ind["neutral"]))
            lines.append("")

        if ind_total == 0:
            lines.append("  No individual submissions found.\n")

        lines.append("─" * 60)

        # ── Closing assessment ──
        lines.append("\nASSESSMENT:")
        biz_pos_pct = round(len(biz["positive"]) / biz_total * 100) if biz_total else 0
        biz_neg_pct = round(len(biz["negative"]) / biz_total * 100) if biz_total else 0
        ind_pos_pct = round(len(ind["positive"]) / ind_total * 100) if ind_total else 0
        ind_neg_pct = round(len(ind["negative"]) / ind_total * 100) if ind_total else 0

        lines.append(
            f"  Business : {biz_pos_pct}% positive, {biz_neg_pct}% critical"
        )
        lines.append(
            f"  Individual: {ind_pos_pct}% positive, {ind_neg_pct}% critical"
        )

        return "\n".join(lines)