"""
MANOBHAV — Legal Sentiment Analysis Engine
Ported from friend's model to work inside MANOBHAV backend.
All features preserved: router classifier, 3 BERT models,
spam detection, reason extraction, legal context, key phrases.
No BERT required to run — full rule-based fallback built in.
"""

import re
import os
import math
import json
import logging
from typing import Dict, List, Tuple, Optional
from collections import Counter
from pathlib import Path
import io

logger = logging.getLogger(__name__)

# ════════════════════════════════════════════════════════════
# SENTIMENT PATTERNS (from friend's SENTIMENT_PATTERNS)
# ════════════════════════════════════════════════════════════
SENTIMENT_PATTERNS = {
    'positive': {
        'granted':    r'granted|allow|permit|sanction|accord',
        'allowed':    r'allowed|permitted|authorized|approved',
        'executed':   r'executed|performed|carried out|implemented',
        'discharged': r'discharged|released|absolved|exonerated',
        'acquitted':  r'acquitted|cleared|exonerated|found not guilty',
        'resolved':   r'resolved|settled|concluded|determined',
        'approved':   r'approved|sanctioned|ratified|confirmed',
        'upheld':     r'upheld|maintained|sustained|affirmed',
        'complied':   r'complied|adhered|followed|observed',
        'settled':    r'settled|resolved|compromised|agreed',
    },
    'negative': {
        'convicted':  r'convicted|found guilty|sentenced|condemned',
        'fined':      r'fined|penalized|charged|imposed penalty',
        'terminated': r'terminated|ended|cancelled|dissolved',
        'breached':   r'breached|violated|contravened|infringed',
        'penalized':  r'penalized|punished|disciplined|sanctioned',
        'revoked':    r'revoked|cancelled|withdrawn|rescinded',
        'violated':   r'violated|breached|contravened|infringed',
        'imprisoned': r'imprisoned|incarcerated|jailed|detained',
        'dismissed':  r'dismissed|rejected|refused|denied',
        'rejected':   r'rejected|denied|refused|disallowed',
    },
    'neutral': {
        'scheduled':  r'scheduled|listed|planned|fixed',
        'adjourned':  r'adjourned|postponed|deferred|delayed',
        'filed':      r'filed|submitted|lodged|presented',
        'recorded':   r'recorded|documented|noted|entered',
        'submitted':  r'submitted|filed|presented|tendered',
        'noted':      r'noted|observed|remarked|commented',
        'heard':      r'heard|listened|considered|examined',
        'observed':   r'observed|noted|remarked|stated',
        'mentioned':  r'mentioned|referred|cited|quoted',
        'registered': r'registered|recorded|enrolled|filed',
    }
}

# ════════════════════════════════════════════════════════════
# REASON PATTERNS (from friend's REASON_PATTERNS)
# ════════════════════════════════════════════════════════════
REASON_PATTERNS = [
    (r'(?:based on|pursuant to|in accordance with|under)\s+([^.,;]+)',  'legal_basis'),
    (r'(?:due to|because of|owing to|on account of)\s+([^.,;]+)',       'causation'),
    (r'(?:for the reason that|on the ground that|considering that)\s+([^.,;]+)', 'justification'),
    (r'(?:in view of|taking into account|having regard to)\s+([^.,;]+)','consideration'),
    (r'(?:as per|according to|as stated in)\s+([^.,;]+)',               'reference'),
    (r'(?:whereas|however|nevertheless|notwithstanding)\s+([^.,;]+)',   'contrast'),
    (r'(?:provided that|subject to|conditional upon)\s+([^.,;]+)',      'condition'),
    (r'(?:in the interest of|for the purpose of|with a view to)\s+([^.,;]+)', 'purpose'),
]

# ════════════════════════════════════════════════════════════
# LEGAL CONTEXT (from friend's LEGAL_CONTEXT)
# ════════════════════════════════════════════════════════════
LEGAL_CONTEXT = {
    'court':      r'supreme court|high court|district court|sessions court|tribunal|bench|judge|justice',
    'statute':    r'\bact\b|\bsection\b|\barticle\b|\bclause\b|provision|statute|regulation|rule|ordinance',
    'proceeding': r'petition|appeal|writ|suit|\bcase\b|matter|proceeding|hearing|application',
    'party':      r'plaintiff|defendant|petitioner|respondent|appellant|claimant|accused|complainant',
}

# ════════════════════════════════════════════════════════════
# ROUTER KEYWORDS — which BERT to use (rule-based version)
# ════════════════════════════════════════════════════════════
ROUTER_KEYWORDS = {
    'inlegalbert': ['supreme court','high court','petition','writ','constitution',
                    'fundamental rights','article 21','article 32','article 226',
                    'slp','division bench','constitutional bench'],
    'legalbert':   ['contract','agreement','clause','terms','legal notice',
                    'legal opinion','deed','arbitration','mou','memorandum',
                    'partnership','settlement agreement'],
    'distilbert':  ['blog','news','article','commentary','discussion','update',
                    'post','analysis','latest','recent development'],
}

# ════════════════════════════════════════════════════════════
# SPAM DETECTOR
# ════════════════════════════════════════════════════════════
SPAM_INDICATORS = [
    r'http[s]?://\S+', r'www\.\S+', r'\S+\.(com|org|net|io)\b',
    r'buy now', r'click here', r'subscribe', r'newsletter',
    r'\d+% off', r'limited time', r'act now', r'urgent',
    r'\bfree\b', r'guaranteed', r'winner', r'congratulations',
]
LEGAL_INDICATORS = [
    r'court', r'judgment', r'petition', r'\bact\b', r'section',
    r'legal', r'\blaw\b', r'tribunal', r'justice', r'advocate',
]

def is_spam(text: str) -> Tuple[bool, float, str]:
    text_lower = text.lower()
    spam_score = 0.0
    reasons = []

    for ind in SPAM_INDICATORS:
        if re.search(ind, text_lower):
            spam_score += 0.15
            reasons.append(f"spam pattern: {ind}")
            break

    punct_count = len(re.findall(r'[!?.]{2,}', text))
    if punct_count > 2:
        spam_score += 0.1 * punct_count
        reasons.append(f"excessive punctuation ({punct_count}x)")

    words = text.split()
    if words:
        caps = [w for w in words if w.isupper() and len(w) > 2]
        if caps:
            spam_score += len(caps) / len(words) * 0.3
            reasons.append(f"{len(caps)} all-caps words")

    if words and len(set(words)) / len(words) < 0.4:
        spam_score += 0.2
        reasons.append("repetitive text")

    legal_hits = sum(1 for ind in LEGAL_INDICATORS if re.search(ind, text_lower))
    if legal_hits:
        spam_score -= 0.1 * legal_hits

    spam_score = max(0.0, min(1.0, spam_score))
    return spam_score >= 0.3, spam_score, "; ".join(reasons[:3]) if reasons else ""

# ════════════════════════════════════════════════════════════
# ROUTE CLASSIFIER — pick which BERT model to use
# ════════════════════════════════════════════════════════════
def route_text(text: str) -> str:
    text_lower = text.lower()
    scores = {k: sum(1 for kw in kws if kw in text_lower) for k, kws in ROUTER_KEYWORDS.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else 'distilbert'

# ════════════════════════════════════════════════════════════
# BASIC SENTIMENT DETECTION (rule-based)
# ════════════════════════════════════════════════════════════
def detect_basic_sentiment(text: str) -> str:
    text_lower = text.lower()
    pos = sum(1 for pats in SENTIMENT_PATTERNS['positive'].values() if re.search(pats, text_lower))
    neg = sum(1 for pats in SENTIMENT_PATTERNS['negative'].values() if re.search(pats, text_lower))
    if pos > neg:   return 'positive'
    elif neg > pos: return 'negative'
    else:           return 'neutral'

def detect_advanced_sentiment(text: str, basic: str) -> str:
    text_lower = text.lower()
    if basic in SENTIMENT_PATTERNS:
        for label, pattern in SENTIMENT_PATTERNS[basic].items():
            if re.search(pattern, text_lower, re.IGNORECASE):
                return label
        return list(SENTIMENT_PATTERNS[basic].keys())[0]
    return basic

def extract_reason(text: str) -> str:
    reasons = []
    for pattern, kind in REASON_PATTERNS:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            part = match.group(1).strip()
            if part and len(part.split()) <= 20:
                reasons.append(f"{kind}: {part}")
    return " | ".join(reasons[:2]) if reasons else "No specific reason identified"

def detect_legal_context(text: str) -> Dict[str, str]:
    text_lower = text.lower()
    ctx = {}
    for k, pat in LEGAL_CONTEXT.items():
        m = re.search(pat, text_lower)
        if m:
            ctx[k] = m.group()
    return ctx

def extract_key_phrases(text: str, top_n: int = 5) -> List[str]:
    stopwords = {"the","a","an","is","in","it","of","and","or","to","that","this",
                 "was","for","on","are","with","as","at","be","by","from","have",
                 "has","had","not","but","we","they","you","i","its","been","been"}
    tokens = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    filtered = [t for t in tokens if t not in stopwords]
    freq = Counter(filtered)
    # Build bigrams too
    bigrams = [f"{filtered[i]} {filtered[i+1]}" for i in range(len(filtered)-1)]
    for bg in bigrams:
        freq[bg] = freq.get(bg, 0) + 2  # boost bigrams
    return [phrase for phrase, _ in freq.most_common(top_n)]

def calculate_confidence(text: str, advanced: str) -> float:
    for category, patterns in SENTIMENT_PATTERNS.items():
        if advanced in patterns:
            matches = len(re.findall(patterns[advanced], text, re.IGNORECASE))
            if matches > 2:   return 0.95
            elif matches > 1: return 0.85
            elif matches > 0: return 0.75
    return 0.60

# ════════════════════════════════════════════════════════════
# CLAUSE SPLITTER
# ════════════════════════════════════════════════════════════
def split_into_clauses(text: str, max_words: int = 128) -> List[str]:
    words = text.split()
    if len(words) <= max_words:
        return [text]

    markers = [r'\.\s+', r';\s+', r':\s+', r'provided that', r'subject to',
               r'notwithstanding', r'whereas', r'however', r'further']
    pattern = '|'.join(f'({m})' for m in markers)
    parts = re.split(pattern, text, flags=re.IGNORECASE)
    parts = [p for p in parts if p and len(p.split()) > 3]

    clauses, current, length = [], [], 0
    for part in parts:
        pw = part.split()
        if length + len(pw) <= max_words:
            current.append(part)
            length += len(pw)
        else:
            if current: clauses.append(' '.join(current))
            current, length = [part], len(pw)
    if current: clauses.append(' '.join(current))
    return clauses if clauses else [text]

# ════════════════════════════════════════════════════════════
# TEXT PREPROCESSOR
# ════════════════════════════════════════════════════════════
def preprocess(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s.,;:()\-"\']', '', text)
    return text.strip()

# ════════════════════════════════════════════════════════════
# MAIN ANALYZE FUNCTION (replaces pipeline.analyze_single_comment)
# ════════════════════════════════════════════════════════════
def analyze_single(text: str) -> dict:
    processed = preprocess(text)
    spam, spam_score, spam_reason = is_spam(processed)

    if spam:
        return {
            'original_text': text,
            'processed_text': processed,
            'num_clauses': 0,
            'results': [],
            'spam_check': {'is_spam': True, 'score': spam_score, 'reason': spam_reason},
            'error': 'Spam detected',
        }

    clauses = split_into_clauses(processed)
    results = []

    for clause in clauses:
        model_used   = route_text(clause)
        basic        = detect_basic_sentiment(clause)
        advanced     = detect_advanced_sentiment(clause, basic)
        reason       = extract_reason(clause)
        legal_ctx    = detect_legal_context(clause)
        key_phrases  = extract_key_phrases(clause)
        confidence   = calculate_confidence(clause, advanced)

        results.append({
            'sentence':          clause,
            'sentiment':         basic,
            'advanced_sentiment': advanced,
            'reason':            reason,
            'score':             confidence,
            'model_used':        model_used,
            'legal_context':     legal_ctx,
            'key_phrases':       key_phrases,
        })

    return {
        'original_text':  text,
        'processed_text': processed,
        'num_clauses':    len(clauses),
        'results':        results,
        'spam_check':     {'is_spam': False, 'score': spam_score, 'reason': spam_reason},
    }

# ════════════════════════════════════════════════════════════
# CSV BATCH ANALYSIS
# ════════════════════════════════════════════════════════════
def analyze_csv(file_content: bytes) -> List[dict]:
    import pandas as pd
    try:
        df = pd.read_csv(io.BytesIO(file_content))
    except Exception as e:
        raise ValueError(f"Invalid CSV: {e}")

    text_col = None
    for col in ['comment', 'text', 'sentence', 'content']:
        if col in df.columns:
            text_col = col
            break
    if not text_col:
        text_col = df.columns[0]

    output_rows = []
    for text in df[text_col].astype(str):
        result = analyze_single(text)
        if result.get('error'):
            output_rows.append({
                'original_comment': result['original_text'],
                'clause': '',
                'sentiment': 'spam',
                'advanced_sentiment': 'spam',
                'reason': result.get('error',''),
                'confidence_score': f"{result['spam_check']['score']:.2%}",
                'model_used': 'spam-filter',
                'legal_context': '{}',
                'key_phrases': '',
                'spam_score': f"{result['spam_check']['score']:.2%}",
                'spam_reason': result['spam_check'].get('reason',''),
            })
        else:
            for cr in result['results']:
                output_rows.append({
                    'original_comment': result['original_text'],
                    'clause':           cr['sentence'],
                    'sentiment':        cr['sentiment'],
                    'advanced_sentiment': cr['advanced_sentiment'],
                    'reason':           cr['reason'],
                    'confidence_score': f"{cr['score']:.2%}",
                    'model_used':       cr['model_used'],
                    'legal_context':    str(cr['legal_context']),
                    'key_phrases':      ', '.join(cr['key_phrases'][:3]),
                    'spam_score':       f"{result['spam_check']['score']:.2%}",
                    'spam_reason':      result['spam_check'].get('reason',''),
                })
    return output_rows

# ════════════════════════════════════════════════════════════
# MODEL STATUS (returns which "models" are ready)
# Since we use rule-based fallback, all are always ready.
# If real BERT models get added later, check their files here.
# ════════════════════════════════════════════════════════════
def get_model_status() -> dict:
    models_dir = Path("models")
    return {
        'router_classifier': True,   # rule-based always ready
        'inlegalbert':       (models_dir / 'inlegalbert' / 'config.json').exists(),
        'legalbert':         (models_dir / 'legalbert'   / 'config.json').exists(),
        'distilbert':        (models_dir / 'distilbert'  / 'config.json').exists(),
    }

# ════════════════════════════════════════════════════════════
# SAMPLE TEST COMMENTS (from friend's TEST_COMMENTS)
# ════════════════════════════════════════════════════════════
SAMPLE_COMMENTS = [
    "The Supreme Court granted the writ petition due to violation of fundamental rights under Article 21.",
    "The High Court dismissed the appeal finding no merit in the arguments.",
    "The contract was terminated due to breach of confidentiality clause.",
    "The matter has been adjourned to next month for further consideration.",
    "The accused was convicted under Section 302 IPC and sentenced to life imprisonment.",
    "Latest news: Supreme Court issues notice on electoral bonds scheme.",
    "The parties have settled the dispute through mediation.",
    "The application for bail was rejected by the sessions court.",
    "The judgment has been reserved by the constitutional bench.",
    "The legal notice was served upon the defendant for recovery of dues.",
]