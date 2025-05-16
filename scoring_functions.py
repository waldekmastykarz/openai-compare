from bert_score import score as bert_score
from rapidfuzz.distance import Levenshtein
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from typing import Dict
import re

normalize_bert = True

def calculate_bleu(reference_texts: list[str], candidate: str) -> float:
    """
    Calculate BLEU score between reference and candidate texts.
    """
    all_results = []
    for reference in reference_texts:
        reference_tokens = reference.split()
        candidate_tokens = candidate.split()
        all_results.append(sentence_bleu([reference_tokens], candidate_tokens, smoothing_function=SmoothingFunction().method1))
    return max(all_results)

def calculate_rouge1(reference_texts: list[str], candidate: str) -> float:
    """
    Calculate ROUGE-1 score between reference and candidate texts.
    """
    return calculate_rouge(reference_texts, candidate, 'rouge1')

def calculate_rouge2(reference_texts: list[str], candidate: str) -> float:
    """
    Calculate ROUGE-2 score between reference and candidate texts.
    """
    return calculate_rouge(reference_texts, candidate, 'rouge2')

def calculate_rougeL(reference_texts: list[str], candidate: str) -> float:
    """
    Calculate ROUGE-L score between reference and candidate texts.
    """
    return calculate_rouge(reference_texts, candidate, 'rougeL')

def calculate_rouge(reference_texts: list[str], candidate: str, rouge_type: str) -> Dict[str, float]:
    """
    Calculate ROUGE scores between reference and candidate texts.
    """
    scorer = rouge_scorer.RougeScorer([rouge_type], use_stemmer=True)
    all_scores = []
    for reference in reference_texts:
        scores = scorer.score(reference, candidate)
        all_scores.append(scores[rouge_type].fmeasure)
    return max(all_scores)

def calculate_bert_f(reference_texts: list[str], candidate: str) -> float:
    """
    Calculate BERTScore between reference and candidate texts.
    """
    return calculate_bert(reference_texts, candidate, "f")

def calculate_bert_r(reference_texts: list[str], candidate: str) -> float:
    """
    Calculate BERTScore between reference and candidate texts.
    """
    return calculate_bert(reference_texts, candidate, "r")

def calculate_bert_p(reference_texts: list[str], candidate: str) -> float:
    """
    Calculate BERTScore between reference and candidate texts.
    """
    return calculate_bert(reference_texts, candidate, "p")

def calculate_bert(reference_texts: list[str], candidate: str, bert_type: str) -> float:
    """
    Calculate BERTScore between reference and candidate texts.
    """
    # P = Precision, Measures how much of the generated text is relevant compared to the reference text.
    # R = Recall, Measures how much of the reference text is covered by the generated text.
    # F1 = F1 Score, Harmonic mean of precision and recall
    P, R, F1 = bert_score([candidate], [reference_texts], lang="en")

    score = 0.0

    if bert_type == "p":
        score = P[0].item()
    elif bert_type == "r":
        score = R[0].item()
    else:
        score = F1[0].item()

    if normalize_bert:
        if score >= 0.95:
            score = 1.0
        elif score >= 0.75:
            score = (score - 0.75) / 0.2
        else:
            score = 0.0

    return score

def calculate_edit_distance(reference_texts: list[str], candidate: str) -> float:
    """
    Calculate normalized edit distance between reference and candidate texts.
    """
    all_results = []
    for reference in reference_texts:
        edit_distance = Levenshtein.distance(reference, candidate)
        max_length = max(len(reference), len(candidate))
        all_results.append(1 - (edit_distance / max_length))
    return max(all_results)

def exact_match(reference_texts: list[str], candidate: str) -> float:
    """
    Calculate exact match score between reference and candidate texts.
    """
    for reference in reference_texts:
        if reference.strip() == candidate.strip():
            return 1.0
    return 0.0

def split_camel_case(text: str) -> list[str]:
    return re.sub(r'(?<![A-Z])([A-Z])', r' \1', text).strip()