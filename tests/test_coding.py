import numpy as np
import pandas as pd

from thematicml.coding import apply_codebook, extract_phrase_codes, keyword_regex, train_record_classifier
from thematicml.config import AnalysisConfig, CodebookRule
from thematicml.features import fit_tfidf, phrase_regex

TEXTS = [
    "The customer service was slow to respond.",
    "Customer service never answered my email.",
    "I love the customer service team.",
    "The menus are confusing and navigation is difficult.",
    "Navigation is difficult on the mobile app.",
    "It is difficult to navigate the settings.",
    "The app is not easy to use.",
    "Pricing is too expensive.",
]


def segs(texts=TEXTS):
    return pd.DataFrame({"record_id": range(len(texts)), "text": texts, "char_start": 0, "char_end": [len(t) for t in texts]})


def test_tfidf_keeps_negations():
    f = fit_tfidf(TEXTS, AnalysisConfig(min_df=1, ngram_max=2))
    vocab = set(f.vocabulary)
    assert "not" in vocab and "not easy" in vocab
    assert "the" not in vocab
    s = f.summary()
    assert s["vocabulary_size"] == len(vocab) and s["matrix_shape"] == [len(TEXTS), len(vocab)]


def test_synonyms_applied():
    f = fit_tfidf(TEXTS, AnalysisConfig(min_df=1, synonyms={"customer service": "support"}))
    assert "support" in set(f.vocabulary)
    assert "customer" not in set(f.vocabulary)


def test_phrase_codes_traceable():
    s = segs()
    f = fit_tfidf(s["text"].tolist(), AnalysisConfig(min_df=1, ngram_max=2))
    codes, assigns = extract_phrase_codes(f, s, n_codes=10, min_support=2)
    assert "kw:customer service" in set(codes["code_key"])
    for a in assigns.itertuples():
        if a.matched_text is not None:
            assert s.loc[a.segment_pos, "text"][a.char_start:a.char_end] == a.matched_text
    # "customer" alone is redundant with "customer service" and is not selected
    assert "kw:customer" not in set(codes["code_key"])


def test_phrase_regex_allows_removed_stopwords():
    assert phrase_regex("difficult navigate").search("It is difficult to navigate the settings")
    assert not phrase_regex("menu").search("menus")


def test_codebook_rules_with_wildcards_and_negation():
    rules = [CodebookRule(label="Navigation", keywords=["navigat*", "menu"]), CodebookRule(label="Ease", keywords=["easy"])]
    codes, assigns = apply_codebook(rules, segs())
    nav = assigns[assigns.code_key == "rule:Navigation"]
    assert set(nav.segment_pos) == {3, 4, 5}
    ease = assigns[assigns.code_key == "rule:Ease"].iloc[0]
    assert bool(ease.negated) is True  # "not easy"
    assert keyword_regex("navigat*").search("Navigation").group(0) == "Navigation"


def test_both_mode_leaves_codebook_terms_to_the_codebook():
    from thematicml import pipeline as P

    rules = [CodebookRule(label="Support", keywords=["customer service"])]
    st = {"segments": segs(), "warnings": [], "summary": {}}
    st["features"] = fit_tfidf(st["segments"]["text"].tolist(), AnalysisConfig(min_df=1, ngram_max=2))
    cfg = AnalysisConfig(coding_method="both", codebook=rules, n_codes=3, min_code_support=2)
    P._stage_coding(None, None, cfg, st)
    keys = set(st["codes"]["code_key"])
    assert "rule:Support" in keys and "kw:customer service" not in keys
    assert len([k for k in keys if k.startswith("kw:")]) == 3  # the freed slot goes to the next-ranked term
    assert "customer service" in st["summary"]["coding"]["tfidf_terms_left_to_codebook"]["top"]
    P._stage_coding(None, None, cfg.model_copy(update={"complement_codebook": False}), st)
    assert "kw:customer service" in set(st["codes"]["code_key"])


def test_supervised_grouped_split_no_leakage():
    rng = np.random.default_rng(0)
    rows = []
    for i in range(80):
        lab = ["nav", "price"][i % 2]
        txt = "navigation menus confusing" if lab == "nav" else "price expensive subscription"
        rows.append({"record_id": i, "record_key": f"P{i // 2}", "text": f"{txt} {rng.integers(1000)}", "label": lab if i < 60 else None})
    recs = pd.DataFrame(rows)
    s = recs.assign(char_start=0, char_end=recs.text.str.len())[["record_id", "text", "char_start", "char_end"]]
    f = fit_tfidf(s["text"].tolist(), AnalysisConfig(min_df=1))
    codes, assigns, ev = train_record_classifier(recs, s, f, 0)
    assert ev["available"] and ev["accuracy"] >= 0.9
    assert ev["split"]["grouped_by"].startswith("record_key")
    assert ev["predicted_records"] == 20
    assert set(assigns["method"]) == {"researcher-label", "supervised"}


def test_contractions_are_stopwords_and_curly_apostrophes_match():
    from thematicml.config import AnalysisConfig
    from thematicml.features import TextNormaliser, phrase_regex

    n = TextNormaliser(AnalysisConfig())
    toks = n.tokenize(n.preprocess("It’s late and I’m sure we don’t know"))
    assert "it's" not in toks and "i'm" not in toks
    assert "don't" in toks  # negations are kept
    m = phrase_regex("don't know").search("We don’t know yet")
    assert m and m.group(0) == "don’t know"
