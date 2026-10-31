"""Offline language identification, so English-only analysis can set aside non-English passages.

Coding rules, English stopwords and VADER sentiment only make sense for English; a Portuguese
passage would otherwise turn into junk codes. Detection scores fixed profiles of very frequent
function words by their share of the passage's words, plus a little character evidence (ã/õ, ñ/¿,
ß/umlauts, ...). No model, no downloads, no network: deterministic, explainable (``language_scores``)
and fast enough to run on every passage.

Tuned for precision on English. Capitalised words in mid-sentence are names (Yang, São Paulo,
Den Haag) and carry no evidence; foreign words that are also common English words ("solo", "on",
"car", "um") are left out of the foreign profiles; ``is_non_english`` fires only when another
language clearly dominates the passage.
"""
from __future__ import annotations

import math
import re

LANGUAGES = ("en", "pt", "es", "fr", "de", "it", "nl", "id")
UNDETERMINED = "und"

# ~100-130 of the most frequent words per language. Deliberately excluded because they are
# also everyday English words: pt/de "um" (filler), es/it "solo", fr "on"/"or"/"plus"/"car"/"ton",
# it "per"/"come"/"via", nl "of"/"over"/"even"/"had"/"door", de "man"/"will"/"also".
_PROFILES = {
    "en": """the of and to a in is it you that he was for on are with as i his they be at have this
        from or had by not but what some we can out other were all there when up your how an she which
        do their if will about many then them would so these her him has been more no could my than who
        its now did may into only our me also just very after should because where any most through
        while before over those such both between here us being does why again too off own under until
        against though never every always often something nothing much well even still might must yet
        whether during without within across around among however although really like get got know
        think going am don't it's i'm can't didn't doesn't isn't wasn't you're i've that's there's
        they're we're let's won't""",
    "pt": """de a o que e do da em para é com não uma os no se na por mais as dos como mas foi ao ele
        das tem à seu sua ou ser quando muito muita muitos muitas há nos já está estão eu também só pelo
        pela pelos pelas até isso isto ela eles elas entre depois sem mesmo mesma aos às ter seus suas
        quem nas me esse essa esses essas este esta estes estas você vocês tinha foram num numa nem meu
        minha têm havia seja qual quais será nós lhe dele dela sobre ainda onde porque então assim pode
        podem deve sido tudo todos todas outro outra outros outras cada após sempre apenas durante
        contra desde além pois aqui agora bem tão algum alguma alguns fazer""",
    "es": """de la que el en y a los se del las un por con no una su para es al lo como más pero sus le
        ha me si sin sobre este ya entre cuando todo esta ser son también fue había muy hasta desde está
        mi porque qué han yo hay puede todos así nos ni tiene él uno donde bien mismo ese ahora cada otro
        después te otros aunque esa eso hace otra durante siempre ella sí sido según menos antes contra
        sino nada hacer estaba poco estos unos les algo hacia ellos mucho mientras además quien esto
        están pues hoy otras entonces todas debe cómo casi toda luego estas sea tenía nunca aquí tienen
        pueden fueron tras o""",
    "fr": """de la le et les des en un du une que est pour qui dans a par pas au sur ne se ce il sont aux
        avec son sa ses cette ou mais elle ils elles nous vous je tu leur leurs été être avoir fait comme
        tout tous toute toutes bien très aussi peut sans sous entre après avant encore même ont avait
        était sera ces cet mon ma mes notre nos votre vos lui eux me te y où quand si donc alors ni chez
        vers depuis contre selon moins autre autres chaque quelques rien peu beaucoup toujours jamais
        déjà ici là ça cela ceci celui celle ceux quoi plusieurs faire dit puis ainsi aujourd'hui""",
    "de": """der die und in den von zu das mit sich des auf für ist im dem nicht ein eine als auch es an
        werden aus hat dass sie nach wird bei einer am sind noch wie einem über einen so zum war haben
        nur oder aber vor zur bis mehr durch sein wurde sei kann ich wir ihr ihre ihren ihrem ihrer sehr
        schon wenn was dann unter wieder gegen diese dieser dieses diesem diesen vom beim kein keine
        keinen mein meine uns mich mir dich dir ihm ihn ihnen hier da weil ob doch ja immer jetzt alle
        alles viel viele etwas nichts können müssen muss soll würde wurden worden waren gibt ganz
        zwischen ohne während seit bereits jedoch denn dort hatte hatten wo wer du dein deine unsere
        unser euch damit sowie habe""",
    "it": """di che e il la a un in è non una i sono si da con del della le al lo ma più anche ha gli dei
        nel nella alla delle degli se ci mi ti ne questo questa questi queste quello quella quelli cosa
        tutto tutti tutta molto molta molti essere fare stato stata suo sua suoi sue loro mio mia io lui
        lei noi voi dove quando perché così già poi ancora sempre tra fra dopo prima senza sul sulla dal
        dalla nei nelle negli alle agli ai erano hanno ho abbiamo avere fatto può deve ogni altro altra
        altri qui oggi bene proprio sia quale quali quanto nostro nostra cui mentre però quindi invece
        oppure o uno""",
    "nl": """de van een het en in is dat op te zijn voor met die niet aan om ook als maar bij uit nog wel
        naar kan dan wordt hij ze zij ik je we wij jij hun haar hem mij me ons onze mijn jouw uw zo meer
        veel geen hebben heeft worden werd waren was kunnen moet moeten zou zal wat wie waar omdat dus
        toch nu hier daar alleen andere tussen onder sinds tegen zonder na deze dit welke elk elke iets
        niets alles iemand niemand altijd nooit zeer erg gaan gaat komt komen doen gedaan geweest werden
        echter waarom zich ben bent eens""",
    "id": """yang dan di ke dari untuk dengan ini itu tidak ada akan pada dalam juga atau saya kami kita
        mereka dia ia bisa dapat oleh karena sebagai lebih bahwa jika tetapi hanya masih harus sangat
        banyak seperti setelah saat antara belum tersebut secara hingga sehingga maka namun bagi agar
        adalah merupakan menjadi sendiri kepada tentang apa semua setiap beberapa lain sudah telah sedang
        pun lagi kalau tapi aku kamu anda begitu bagaimana mengapa siapa yaitu serta maupun bahkan tanpa
        hal sebuah seorang tak sejak selama terhadap melalui menurut sebelum ketika supaya walaupun
        meskipun demikian selain yakni ialah punya mau""",
}
# Elided articles/pronouns written with an apostrophe (l'homme, dell'anno): scored as a word.
_ELISIONS = {
    "l'": ("fr", "it"), "d'": ("fr",), "j'": ("fr",), "qu'": ("fr",), "n'": ("fr",), "s'": ("fr",),
    "c'": ("fr", "it"), "m'": ("fr",), "t'": ("fr",), "jusqu'": ("fr",), "lorsqu'": ("fr",),
    "puisqu'": ("fr",), "un'": ("it",), "dell'": ("it",), "nell'": ("it",), "all'": ("it",),
    "dall'": ("it",), "sull'": ("it",), "quest'": ("it",), "quell'": ("it",),
}
# Letters that English words almost never contain, mapped to the languages that use them.
_CHAR_LANGS = {
    "ã": ("pt",), "õ": ("pt",), "ç": ("pt", "fr"), "ê": ("pt", "fr"), "â": ("pt", "fr"),
    "ô": ("pt", "fr"), "à": ("pt", "fr", "it"), "á": ("es", "pt"), "í": ("es", "pt"),
    "ó": ("es", "pt"), "ú": ("es", "pt"), "é": ("fr", "es", "pt", "it"), "è": ("fr", "it"),
    "ì": ("it",), "ò": ("it",), "ù": ("it", "fr"), "ñ": ("es",), "ä": ("de",), "ö": ("de",),
    "ü": ("de",), "ß": ("de",), "î": ("fr",), "ï": ("fr", "nl"), "û": ("fr",), "œ": ("fr",),
    "ë": ("fr", "nl"),
}
_CHAR_WEIGHT = 0.5      # a distinctive letter is weaker evidence than a function word
_BETA = 1.0             # log-odds per net function-word hit, for the confidence
_FULL_SHARE = 0.25      # running prose has >= 25-50% function words; tables and name lists have far fewer
_MIN_SHARE = 0.12       # below this share the passage is mostly names/numbers: undetermined
_MIN_CONF = 0.5
# is_non_english: the foreign language must clearly dominate English.
_FLAG_MIN_SHARE = 0.2
_FLAG_MIN_MARGIN = 3.0  # net hits over English
_FLAG_EN_RATIO = 0.5    # English evidence at most half the foreign evidence

_IDX = {lang: i for i, lang in enumerate(LANGUAGES)}
_WORDS: dict[str, tuple[int, ...]] = {}
for _lang, _words in _PROFILES.items():
    for _w in _words.split():
        _WORDS[_w] = _WORDS.get(_w, ()) + (_IDX[_lang],)
_ELISION_IDX = {k: tuple(_IDX[x] for x in v) for k, v in _ELISIONS.items()}
_CHAR_IDX = {k: tuple(_IDX[x] for x in v) for k, v in _CHAR_LANGS.items()}
_EN, _ES, _NL, _ID = _IDX["en"], _IDX["es"], _IDX["nl"], _IDX["id"]

_TOKEN = re.compile(r"[^\W\d_]+(?:['’-][^\W\d_]+)*")
# Latin tags and abbreviations in English prose whose parts look like Romance words ("et", "e", "se").
_LATIN = re.compile(r"\b(?:et\s+al|e\.\s?g|i\.\s?e|per\s+se|a\s+priori|ad\s+hoc|de\s+facto|vice\s+versa"
                    r"|status\s+quo|et\s+cetera|etc|cf|viz|vs)\b\.?", re.IGNORECASE)
_URL = re.compile(r"https?://\S+|www\.\S+|\S+@\S+\.\w+")
_HYPHEN_BREAK = re.compile(r"(?<=[^\W\d_])- +(?=[^\W\d_])")  # PDF line-break hyphenation: "desloca- mentos"
_BOUNDARY = re.compile(r"[.!?…:;\n•–—]")


def _evidence(text: str) -> tuple[int, list[float]]:
    """Return (word count, per-language evidence) for ``text``."""
    text = _HYPHEN_BREAK.sub("", _LATIN.sub(" ", _URL.sub(" ", text)))
    caps = text.isupper()
    hits = [0.0] * len(LANGUAGES)
    chars = [0] * len(LANGUAGES)
    n = 0
    prev = -1
    for m in _TOKEN.finditer(text):
        n += 1
        t = m.group()
        start, end = m.span()
        sentence_start = prev < 0 or _BOUNDARY.search(text, prev, start) is not None
        prev = end
        if t.islower():
            w = t
        elif caps or t == "I" or t.startswith(("I'", "I’")):
            w = t.lower()
        elif sentence_start and (t[1:].islower() or (len(t) == 1 and text[end:end + 1] != ".")):
            w = t.lower()   # sentence-initial word; single capitals followed by "." are initials
        else:
            continue        # names, acronyms, German nouns: no function-word evidence
        if "’" in w:
            w = w.replace("’", "'")
        if "'" in w and w not in _WORDS:
            cut = w.find("'") + 1
            for i in _ELISION_IDX.get(w[:cut], ()):
                hits[i] += 1
            w = w[cut:]
        for i in _WORDS.get(w, ()):
            hits[i] += 1
        if not w.isascii():
            seen: set[int] = set()
            for ch in w:
                seen.update(_CHAR_IDX.get(ch, ()))
            for i in seen:
                chars[i] += 1
        elif "ij" in w:
            chars[_NL] += 1
        if len(w) >= 6 and w.endswith(("nya", "kan")):
            chars[_ID] += 1
    chars[_ES] += text.count("¿") + text.count("¡")
    return n, [h + _CHAR_WEIGHT * c for h, c in zip(hits, chars)]


def _softmax_top(ev: list[float], top: int) -> float:
    return 1.0 / sum(math.exp(-_BETA * (ev[top] - e)) for e in ev)


def language_scores(text: str) -> dict[str, float]:
    """Per-language evidence as a share of the passage's words (function words plus
    weighted distinctive letters), with ``n_words``. Explains a ``detect_language`` call."""
    n, ev = _evidence(text or "")
    out = {lang: round(e / n, 4) if n else 0.0 for lang, e in zip(LANGUAGES, ev)}
    out["n_words"] = n
    return out


def detect_language(text: str, min_words: int = 8) -> tuple[str, float]:
    """Return (ISO 639-1 code, confidence in [0, 1]) for one of ``LANGUAGES``.

    Returns ("und", 0.0) when the text has fewer than ``min_words`` words or the evidence is weak
    (mostly names, numbers or references) or split between languages. The confidence combines how
    clearly the best language beats the others (softmax over net function-word hits) with how much
    of the text is evidence at all, so short or name-heavy passages score lower.
    """
    n, ev = _evidence(text or "")
    if n < min_words:
        return UNDETERMINED, 0.0
    top = max(range(len(ev)), key=ev.__getitem__)  # ties go to the earlier language, i.e. English
    share = ev[top] / n
    conf = _softmax_top(ev, top) * min(1.0, share / _FULL_SHARE)
    if share < _MIN_SHARE or conf < _MIN_CONF:
        return UNDETERMINED, 0.0
    return LANGUAGES[top], round(conf, 3)


def is_non_english(text: str, min_words: int = 8) -> bool:
    """True only when another language clearly dominates the passage.

    Stricter than ``detect_language(text)[0] != "en"``: English that quotes foreign names, places or
    short phrases stays English; short, ambiguous and mixed passages are kept (False) unless the
    foreign part clearly dominates (roughly four fifths of the text or more). Which foreign language
    it is need not be certain (Portuguese vs Spanish), only that it is not English.
    """
    n, ev = _evidence(text or "")
    if n < min_words:
        return False
    top = max(range(1, len(ev)), key=ev.__getitem__)
    foreign, english = ev[top], ev[_EN]
    return (foreign / n >= _FLAG_MIN_SHARE and foreign - english >= _FLAG_MIN_MARGIN
            and english <= _FLAG_EN_RATIO * foreign)
