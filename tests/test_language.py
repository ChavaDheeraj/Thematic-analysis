from thematicml.data.language import detect_language, is_non_english, language_scores

EN = "I travelled alone through Portugal and Spain last year and felt safe most of the time, even at night."
PT = "Observou-se que o medo é um sentimento experienciado pelas mulheres durante as viagens realizadas no Brasil."
ES = "Se trata de un estudio exploratorio con la aplicación de un cuestionario en línea a mujeres que viajan solas."
FR = "Elle a voyagé seule pendant trois mois et elle dit que ce voyage a changé sa vie pour toujours."
DE = "Sie ist allein durch Südamerika gereist und hat dabei viele Menschen kennengelernt, die ihr geholfen haben."


def test_detects_supported_languages():
    assert detect_language(EN)[0] == "en"
    assert detect_language(PT)[0] == "pt"
    assert detect_language(ES)[0] == "es"
    assert detect_language(FR)[0] == "fr"
    assert detect_language(DE)[0] == "de"


def test_short_or_name_heavy_text_is_undetermined():
    assert detect_language("Muito obrigada!") == ("und", 0.0)
    assert detect_language("São Paulo, Rio de Janeiro, Salvador, Fortaleza, Recife, Natal, Maceió")[0] in ("und", "pt")
    assert not is_non_english("Muito obrigada!")


def test_english_with_foreign_names_stays_english():
    text = "We stayed in a pousada near Praia dos Carneiros and the owner, Dona Maria, cooked feijoada for us every Sunday."
    assert detect_language(text)[0] == "en"
    assert not is_non_english(text)
    assert not is_non_english(EN)


def test_non_english_flag_and_explanation():
    assert is_non_english(PT) and is_non_english(ES)
    scores = language_scores(PT)
    assert scores["pt"] > scores["en"] and scores["n_words"] >= 8
    # a mostly English passage quoting a short Portuguese phrase is kept
    assert not is_non_english("She told me the locals kept saying 'tudo bem' to her, and that it made her feel welcome every day.")
