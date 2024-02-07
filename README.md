# NAI
## Opis problemu i podejście
Zadaniem jest klasyfikacja sentymentu tekstu przy użyciu trzech różnych modeli: DistilBERT(lxyuan/distilbert-base-multilingual-cased-sentiments-student), XDistil(hakonmh/sentiment-xdistil-uncased) oraz RoBERTa(cardiffnlp/twitter-roberta-base-sentiment-latest). Do dyspozycji mamy zbiór danych - ihassan1/auditor-sentiment. Następnie wszystkie modele porównywamy badając ich końcowe wyniki metryk.

Podejście do Rozwiązania - wykorzystujemy najpopularniejsze modele z Hugging Face na podstawie najczęstszych pobrań w ciągu miesiąca.
## Modele
Między innymi korzystamy z:

BERT (Bidirectional Encoder Representations from Transformers) - jest przydatny do zadań klasyfikacji tekstu, w tym identyfikacji języka również jest potężnym modelem pre-trenowanym, który potrafi efektywnie reprezentować semantykę tekstu.

FASTTEXT: FastText to lekki model, który dobrze radzi sobie z klasyfikacją tekstu na podstawie języka. Jego siłą leży w szybkości działania i zdolności do obsługi tekstu w formie n-gramów, co może być korzystne w identyfikacji języka.

XLM-RoBERTa: XLM-RoBERTa to model opracowany przez Facebook Research, skoncentrowany na rozszerzaniu zdolności RoBERTa do obsługi wielu języków. Jest silny w identyfikacji języka, a także radzi sobie z różnicami między językami.
