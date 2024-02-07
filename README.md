# NAI
## Opis problemu i podejście
Zadaniem naszej aplikacji jest klasyfikacja tekstu na podstawie języka, w jakim jest napisany. Klasyfikacja tekstu według języka jest ważna w wielu dziedzinach, takich jak analiza sentymentu, personalizacja treści, tłumaczenie maszynowe, oraz w pracy nad wielojęzycznymi systemami informatycznymi.

Podejście do Rozwiązania - wykorzystujemy najpopularniejsze modele z Hugging Face na podstawie najczęstszych pobrań w ciągu miesiąca.
Między innymi korzystamy z:

BERT (Bidirectional Encoder Representations from Transformers) - jest przydatny do zadań klasyfikacji tekstu, w tym identyfikacji języka również jest potężnym modelem pre-trenowanym, który potrafi efektywnie reprezentować semantykę tekstu.

FASTTEXT: FastText to lekki model, który dobrze radzi sobie z klasyfikacją tekstu na podstawie języka. Jego siłą leży w szybkości działania i zdolności do obsługi tekstu w formie n-gramów, co może być korzystne w identyfikacji języka.

XLM-RoBERTa: XLM-RoBERTa to model opracowany przez Facebook Research, skoncentrowany na rozszerzaniu zdolności RoBERTa do obsługi wielu języków. Jest silny w identyfikacji języka, a także radzi sobie z różnicami między językami.
