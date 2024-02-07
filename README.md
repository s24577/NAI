# NAI
## Opis problemu i podejście
Zadaniem jest klasyfikacja sentymentu tekstu przy użyciu trzech różnych modeli: DistilBERT(lxyuan/distilbert-base-multilingual-cased-sentiments-student), XDistil(hakonmh/sentiment-xdistil-uncased) oraz RoBERTa(cardiffnlp/twitter-roberta-base-sentiment-latest). Do dyspozycji mamy zbiór danych - ihassan1/auditor-sentiment. Następnie wszystkie modele porównywamy badając ich końcowe wyniki metryk.

Podejście do Rozwiązania - wykorzystujemy najpopularniejsze modele z Hugging Face na podstawie najczęstszych pobrań w ciągu miesiąca.
## Modele
### Model 1
Skrypt DistilbertBase.py wykorzystuje model DistilBERT do analizy sentymentu tekstu. Wczytuje zestaw danych, generuje sentyment tekstu i ocenia wydajność, korzystając z precyzji, czułości i wyniku F1.
### Model 2
Skrypt SentimentXDistil.py wykorzystuje model XDistil do analizy sentymentu tekstu. Wczytuje zestaw danych, generuje sentyment tekstu i ocenia wydajność, korzystając z precyzji, czułości i wyniku F1.
### Model 3
Skrypt TwitterRobertBase.py wykorzystuje model RoBERTa do analizy sentymentu tekstu. Wczytuje zestaw danych, generuje sentyment tekstu i ocenia wydajność, korzystając z precyzji, czułości i wyniku F1.
