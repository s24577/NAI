# Projekt NAI
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

## Jak zacząć
### 1. Sklonuj repozytorium:
```bash
git clone https://github.com/twoja-nazwa-uzytkownika/NAI.git
cd NAI
```
### 2. Zainstaluj zależności:
```bash
pip install -r requirements.txt
```
### 3. Uruchom odpowiedni skrypt modelu:
```bash
python3 main.py
```
### 4. Zbiór danych
Upewnij się, że twój zbiór danych jest w odpowiednim formacie csv i zawiera niezbędne kolumny dla każdego skryptu.

## Współtwórcy
Witkoria Kostrzewa (s24548),
Grzegorz Broszko (s24577),
Jakub Zawadzki (s23214)
