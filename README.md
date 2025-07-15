# 10_MLOps

- Il file *fase_1.py* contiene la valutazione del modello FastText (il cui wrapper di classe è implementato nel file "*models/FastText.py*") pre-addestrato sul dataset *tweet eval*, pubblico su hugging face.

- Ogni qualvolta viene eseguito un push di modifiche all'interno della cartella data (che siano esse per train o per test), il modello viene finetunato sul nuovo dataset, viene pushato il modello su hugging face, viene testato il modello e salvate le metriche come artefatto ed infine viene eseguito un test di integrazione (file "*CI_CD/tests/test_model.py*").

- Dopo ogni training basta scaricare l'artefatto che contiene le metriche di valutazione del modello per capire il livello di affidabilità di esso sul dataset test che si è scelto come sistema di valutazione.