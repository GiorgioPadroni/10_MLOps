# 10_MLOps

- Il file *fase_1.py* contiene la valutazione del modello FastText (il cui wrapper di classe è implementato nel file "*models/FastText.py*") pre-addestrato sul dataset *tweet eval*, pubblico su hugging face.

- Ogni qualvolta viene eseguito un push di nuove porzioni di dataset, il modello viene finetunato sul nuovo dataset, viene pushato il modello su hugging face ed infine viene eseguito un test di integrazione (file "*CI_CD/tests/test_model.py*").

