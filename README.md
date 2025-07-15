# 10_MLOps

- Il file *fase_1.py* contiene la valutazione del modello FastText (il cui wrapper di classe è implementato nel file "*models/FastText.py*") pre-addestrato sul dataset *tweet eval*, pubblico su hugging face.

- Ogni qualvolta viene eseguito un push su main, il modello viene finetunato sul dataset scelto, in questo caso nuovamente il *tweet eval* dataset -a scopo dimostrativo- viene pushato il modello su hugging face ed infine viene eseguito un test di integrazione (file "*CI_CD/tests/test_model.py*"). Ci tengo a precisare che il file ha solo scopo dimostrativo, inoltre per velocizzare l'allenamento viene utilizzato un piccolissimo subset del dataset.

