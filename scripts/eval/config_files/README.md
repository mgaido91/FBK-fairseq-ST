# JSON Metrics Definition

The JSON file should contain a `metrics` property, whose value is
an array of the metrics definitions. Each metric definition is a JSON
object with the following fields:

 - **dimensions**:  within this field, you can specify as parameters
   the name of the columns produced by the `mustshe_acc_fulltable.py` or `mustshe_agr_fulltable.py`.
   Within "dimensions", you can indicate more than one dimension in case you want parameters
   to be entangled and get more specific (e.g. you want to see the performance for class  - open vs closed -
   across genders - masculine vs feminine -,  {"dimensions": ["class", "gender"]).
   If left empty, you get the results for all the gender-marked words (e.g. {"dimensions":[]})
 - **filters**: filters serves to specify a specific range of parameters of interest within each dimension.
   It is used with "min_val" and "max_val", which set the limits of your range
   (the range can be in numerical or alphabetical order),
   e.g. "filters": [{"dimension": "category", "min_val": "1F", "max_val": "2M"}]
   considers only categories 1F, 1M, 2F, and 2M of MuST-SHE, excluding 3F/M and 4F/M.
   If you, for instance, want to focus on one single aspect within a dimension,
   for instance only feminine phenomena of the gender dimension (F, M),
   you can set:  "min_val": "F", "max_val": "F".

In this folder you can find complete examples that were used in the paper:

B. Savoldi et al., "Under the Morphosyntactic Lens: A Multifaceted Evaluation of Gender Bias in Speech Translation", ACL 2022
