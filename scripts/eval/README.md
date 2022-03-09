# Evaluation Scripts

This folder contains scripts for the evaluation of system outputs
on different aspects:

 - `mustshe_acc`: computes the _term coverage_ and _gender accuracy_
   for the MuST-SHE benchmark, as described in

M. Gaido, B. Savoldi et al., "Breeding Gender-aware Direct Speech Translation Systems", COLING 2020

 - `ne_terms_accuracy.py`: computes the NE and terminology accuracy
   for the NEuRoparl-ST benchmark. For further information, please refer to

M. Gaido et al., "Is "moby dick" a Whale or a Bird? Named Entities and Terminology in Speech Translation", EMNLP 2021

 - `mustshe_acc_fulltable.py`, `accuracy_from_fulltable.py`:
   compute scores per each POS for MuST-SHE, possibly combined with other dimensions.
 - `mustshe_agr_fulltable.py`, `agreement_from_fulltable.py`:
   compute scores for gender agreement for MuST-SHE.

B. Savoldi et al., "Under the Morphosyntactic Lens: A Multifaceted Evaluation of Gender Bias in Speech Translation", ACL 2022


## MuST-SHE Agreement and POS Accuracy Scripts Usage

In this section we present the steps needed to replicate the evaluation performed in the paper:

B. Savoldi et al., "Under the Morphosyntactic Lens: A Multifaceted Evaluation of Gender Bias in Speech Translation", ACL 2022

The steps are:
 - generate the outputs of interest (e.g. of different systems) in the same folder.
   The format should be a txt file with one line per sample;
 - compute the word/agreement-level statistics for all the systems using either `mustshe_acc_fulltable.py` (for POS)
   or `mustshe_agr_fulltable.py` (for gender agreement);
 - generate a JSON file containing the metrics definitions (filters, dimensions, ...).
   The JSON files used in the paper are present in the folder `config_files`, which also
   contains a README with instructions on how to create a new config file.
 - compute the metrics using either `accuracy_from_fulltable.py` (for POS) or
   `agreement_from_fulltable.py` (for agreement), using the prepared JSON config file
   and the previously generated TSV with sentence level statistics.

### Example Script for POS

```bash
#!/bin/bash

python mustshe_acc_fulltable.py \
   --input-prefix /path/to/sys/outputs/sys*.txt \
   --tsv-definition MuST-SHE.v1.2.it.tsv \
   --pos-definition MuST-SHE_v1.2_POS.it.tsv \
   --output /path/to/sys/outputs/all_stats_pos.tsv

python accuracy_from_fulltable.py \
   --fulltable /path/to/sys/outputs/all_stats_pos.tsv \
   --config config_files/config.word-level.json \
   --output /path/to/sys/outputs/pos_scores.tsv
   
cat /path/to/sys/outputs/pos_scores.tsv
```

### Example Script for Gender Agreement

```bash
#!/bin/bash

python mustshe_agr_fulltable.py \
   --input-prefix /path/to/sys/outputs/sys*.txt \
   --tsv-definition MuST-SHE.v1.2.it.tsv \
   --agr-definition MuST-SHE_v1.2_AGR.it.tsv \
   --output /path/to/sys/outputs/all_stats_agreement.tsv

python agreement_from_fulltable.py \
   --fulltable /path/to/sys/outputs/all_stats_agreement.tsv \
   --config config_files/config.agr-level.json \
   --output /path/to/sys/outputs/agr_scores.tsv
   
cat /path/to/sys/outputs/agr_scores.tsv
```


