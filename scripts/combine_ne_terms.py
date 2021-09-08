#!/usr/bin/python3
# Copyright 2021 FBK

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License

# The script takes two positional arguments:
# 1. The IOB file containing the NE annotations
# 2. The IOB file containing the terminology annotation
# And it deals with the merge of the two files into a single IOB file
# giving priority to NE when there is a conflict in the annotations and
# recovering from possibly different tokenization of the two files.
# The output is written to stdout, so an example of usage of this script is:
# python combine_ne_terms.py ne.iob.en terms.iob.en > all.iob.en

# If using, please cite:
# M. Gaido et al., 2021. Is "moby dick" a Whale or a Bird? Named Entities and Terminology in Speech Translation,
# Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing (EMNLP)

import sys

ner_detected_fn = sys.argv[1]
term_detected_fn = sys.argv[2]


def select_type(types):
    # It might continue in the next line...
    if types[-1] != "O":
        return types[-1]
    return sorted(types, key=types.count, reverse=True)[0]


NER_BUFFER = []
NER_TYPES_BUFFER = []
term_line = None
prev_type = None
l_idx = 0
with open(ner_detected_fn) as ner_f, open(term_detected_fn) as term_f:
    for ner_line in ner_f:
        ner_items = ner_line.split('\t')
        if len(ner_items) < 3:
            term_line = term_f.readline()
            assert len(term_line.split('\t')) < 3, "Mismatch at line: {} --- {}".format(ner_line, term_line)
            l_idx = 0
            sys.stdout.write(ner_line)
        else:
            assert len(ner_items) == 3
            if len(NER_BUFFER) == 0:
                term_line = term_f.readline()
                term_items = [t.strip() for t in term_line.split("\t")]
            NER_BUFFER.append(ner_items[1])
            ner_term = "".join(NER_BUFFER)
            ner_type = ner_items[2].strip()
            if ner_term == term_items[1]:
                if NER_TYPES_BUFFER:
                    NER_TYPES_BUFFER.append(ner_type)
                    ner_types = [t.split("-")[-1] for t in NER_TYPES_BUFFER]
                    ner_type = select_type(ner_types)
                    if ner_type != "O":
                        if "B" in [t.split("-")[0] for t in NER_TYPES_BUFFER]:
                            ner_type = "B-" + ner_type
                        else:
                            ner_type = "I-" + ner_type

                NER_BUFFER = []
                NER_TYPES_BUFFER = []
            else:
                if len(ner_term) < len(term_items[1]):
                    NER_TYPES_BUFFER.append(ner_type)
                    continue
                else:
                    term_term = term_items[1]
                    term_types_buffer = [term_items[2]]
                    term_ids = []
                    if len(term_items) > 3:
                        term_ids.append(term_items[3])
                    missing_ner_items = False
                    while term_term != ner_term:
                        if len(ner_term) > len(term_term):
                            term_line = term_f.readline()
                            term_items = term_line.split("\t")
                            term_term += term_items[1]
                            term_types_buffer.append(term_items[2].strip())
                            if len(term_items) > 3:
                                term_ids.append(term_items[3].strip())
                        else:
                            missing_ner_items = True
                            break
                    term_types = [t.split("-")[-1] for t in term_types_buffer]
                    term_type = select_type(term_types)
                    if term_type != "O":
                        if "B" in [t.split("-")[0] for t in term_types_buffer]:
                            term_type = "B-" + term_type
                        else:
                            term_type = "I-" + term_type
                        term_items = [term_items[0], term_term, term_type, "".join(term_ids)]
                    else:
                        term_items = [term_items[0], term_term, term_type]
                    if missing_ner_items:
                        continue
                    else:
                        NER_BUFFER = []
                        NER_TYPES_BUFFER = []

            l_idx += 1
            if ner_type.strip() == 'O':
                if term_items[2] == "I-TERM" and prev_type not in ["B-TERM", "I-TERM"]:
                    # Most likely part of a term has been considered as a NE, so ignore it
                    term_items[2] = "O"
                sys.stdout.write("{}\t{}\n".format(l_idx, "\t".join(term_items[1:])))
                prev_type = term_items[2]
            else:
                if ner_type.startswith("I-") and prev_type not in [ner_type, "B-" + ner_type.split("-")[1]]:
                    ner_type = "B-" + ner_type.split("-")[1]
                sys.stdout.write("{}\t{}\t{}\n".format(l_idx, ner_term, ner_type))
                prev_type = ner_type
