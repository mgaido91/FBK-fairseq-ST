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

# Script for the evaluation of gender accuracy divided according to the POS tag and/or other dimensions on MuST-SHE.
# This script generates a TSV with sentence level statistics that are used to
# compute metrics with the script accuracy_from_fulltable.py.
# Details on the required annotated data can be found in the below paper.
# If using, please cite:
# B. Savoldi et al., 2022. Under the Morphosyntactic Lens:
# A Multifaceted Evaluation of Gender Bias in Speech Translation,
# Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics
import argparse
import csv
import glob
import os


def sentence_level_scores(in_f, tsv_f, pos_definitions):
    # Computes sentence-level statistics, adding dimensions related to POS tags (pos and class)
    # and the kind of phenomenon (category, gender, speaker gender)
    terms_stats = []
    with open(in_f) as i_f, open(tsv_f) as t_f:
        tsv_reader = csv.DictReader(t_f, delimiter='\t')
        for (i_line, terms_f) in zip(i_f, tsv_reader):
            gender_marked_terms = terms_f['GENDERTERMS'].strip().lower().split(";")
            terms_pos = pos_definitions[terms_f['ID']].lower().split(";")
            generated_terms = i_line.strip().lower().split()
            for t, pos in zip(gender_marked_terms, terms_pos):
                term = t.split(" ")
                found_correct = False
                found_wrong = False
                correct_term = term[0]
                wrong_term = term[1]
                try:
                    pos_found = generated_terms.index(correct_term)
                    # Avoid re-matching the same term two times
                    del generated_terms[pos_found]
                    found_correct = True
                except ValueError:
                    pass
                try:
                    pos_found = generated_terms.index(wrong_term)
                    # Avoid re-matching the same term two times
                    del generated_terms[pos_found]
                    found_wrong = True
                except ValueError:
                    pass

                terms_stats.append({
                    "found": int(found_wrong or found_correct),
                    "sentence_id": terms_f['ID'],
                    "found_correct": int(found_correct),
                    "found_wrong": int(found_wrong),
                    "pos": pos,
                    "class": "closed" if pos.lower() in ["art/prep", "pronoun", "adj-determiner"] else "open",
                    "speaker_gender": terms_f["GENDER"],
                    "gender": terms_f["CATEGORY"][1],
                    "category": terms_f["CATEGORY"],
                    "correct": correct_term,
                    "wrong": wrong_term})

    return terms_stats


def write_sentence_acc(out_f, all_stats):
    # Writes the statistics collected for all the files into TSV files.
    headers = ["sentence_id", "correct", "wrong", "speaker_gender", "category", "pos", "class", "gender"]
    rows = []
    for e, stats_terms in all_stats:
        for i, stats in enumerate(stats_terms):
            if len(rows) <= i:
                rows.append({
                        "sentence_id": stats['sentence_id'],
                        "pos": stats['pos'],
                        "speaker_gender": stats["speaker_gender"],
                        "category": stats["category"],
                        "correct": stats["correct"],
                        "wrong": stats["wrong"],
                        "gender": stats["gender"],
                        "class": stats["class"]})
            for h in ["found", "found_correct", "found_wrong"]:
                if i == 0:
                    headers.append(e + "_" + h)
                rows[i][e + "_" + h] = stats[h]
    with open(out_f, 'w') as f_w:
        writer = csv.DictWriter(f_w, headers, delimiter='\t')
        writer.writeheader()
        writer.writerows(rows)


def read_pos_definition(tsv_pos_f):
    # Reads the POS definitions in tsv_pos_f and returns them in a dictionary
    with open(tsv_pos_f) as tf:
        tsv_reader = csv.DictReader(tf, delimiter='\t')
        pos_definitions = {}
        for line in tsv_reader:
            pos_definitions[line["ID"]] = line["POS"].strip()
    return pos_definitions


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-prefix', required=True, type=str, metavar='FILE',
                        help='Input prefix of files to be used to compute accuracies (they must be tokenized).')
    parser.add_argument('--tsv-definition', required=True, type=str, metavar='FILE',
                        help='TSV MuST-SHE definitions file.')
    parser.add_argument('--pos-definition', required=True, type=str, metavar='FILE',
                        help='TSV file containing the POS definitions.')
    parser.add_argument('--output', required=True, default=None, type=str, metavar='FILE',
                        help='Sentence level accuracies are written into this file.')

    args = parser.parse_args()

    all_epochs_stats = []
    pos_definitions = read_pos_definition(args.pos_definition)
    for input_f in glob.glob(args.input_prefix + "*"):
        fname = os.path.basename(input_f)
        terms_stats = sentence_level_scores(
            input_f, args.tsv_definition, pos_definitions)
        all_epochs_stats.append((fname, terms_stats))
    write_sentence_acc(args.output, all_epochs_stats)
