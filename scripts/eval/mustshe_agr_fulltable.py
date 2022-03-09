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

# Script for the evaluation of gender agreements on MuST-SHE.
# This script generates a TSV with sentence level statistics that are used
# compute metrics with the script agreement_from_fulltable.py.
# Details on the required annotated data can be found in the below paper.
# If using, please cite:
# B. Savoldi et al., 2022. Under the Morphosyntactic Lens:
# A Multifaceted Evaluation of Gender Bias in Speech Translation,
# Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics
import argparse
import csv
import glob
import os


def sentences_by_id(tsv_f, in_f):
    # Reads the MuST-SHE TSV definition file and
    # a txt file containing the output generated by a system for MuST-SHE.
    # It returns a dictionary that maps the MuST-SHE sentence id to the category (1F, 1M, 2F, 2M)
    # and the system output.
    sentences = {}
    with open(in_f) as i_f, open(tsv_f) as t_f:
        tsv_reader = csv.DictReader(t_f, delimiter='\t')
        for (i_line, terms_f) in zip(i_f, tsv_reader):
            sentences[terms_f['ID']] = {"sys_out": i_line.strip(), "CATEGORY": terms_f['CATEGORY']}
    return sentences


def agreement_stats(agr_definition, sentences):
    # Computes sentence-level statistics, trying to disambiguate terms that are found for both genders.
    # Returns an iterator with a dictionary for each line (sample) of the MuST-SHE benchmark.
    with open(agr_definition) as tsv_f:
        tsv_reader = csv.DictReader(tsv_f, delimiter='\t')
        for agr_line in tsv_reader:
            i_line = sentences[agr_line['ID']]["sys_out"]
            gender_marked_terms = agr_line['AGR_TERMS'].strip().lower().split(";")
            generated_terms = i_line.strip().lower().split()
            terms_found = []
            not_found_list = []
            # For each term, look for the index(es) at which it is present (if any) in the system output
            # and at which the corresponding wrong term is found
            for t in gender_marked_terms:
                term = t.split(" ")
                correct_term = term[0]
                wrong_term = term[1]
                correct_indexes = [i for i, x in enumerate(generated_terms) if x == correct_term]
                wrong_indexes = [i for i, x in enumerate(generated_terms) if x == wrong_term]
                terms_found.append({"correct": correct_indexes, "wrong": wrong_indexes})
                if len(correct_indexes) == 0 and len(wrong_indexes) == 0:
                    not_found_list.append(correct_term)

            stats = {"correct": 0, "wrong": 0, "both": 0, "not_found": 0}
            to_disambiguate = []
            fixed_items = []
            for t in terms_found:
                if len(t["correct"]) > 0 and len(t["wrong"]) > 0:
                    stats["both"] += 1
                    to_disambiguate.append(t)
                elif len(t["correct"]) > 0 and len(t["wrong"]) == 0:
                    stats["correct"] += 1
                    if len(t["correct"]) == 1:
                        fixed_items.append(t["correct"][0])
                elif len(t["correct"]) == 0 and len(t["wrong"]) > 0:
                    stats["wrong"] += 1
                    if len(t["wrong"]) == 1:
                        fixed_items.append(t["wrong"][0])
                else:
                    stats["not_found"] += 1
            stats["correct_disambiguated"] = stats["correct"]
            stats["wrong_disambiguated"] = stats["wrong"]
            stats["not_disambiguated"] = 0

            # Try to disambiguate those terms that are found together with the corresponding wrong term.
            # To disambiguate, we look for the closest occurrence to agreement terms that are not ambiguous.
            # If no element is not ambiguous, the disambiguation fails.
            if len(to_disambiguate) > 0:
                if len(fixed_items) == 0:
                    stats["not_disambiguated"] = len(to_disambiguate)
                else:
                    for t in to_disambiguate:
                        correct_min_dist = min([sum(abs(tc - fp) for fp in fixed_items) for tc in t["correct"]])
                        wrong_min_dist = min([sum(abs(tw - fp) for fp in fixed_items) for tw in t["wrong"]])
                        if correct_min_dist <= wrong_min_dist:
                            stats["correct_disambiguated"] += 1
                        else:
                            stats["wrong_disambiguated"] += 1

            stats["agreement_correct"] = stats["wrong_disambiguated"] == 0 and \
                stats["not_found"] + stats["not_disambiguated"] == 0
            stats["agreement_wrong"] = stats["correct_disambiguated"] == 0 and \
                stats["not_found"] + stats["not_disambiguated"] == 0
            stats["out_of_coverage"] = stats["not_found"] + stats["not_disambiguated"] > 0
            stats["no_agreement"] = not (stats["out_of_coverage"] or stats["agreement_correct"]
                                         or stats["agreement_wrong"])
            for f in ["ID", "AGR_KIND", "AGR_TERMS", "AGR_IDS"]:
                stats[f] = agr_line[f]
            stats["CATEGORY_TYPE"] = sentences[agr_line['ID']]["CATEGORY"][0]
            stats["CATEGORY_GENDER"] = sentences[agr_line['ID']]["CATEGORY"][1]
            yield stats


def write_sentence_acc(out_f, all_stats):
    # Writes the statistics collected for all the files into TSV files.
    headers = ["ID", "AGR_KIND", "AGR_TERMS", "AGR_IDS", "CATEGORY_TYPE", "CATEGORY_GENDER"]
    rows = []
    for e, stats_terms in all_stats:
        for i, stats in enumerate(stats_terms):
            if len(rows) <= i:
                rows.append({
                        "ID": stats['ID'],
                        "AGR_KIND": stats['AGR_KIND'],
                        "AGR_TERMS": stats["AGR_TERMS"],
                        "AGR_IDS": stats["AGR_IDS"],
                        "CATEGORY_TYPE": stats["CATEGORY_TYPE"],
                        "CATEGORY_GENDER": stats["CATEGORY_GENDER"]
                })
            for h in ["not_found", "correct_disambiguated", "wrong_disambiguated", "not_disambiguated",
                      "agreement_correct", "agreement_wrong", "out_of_coverage", "no_agreement"]:
                if i == 0:
                    headers.append(e + "_" + h)
                rows[i][e + "_" + h] = stats[h]
    with open(out_f, 'w') as f_w:
        writer = csv.DictWriter(f_w, headers, delimiter='\t')
        writer.writeheader()
        writer.writerows(rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-prefix', required=True, type=str, metavar='FILE',
                        help='Input prefix of files to be used to compute accuracies (they must be tokenized).')
    parser.add_argument('--tsv-definition', required=True, type=str, metavar='FILE',
                        help='TSV MuST-SHE definitions file.')
    parser.add_argument('--output', required=True, default=None, type=str, metavar='FILE',
                        help='Sentence level accuracies are written into this file.')
    parser.add_argument('--agr-definition', required=True, type=str, metavar='FILE',
                        help='TSV file containing the agreement definitions.')

    args = parser.parse_args()

    all_epochs_stats = []
    for input_f in glob.glob(args.input_prefix + "*"):
        fname = os.path.basename(input_f)
        sentences = sentences_by_id(args.tsv_definition, input_f)
        sentence_level_stats = list(agreement_stats(args.agr_definition, sentences))
        all_epochs_stats.append((fname, sentence_level_stats))
    write_sentence_acc(args.output, all_epochs_stats)
