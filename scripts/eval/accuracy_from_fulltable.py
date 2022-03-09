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
# This script computes metrics from a TSV with sentence level statistics produced with mustshe_acc_fulltable.py.
# Details on the required annotated data can be found in the below paper.
# If using, please cite:
# B. Savoldi et al., 2022. Under the Morphosyntactic Lens:
# A Multifaceted Evaluation of Gender Bias in Speech Translation,
# Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics
import argparse
from collections import defaultdict
import csv
import json


def read_configs(config_file):
    # Reads the JSON file containing the metrics definitions
    with open(config_file) as f:
        configs = json.load(f)
    return configs["metrics"]


def check_filters(term, filters):
    # Returns True if the given term with all the dimensions
    # satisfies all the conditions defined by filters
    for f in filters:
        if not (f["max_val"] >= term[f["dimension"]] >= f["min_val"]):
            return False
    return True


def write_rows_to_tsv(out_f, headers, rows):
    # Writes the header and rows to the given file as a TSV
    with open(out_f, 'w') as f_w:
        writer = csv.DictWriter(f_w, headers, delimiter='\t')
        writer.writeheader()
        writer.writerows(rows)


def accuracy_scores(in_f, metrics):
    results = {}
    with open(in_f) as i_f:
        tsv_reader = csv.DictReader(i_f, delimiter='\t')
        for term in tsv_reader:
            # Metrics are computed for all systems
            # System names are extracted from the *_found columns
            systems = [h[:-6] for h in term.keys() if h.endswith("_found")]
            for m_i, m in enumerate(metrics):
                if "filters" not in m or check_filters(term, m["filters"]):
                    key = "-".join([term[d] for d in m["dimensions"]])
                    if key not in results:
                        results[key] = defaultdict(lambda: 0)
                        results[key]["order"] = m_i
                    results[key]["num_terms"] += 1
                    for h in ["found", "found_correct", "found_wrong"]:
                        for s in systems:
                            results[key][s + "_" + h] += int(term[s + "_" + h])
    return results


def write_sentence_acc(out_f, all_stats):
    headers = ["metric", "num_terms"]
    rows = []
    metrics = sorted(list(all_stats.keys()), key=lambda x: (all_stats[x]["order"], x))
    systems = [h[:-6] for h in all_stats[metrics[0]].keys() if h.endswith("_found")]
    for s in systems:
        for scores_h in ["found", "found_correct", "found_wrong", "term_coverage", "gender_accuracy"]:
            headers.append(s + "_" + scores_h)
    for m in metrics:
        r = dict(all_stats[m])
        r["metric"] = m
        del r["order"]
        for s in systems:
            r[s + "_term_coverage"] = float(r[s + "_found"]) / r["num_terms"]
            tot_for_accuracy = r[s + "_found_correct"] + r[s + "_found_wrong"]
            if tot_for_accuracy > 0:
                r[s + "_gender_accuracy"] = float(r[s + "_found_correct"]) / tot_for_accuracy
            else:
                r[s + "_gender_accuracy"] = 0.0
        rows.append(r)
    write_rows_to_tsv(out_f, headers, rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fulltable', required=True, type=str, metavar='FILE',
                        help='Input file to be used to compute accuracies (they must be tokenized).')
    parser.add_argument('--config', required=True, type=str, metavar='FILE',
                        help='JSON file with epochs configurations.')
    parser.add_argument('--output', required=True, default=None, type=str, metavar='FILE',
                        help='Accuracies are written into this file.')

    args = parser.parse_args()

    configs = read_configs(args.config)
    all_epochs_stats = accuracy_scores(args.fulltable, configs)
    write_sentence_acc(args.output, all_epochs_stats)
