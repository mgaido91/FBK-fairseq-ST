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


def agr_scores(in_f, metrics):
    results = {}
    with open(in_f) as i_f:
        tsv_reader = csv.DictReader(i_f, delimiter='\t')
        for term in tsv_reader:
            # Metrics are computed for all systems
            # System names are extracted from the *__no_agreement columns
            systems = [h[:-13] for h in term.keys() if h.endswith("_no_agreement")]
            for m_i, m in enumerate(metrics):
                if "filters" not in m or check_filters(term, m["filters"]):
                    key = "-".join([term[d] for d in m["dimensions"]])
                    if key not in results:
                        results[key] = defaultdict(lambda: 0)
                        results[key]["order"] = m_i
                    results[key]["num_agrs"] += 1
                    for h in ["agreement_correct", "agreement_wrong", "out_of_coverage", "no_agreement"]:
                        for s in systems:
                            results[key][s + "_" + h] += int(eval(term[s + "_" + h]))
    return results


def write_sentence_acc(out_f, all_stats):
    headers = ["metric", "num_agrs"]
    rows = []
    metrics = sorted(list(all_stats.keys()), key=lambda x: (all_stats[x]["order"], x))
    systems = [h[:-13] for h in all_stats[metrics[0]].keys() if h.endswith("_no_agreement")]
    for s in systems:
        for scores_h in ["agreement_correct", "agreement_wrong", "out_of_coverage", "no_agreement", "in_coverage"]:
            headers.append(s + "_" + scores_h)
            headers.append(s + "_" + scores_h + "_percent")
    for m in metrics:
        r = dict(all_stats[m])
        r["metric"] = m
        del r["order"]
        for s in systems:
            r[s + "_out_of_coverage_percent"] = float(r[s + "_out_of_coverage"]) / r["num_agrs"]
            r[s + "_in_coverage"] = float(r["num_agrs"] - r[s + "_out_of_coverage"])
            r[s + "_in_coverage_percent"] = r[s + "_in_coverage"] / r["num_agrs"]
            for scores_h in ["agreement_correct", "agreement_wrong", "no_agreement"]:
                covered_agrs = (r["num_agrs"] - r[s + "_out_of_coverage"])
                if covered_agrs == 0:
                    r[s + "_" + scores_h + "_percent"] = 0.0
                else:
                    r[s + "_" + scores_h + "_percent"] = float(r[s + "_" + scores_h]) / covered_agrs
        rows.append(r)
    write_rows_to_tsv(out_f, headers, rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fulltable', required=True, type=str, metavar='FILE',
                        help='Input file to be used to compute agreements (they must be tokenized).')
    parser.add_argument('--config', required=True, type=str, metavar='FILE',
                        help='JSON file with epochs configurations.')
    parser.add_argument('--output', required=True, default=None, type=str, metavar='FILE',
                        help='Accuracies are written into this file.')

    args = parser.parse_args()

    configs = read_configs(args.config)
    all_epochs_stats = agr_scores(args.fulltable, configs)
    write_sentence_acc(args.output, all_epochs_stats)
