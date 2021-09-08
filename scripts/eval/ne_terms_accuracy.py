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

# Script for the evaluation of accuracies on Named Entities and terminology.
# Details on the required annotated data can be found in the below paper.
# If using, please cite:
# M. Gaido et al., 2021. Is "moby dick" a Whale or a Bird? Named Entities and Terminology in Speech Translation,
# Proceedings of Empirical Methods for Natural Language Processing 2021 (EMNLP)
import argparse
import spacy


def ne_and_terms(fp):
    tokens = []
    full_entities = []
    while True:
        ln = fp.readline().strip()
        if ln == "":
            break
        items = ln.split("\t")
        if items[2] != "O":
            entity_type = items[2].split("-")[1]
            entity_pos = items[2].split("-")[0]
            tokens.append((items[1], entity_type))
            if entity_pos == "B":
                full_entities.append(([items[1]], entity_type))
            elif entity_pos == "I":
                full_entities[-1][0].append(items[1])
            else:
                raise ValueError("Unrecognized position {} in \"{}\"".format(entity_pos, ln))
    return tokens, full_entities


def full_entity_index(full_entity, hypothesis):
    tokens_to_match = len(full_entity)
    for i in range(len(hypothesis) - tokens_to_match):
        if hypothesis[i:i+tokens_to_match] == full_entity:
            return i
    return -1


def scores_by_type(in_f, tsv_reference, tokenizer):
    entity_items_scores = {}
    full_entities_scores = {}
    with open(in_f) as i_f, open(tsv_reference) as r_f:
        for i_line in i_f:
            reference_tokens, reference_entities = ne_and_terms(r_f)
            tokenized = [str(tok) for tok in tokenizer(i_line)]
            lowercase_tokenized = [tok.lower() for tok in tokenized]

            tokenized_clone = tokenized.copy()
            lowercase_tokenized_clone = lowercase_tokenized.copy()

            for token, entity_type in reference_tokens:
                if entity_type not in entity_items_scores:
                    entity_items_scores[entity_type] = {
                        "found": 0, "total": 0, "ci_found": 0}
                entity_items_scores[entity_type]["total"] += 1
                if token in tokenized:
                    tokenized.remove(token)
                    entity_items_scores[entity_type]["found"] += 1
                if token.lower() in lowercase_tokenized:
                    lowercase_tokenized.remove(token.lower())
                    entity_items_scores[entity_type]["ci_found"] += 1

            for entity, entity_type in reference_entities:
                if entity_type not in full_entities_scores:
                    full_entities_scores[entity_type] = {
                        "found": 0, "total": 0, "ci_found": 0}
                full_entities_scores[entity_type]["total"] += 1
                idx = full_entity_index(entity, tokenized_clone)
                if idx >= 0:
                    del tokenized_clone[idx:idx+len(entity)]
                    full_entities_scores[entity_type]["found"] += 1
                idx_lower = full_entity_index(
                    [t.lower() for t in entity], lowercase_tokenized_clone)
                if idx_lower >= 0:
                    del lowercase_tokenized_clone[idx:idx+len(entity)]
                    full_entities_scores[entity_type]["ci_found"] += 1

    return entity_items_scores, full_entities_scores


def print_scores(out_scores, score_type, print_latex=False):
    categories = list(out_scores.keys())
    categories.sort()
    print("{} Scores".format(score_type))
    print("Category\tTotal\tFound\tCase Insensitive Found\tAccuracy\tCase Insensitive Accuracy")
    print("------------------------------------------------------" * 2)
    printed_scores = {}
    for c in categories:
        printed_scores[c] = {}
        printed_scores[c]["Total"] = out_scores[c]["total"]
        printed_scores[c]["Found"] = out_scores[c]["found"]
        printed_scores[c]["Case Insensitive Found"] = out_scores[c]["ci_found"]
        printed_scores[c]["Accuracy"] = float(out_scores[c]["found"]) / out_scores[c]["total"]
        printed_scores[c]["Case Insensitive Accuracy"] =\
            float(out_scores[c]["ci_found"]) / out_scores[c]["total"]
        print("{}\t{}\t{}\t{}\t{}\t{}".format(
            c, printed_scores[c]["Total"], printed_scores[c]["Found"], printed_scores[c]["Case Insensitive Found"],
            printed_scores[c]["Accuracy"], printed_scores[c]["Case Insensitive Accuracy"]))
    if print_latex:
        import pandas as pd
        df = pd.DataFrame.from_dict(printed_scores, orient='index')
        print(df.to_latex())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, type=str, metavar='FILE',
                        help='Input file to be used to compute accuracies (it must be detokenized).')
    parser.add_argument('--tsv-ref', required=True, type=str, metavar='REFERENCE',
                        help='TSV with NE and terms definition file.')
    parser.add_argument('--lang', required=True, type=str, metavar='LANG',
                        help='Target language.')
    parser.add_argument('--debug', required=False, action='store_true', default=False)
    parser.add_argument("--print-latex", required=False, action='store_true', default=False)

    args = parser.parse_args()

    LANG_MAP = {
        "en": "en_core_web_lg",
        "es": "es_core_news_lg",
        "fr": "fr_core_news_lg",
        "it": "it_core_news_lg"}

    nlp = spacy.load(LANG_MAP[args.lang], disable=['parser', 'ner'])

    items_scores, entities_scores = scores_by_type(args.input, args.tsv_ref, nlp)
    print_scores(items_scores, "Items", print_latex=args.print_latex)
    print_scores(entities_scores, "Full Entities", print_latex=args.print_latex)
