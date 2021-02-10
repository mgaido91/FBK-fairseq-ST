#!/usr/bin/python3
# Scripts for the evaluation of accuracies on MuST-SHE.
# If using, please consider citing:
# - M. Gaido, B. Savoldi et al., "Breeding Gender-aware Direct Speech Translation Systems", COLING 2020
# Version: 1.0
import argparse
import csv


def sentence_level_scores(in_f, tsv_f):
    sentences = []
    with open(in_f) as i_f, open(tsv_f) as t_f:
        tsv_reader = csv.DictReader(t_f, delimiter='\t')
        for (i_line, terms_f) in zip(i_f, tsv_reader):
            sentence_correct = 0
            sentence_wrong = 0
            sentence_found = 0
            gender_marked_terms = terms_f['GENDERTERMS'].strip().lower().split(";")
            generated_terms = i_line.strip().lower().split()
            for t in gender_marked_terms:
                term = t.split(" ")
                found = False
                correct_term = term[0]
                wrong_term = term[1]
                try:
                    pos_found = generated_terms.index(correct_term)
                    # Avoid re-matching the same term two times
                    del generated_terms[pos_found]
                    sentence_correct += 1
                    found = True
                except ValueError:
                    pass
                try:
                    pos_found = generated_terms.index(wrong_term)
                    # Avoid re-matching the same term two times
                    del generated_terms[pos_found]
                    sentence_wrong += 1
                    found = True
                except ValueError:
                    pass
                if found:
                    sentence_found += 1
            sentences.append({
                "num_terms": len(gender_marked_terms),
                "num_terms_found": sentence_found,
                "num_correct": sentence_correct,
                "num_wrong": sentence_wrong})

    return sentences


def write_sentence_acc(out_f, sentence_scores):
    with open(out_f, 'w') as f_w:
        writer = csv.DictWriter(
            f_w, ["num_terms", "num_terms_found", "num_correct", "num_wrong"], delimiter='\t')
        writer.writeheader()
        writer.writerows(sentence_scores)


def global_scores(sentence_scores, tsv_f, debug=False):
    i = 0
    category_buffers = {}
    with open(tsv_f, 'r') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for line in reader:
            category = line["CATEGORY"]
            if category not in category_buffers:
                category_buffers[category] = {"num_terms": 0, "num_correct": 0, "num_wrong": 0, "num_terms_found": 0}
            category_buffers[category]["num_terms"] += sentence_scores[i]["num_terms"]
            category_buffers[category]["num_terms_found"] += sentence_scores[i]["num_terms_found"]
            category_buffers[category]["num_correct"] += sentence_scores[i]["num_correct"]
            category_buffers[category]["num_wrong"] += sentence_scores[i]["num_wrong"]
            i += 1
    if debug:
        print("Evaluated {} sentences...".format(i))
    overall_scores = {}
    tot_terms = 0
    tot_found = 0
    tot_correct = 0
    tot_wrong = 0
    for c in category_buffers:
        term_cov = float(category_buffers[c]["num_terms_found"]) / category_buffers[c]["num_terms"]
        if category_buffers[c]["num_terms_found"] > 0:
            gender_acc = float(category_buffers[c]["num_correct"]) / \
                         (category_buffers[c]["num_correct"] + category_buffers[c]["num_wrong"])
        else:
            gender_acc = 0.0
        overall_scores[c] = {"term_coverage": term_cov, "gender_accuracy": gender_acc}
        if debug:
            print("Category {}: all->{}, found->{}; correct->{}; wrong->{}".format(
                c, category_buffers[c]["num_terms"],
                category_buffers[c]["num_terms_found"],
                category_buffers[c]["num_correct"],
                category_buffers[c]["num_wrong"]))
        tot_terms += category_buffers[c]["num_terms"]
        tot_found += category_buffers[c]["num_terms_found"]
        tot_correct += category_buffers[c]["num_correct"]
        tot_wrong += category_buffers[c]["num_wrong"]
    if debug:
        print("Global: all->{}; found->{}; correct->{}; wrong->{}".format(
            tot_terms, tot_found, tot_correct, tot_wrong))
    overall_scores["Global"] = {
        "term_coverage": tot_found / tot_terms,
        "gender_accuracy": tot_correct / (tot_correct + tot_wrong)}
    return overall_scores


def print_scores(out_scores, print_latex=False):
    categories = list(out_scores.keys())
    categories.sort()
    print("Category\tTerm Coverage\tGender Accuracy")
    print("-------------------------------------------------")
    for c in categories:
        if c == "Global":
            print("-------------------------------------------------")
        print("{}\t{}\t{}".format(c, out_scores[c]["term_coverage"], out_scores[c]["gender_accuracy"]))
        if c == "Global":
            print("-------------------------------------------------")
    if print_latex:
        import pandas as pd
        df = pd.DataFrame.from_dict(out_scores, orient='index')
        print(df.to_latex())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, type=str, metavar='FILE',
                        help='Input file to be used to compute accuracies (it must be tokenized).')
    parser.add_argument('--tsv-definition', required=True, type=str, metavar='FILE',
                        help='TSV MuST-SHE definitions file.')
    parser.add_argument('--sentence-acc', required=False, default=None, type=str, metavar='FILE',
                        help='If set, sentence level accuracies are written into this file.')
    parser.add_argument('--debug', required=False, action='store_true', default=False)
    parser.add_argument("--print-latex", required=False, action='store_true', default=False)

    args = parser.parse_args()

    sl_scores = sentence_level_scores(args.input, args.tsv_definition)
    if args.sentence_acc:
        write_sentence_acc(args.sentence_acc, sl_scores)
    scores = global_scores(sl_scores, args.tsv_definition, args.debug)
    print_scores(scores, args.print_latex)
