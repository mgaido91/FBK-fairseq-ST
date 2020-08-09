import logging
import math
from collections import defaultdict
import json
import random
import sys
import string
import yaml

# We don't remove underscores, as gentle doesn't remove them either
PUNCT_REMOVAL_TABLE = str.maketrans(
    string.punctuation.replace('_', '') + '•—♫’–♪”…‘', ' ' * (len(string.punctuation) + 8))
UNCODE_CHARS_TO_CLEAN = str.maketrans('\x80\x94', '  ')
NUM_RETRIES = 50
BLACKLIST = ['ted_1745', 'ted_2780']  # 1745: thai chars; 2780: ???
WORD_SEPARATORS = ['-', ',']

logger = logging.getLogger(__name__)


def match_beginning_word_with_separator(source_sentence_piece, tokenized_words, w_idx, separator):
    if separator in tokenized_words[w_idx]:
        return remove_punctuation(''.join(tokenized_words[:w_idx + 1])).replace(' ', '') == \
                remove_punctuation(source_sentence_piece).replace(' ', '') + \
                tokenized_words[w_idx][tokenized_words[w_idx].rfind(separator) + 1:]
    return False


def match_end_word_with_separator(source_sentence_piece, tokenized_words, w_idx, separator):
    if separator in tokenized_words[w_idx]:
        return remove_punctuation(''.join(tokenized_words[w_idx:])).replace(' ', '') == \
                        tokenized_words[w_idx][:tokenized_words[w_idx].rfind(separator)] + \
                        remove_punctuation(source_sentence_piece).replace(' ', '')
    return False


def get_target_sentence_beginning(sentence_id, source_sentence_piece):
    tokenized_words = tokenized_transcriptions[sentence_id].split(" ")
    eos_idx = None
    for i in range(len(tokenized_words)):
        if remove_punctuation(''.join(tokenized_words[:i+1])).replace(' ', '') == \
                remove_punctuation(source_sentence_piece).replace(' ', ''):
            eos_idx = i
        else:
            for w_s in WORD_SEPARATORS:
                if match_beginning_word_with_separator(source_sentence_piece, tokenized_words, i, w_s):
                    eos_idx = i
                    break
    if eos_idx is None:
        raise ValueError(
            "{} cannot be matched to {}".format(tokenized_words, source_sentence_piece))
    aligned_target_words = []
    aligns = text_aligns[sentence_id]
    for i in range(eos_idx+1):
        target_idx = aligns.get(i)
        if target_idx is not None:
            aligned_target_words.append(target_idx)
    aligned_target_words.sort(reverse=True)
    target_eos_idx = None
    for i, idx in enumerate(aligned_target_words):
        if i + 1 < len(aligned_target_words) and idx - aligned_target_words[i+1] > 5:
            # maybe we have a single word very far...
            # in that case we skip it
            continue
        target_eos_idx = idx
        break
    if target_eos_idx is None:
        if len([x for x in source_sentence_piece.split(' ') if x != '']) < 3:
            # For few words, probably they are missing in the translation...
            return ''
        raise ValueError(
            "{} cannot be aligned with {}".format(source_sentence_piece, aligns))
    if len(tokenized_words) <= 2:
        # In case of few words (eg. Laughter, so on), just do brutal alignment
        target_eos_idx = eos_idx
    return " ".join(tokenized_translations[sentence_id].split(" ")[:target_eos_idx+1])


def get_target_sentence_end(sentence_id, source_sentence_piece):
    tokenized_words = tokenized_transcriptions[sentence_id].split(" ")
    bos_idx = None
    for i in range(len(tokenized_words)):
        if remove_punctuation(''.join(tokenized_words[i:])).replace(' ', '') == \
                remove_punctuation(source_sentence_piece).replace(' ', ''):
            bos_idx = i
        else:
            for w_s in WORD_SEPARATORS:
                if match_end_word_with_separator(source_sentence_piece, tokenized_words, i, w_s):
                    bos_idx = i
                    break
    if bos_idx is None:
        raise ValueError(
            "{} cannot be matched to {}".format(tokenized_words, source_sentence_piece))
    aligned_target_words = []
    aligns = text_aligns[sentence_id]
    for i in range(bos_idx, len(tokenized_words)):
        target_idx = aligns.get(i)
        if target_idx is not None:
            aligned_target_words.append(target_idx)
    aligned_target_words.sort()
    target_bos_idx = None
    for i, idx in enumerate(aligned_target_words):
        if i + 1 < len(aligned_target_words) and idx - aligned_target_words[i+1] < -5:
            # maybe we have a single word very far...
            # in that case we skip it
            continue
        target_bos_idx = idx
        break
    if len(tokenized_words) <= 2:
        # In case of few words (eg. Laughter, so on), just do brutal alignment
        target_bos_idx = bos_idx
    if target_bos_idx is None:
        raise ValueError(
            "{} cannot be aligned with {}".format(source_sentence_piece, aligns))

    return " ".join(tokenized_translations[sentence_id].split(" ")[target_bos_idx:])


def remove_punctuation(input_str):
    nopunct = input_str.replace("'", "").translate(PUNCT_REMOVAL_TABLE)
    # Remove weird unicode chars....
    nopunct = nopunct.translate(UNCODE_CHARS_TO_CLEAN)
    return ' '.join(nopunct.split())


def build_sentence_definition(sentence_times, previous_definition):
    return {
        'duration': sentence_times[1] - sentence_times[0],
        'offset': sentence_times[0],
        'speaker_id': previous_definition['speaker_id'],
        'wav': previous_definition['wav']
    }


yaml_sentence_definition = sys.argv[1]
sentence_transcriptions = sys.argv[2]
json_alignments_dir = sys.argv[3]
text_alignments_file = sys.argv[4]
source_tokenized_file = sys.argv[5]
target_tokenized_file = sys.argv[6]
output_prefix = sys.argv[7]

logger.info("Reading sentence definitions...")
with open(yaml_sentence_definition, 'r', encoding="utf8") as f:
    try:
        sentences = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        print(exc)
        exit(1)

audio_files_to_sentences = defaultdict(list)
for idx in range(len(sentences)):
    audio_name = sentences[idx]['wav'].split(".")[0]
    audio_files_to_sentences[audio_name].append(idx)

logger.info("Reading sentence transcriptions...")
transcriptions = []
with open(sentence_transcriptions, 'r', encoding="utf8") as f:
    for line in f:
        transcriptions.append(line.strip())


logger.info("Reading text alignments...")
text_aligns = []
with open(text_alignments_file, 'r') as f:
    for line in f:
        if line.strip():
            current_aligns = {
                int(pairs[0]): int(pairs[1])
                for pairs in [t.split("-") for t in line.split(" ")]}
        else:
            current_aligns = {}
        text_aligns.append(current_aligns)

logger.info("Reading tokenized sentence transcriptions...")
tokenized_transcriptions = []
with open(source_tokenized_file, 'r', encoding="utf8") as f:
    for line in f:
        tokenized_transcriptions.append(line.strip())

logger.info("Reading tokenized sentence transcriptions...")
tokenized_translations = []
with open(target_tokenized_file, 'r', encoding="utf8") as f:
    for line in f:
        tokenized_translations.append(line.strip())

logger.info("Generating definitions...")
all_sentence_definitions = []
all_sentence_transcriptions = []
all_sentence_translations = []
all_context_definitions = []
all_context_transcriptions = []
all_context_translations = []
for audio_file in audio_files_to_sentences:
    if audio_file in BLACKLIST:
        logger.warning("Skipping {} because it is blacklisted".format(audio_file))
        continue
    try:
        with open(json_alignments_dir + '/' + audio_file + '.json', 'r') as f:
            alignments = json.load(f)
    except FileNotFoundError as exc:
        logger.warning("Not found: " + json_alignments_dir + '/' + audio_file + '.json, skipping it....')
        continue
    words = alignments['words']
    current_sentences = audio_files_to_sentences[audio_file]
    current_transcription = ""
    split_idx_by_sentence = {}
    sentence_start_idx_by_sentence = {}
    current_sentence_start_idx = 0

    sentences_in_talk = audio_files_to_sentences[audio_file]
    gentle_sentences = alignments["transcript"].split("\n")
    talk_sentence_i = 0
    num_removed_chars = 0
    for s_idx in sentences_in_talk:
        transcription_clean = remove_punctuation(transcriptions[s_idx])
        while transcription_clean.replace(' ', '') != \
                remove_punctuation(gentle_sentences[talk_sentence_i]).replace(' ', ''):
            # Little hack for sentences which were filtered out because they were
            # not found in audio
            for j in range(len(remove_punctuation(gentle_sentences[talk_sentence_i]).split(' '))):
                del words[current_sentence_start_idx]
            num_removed_chars += len(gentle_sentences[talk_sentence_i]) + 1  # We need to account also for \n
            del gentle_sentences[talk_sentence_i]

        num_words_in_sentence = len(transcription_clean.split(" "))
        split_idx = None
        retry_num = 0
        if current_sentence_start_idx + num_words_in_sentence > len(words):
            logger.error("trascription/text mismatch for talk {}".format(audio_file))
            raise ValueError
        for w_i in range(current_sentence_start_idx, current_sentence_start_idx + num_words_in_sentence):
            words[w_i]['startOffset'] -= num_removed_chars
        if num_words_in_sentence != 0:
            while split_idx is None and retry_num < NUM_RETRIES:
                split_idx = random.randint(
                    current_sentence_start_idx,
                    current_sentence_start_idx + num_words_in_sentence - 1)
                if words[split_idx]['case'] == 'not-found-in-audio':
                    split_idx = None
                retry_num += 1
            if split_idx is None:
                logger.warning("Not able to find split for sentence after {} retries...".format(retry_num))
        split_idx_by_sentence[s_idx] = split_idx
        sentence_start_idx_by_sentence[s_idx] = current_sentence_start_idx
        current_sentence_start_idx += num_words_in_sentence
        talk_sentence_i += 1

    # Eventually remove sentences at the end which are not recognized in the transcriptions
    while talk_sentence_i < len(gentle_sentences):
        if gentle_sentences[talk_sentence_i] != '':
            for j in range(len(remove_punctuation(gentle_sentences[talk_sentence_i]).split(' '))):
                del words[current_sentence_start_idx]
        del gentle_sentences[talk_sentence_i]
    if current_sentence_start_idx < len(words):
        logger.error("transcription/text mismatch for talk {}".format(audio_file))
        raise ValueError

    gentle_transcript = '\n'.join(gentle_sentences)
    for i in range(len(sentences_in_talk)):
        try:
            original_sentence_definition = sentences[sentences_in_talk[i]]
            current_sentence_split_idx = split_idx_by_sentence[sentences_in_talk[i]]
            if current_sentence_split_idx is None:  # We discard this sentence: we don't know where to split...
                logger.warning("Discarding a sentence, because its split is unknown...")
                continue
            current_split_word = words[current_sentence_split_idx]

            if i == 0:
                prev_sentence = gentle_transcript[:current_split_word["startOffset"]]
                prev_sentence_time = (0.0, current_split_word["start"])
                prev_target = get_target_sentence_beginning(sentences_in_talk[i], prev_sentence)
            else:
                current_sentence_start_idx = sentence_start_idx_by_sentence[sentences_in_talk[i]]
                current_sentence_start_word = words[current_sentence_start_idx]
                prev_sentence_split_idx = split_idx_by_sentence[sentences_in_talk[i - 1]]
                if prev_sentence_split_idx is None:  # We discard this sentence: we don't know where to start its context...
                    logger.warning("Discarding a sentence, because its previous split is unknown...")
                    continue
                prev_split_word = words[prev_sentence_split_idx]

                prev_sentence = gentle_transcript[prev_split_word["startOffset"]:current_split_word["startOffset"]]
                prev_sentence_time = (prev_split_word["start"], current_split_word["start"])
                if prev_sentence_split_idx == current_sentence_start_idx:
                    prev_target = ''
                else:
                    prev_target = get_target_sentence_end(
                        sentences_in_talk[i-1],
                        gentle_transcript[
                            prev_split_word["startOffset"]:current_sentence_start_word["startOffset"]]) + " "
                if current_sentence_split_idx != current_sentence_start_idx:
                    prev_target += get_target_sentence_beginning(
                        sentences_in_talk[i],
                        gentle_transcript[
                            current_sentence_start_word["startOffset"]:current_split_word["startOffset"]])

            if i + 1 >= len(sentences_in_talk):
                curr_sentence = gentle_transcript[current_split_word["startOffset"]:]
                curr_sentence_time = (
                    current_split_word["start"],
                    original_sentence_definition['offset'] + original_sentence_definition['duration'])
                curr_target = get_target_sentence_end(sentences_in_talk[i], curr_sentence)
            else:
                next_sentence_start_idx = sentence_start_idx_by_sentence[sentences_in_talk[i+1]]
                next_sentence_start_word = words[next_sentence_start_idx]
                next_sentence_split_idx = split_idx_by_sentence[sentences_in_talk[i + 1]]
                if next_sentence_split_idx is None:  # We discard this sentence: we don't know where to finish it...
                    logger.warning("Discarding a sentence, because its following split is unknown...")
                    continue
                next_split_word = words[next_sentence_split_idx]

                curr_sentence = gentle_transcript[current_split_word["startOffset"]:next_split_word["startOffset"]]
                curr_sentence_time = (current_split_word["start"], next_split_word["start"])
                if current_sentence_split_idx == next_sentence_start_idx:
                    curr_target = ''
                else:
                    curr_target = get_target_sentence_end(
                        sentences_in_talk[i],
                        gentle_transcript[
                            current_split_word["startOffset"]:next_sentence_start_word["startOffset"]]) + " "
                if next_sentence_split_idx != next_sentence_start_idx:
                    curr_target += get_target_sentence_beginning(
                        sentences_in_talk[i+1],
                        gentle_transcript[next_sentence_start_word["startOffset"]:next_split_word["startOffset"]])

            sentence_definition = build_sentence_definition(curr_sentence_time, original_sentence_definition)
            context_definition = build_sentence_definition(prev_sentence_time, original_sentence_definition)

            if sentence_definition['duration'] < 0.25 or context_definition['duration'] < 0.25:
                logger.warning("Skip sentence because either the context or the sentence is too short")
                continue

            all_sentence_definitions.append(sentence_definition)
            all_sentence_transcriptions.append(curr_sentence.replace('\n', ' '))
            all_sentence_translations.append(curr_target.replace('\n', ' '))
            all_context_definitions.append(context_definition)
            all_context_transcriptions.append(prev_sentence.replace('\n', ' '))
            all_context_translations.append(prev_target.replace('\n', ' '))
        except ValueError as ve:
            logger.warning("Skipping sentence because of text alignment issues: {}".format(ve))
    logger.info("Finished processing of {}".format(audio_file))

logger.info("Writing new sentence definitions...")
with open("{}.yaml".format(output_prefix), 'w', encoding="utf8") as yaml_file:
    yaml.dump(all_sentence_definitions, yaml_file, default_flow_style=None)
with open("{}.en".format(output_prefix), 'w', encoding="utf8") as txt_file:
    for l in all_sentence_transcriptions:
        txt_file.write(l + '\n')
with open("{}.de".format(output_prefix), 'w', encoding="utf8") as txt_file:
    for l in all_sentence_translations:
        txt_file.write(l + '\n')

logger.info("Writing new context definitions...")
with open("{}.context.yaml".format(output_prefix), 'w', encoding="utf8") as yaml_file:
    yaml.dump(all_context_definitions, yaml_file, default_flow_style=None)
with open("{}.context.en".format(output_prefix), 'w', encoding="utf8") as txt_file:
    for l in all_context_transcriptions:
        txt_file.write(l + '\n')
with open("{}.context.de".format(output_prefix), 'w', encoding="utf8") as txt_file:
    for l in all_context_translations:
        txt_file.write(l + '\n')
