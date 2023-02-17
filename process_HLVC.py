#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 17:41:05 2022

@author: Katrina Li
"""

from multiprocessing import Process, freeze_support, set_start_method
from glob import glob
import re
import pycantonese
import spacy
from itertools import groupby
import chinese_converter


nlp_en = spacy.load("en_core_web_sm", disable=["parser", "ner"])
out_tsv = open("./output/processed_canCHN_delete.tsv", "w") # All token information
freq_tsv = open("./output/processed_canCHN_freq_delete.tsv", "w") # Frequency information
corpus_tsv = open("./output/processed_can_corpus_delete.tsv", "w") # Annotated corpus
corpus_switch_tsv = open("./output/processed_can_switch_corpus_delete.tsv", "w") # Annotated switch corpus

# Filler list
fillers = ['eh', 'ah', 'uh', 'um', 'uhh', 'okay', 'yeah', 'ehh', 'umm', 'ahm', 'hmm', 'ok', 'hm', 'oh', 'wor', 'mmm', 'ehm', 'ehhh', 'mm', '啊', '啦', '咧', '囉', '呢', '㗎', '嚇', '啫', '喇', '咩', '呀','咯','嘎','喎','嘛','mhm', 'em', 'uhhh','markham','ahuh','duh','huh','uhm','uhmm','uhmmm','哈','哇','噶']

# File path
hlvc_files = glob("/Users/kechun/Github/Code-switching/data_CantoneseCHN/*.txt")

def flat(l):
    for k in l:
        if not isinstance(k, (list, tuple)):
           yield k
        else:
           yield from flat(k)

if __name__ == '__main__':
    freeze_support()

    # Count the size of the corpus
    max_total = 0
    filt_total = 0
    mixed_total = 0
    mixed_nulls_total = 0
    mixed = 0

    cne_count = 0
    enc_count = 0
    ce_count = 0
    ec_count = 0

    total_tokens = 0
    # Token and tone type counts
    en_dict = {}
    cn_dict = {}
    cn_char_dict = {}
    tone_dict = {}
    filler_dict = {}
    # CN -> EN counts
    cn_en_en_dict = {}
    cn_en_cn_dict = {}
    cn_en_tone_dict = {}
    cn_en_cn_char_dict = {}
    # EN -> CN counts
    en_cn_en_dict = {}
    en_cn_cn_dict = {}
    en_cn_tone_dict = {}
    en_cn_cn_char_dict = {}
    # CN -> Nulls -> EN counts
    cn_nulls_en_cn_dict = {}
    cn_nulls_en_cn_char_dict = {}
    cn_nulls_en_nulls_dict = {}
    cn_nulls_en_nulls_char_dict = {}
    cn_nulls_en_en_dict = {}
    cn_nulls_en_tone_dict = {}
    # EN -> Nulls -> CN counts
    en_nulls_cn_cn_dict = {}
    en_nulls_cn_cn_char_dict = {}
    en_nulls_cn_nulls_dict = {}
    en_nulls_cn_nulls_char_dict = {}
    en_nulls_cn_en_dict = {}
    en_nulls_cn_tone_dict = {}


    # Loop through files
    for filename in hlvc_files:
        # Open each file
        with open(filename) as text:
            # Process line by line
            lastline = ""
            sentence_n = 1
            for line in text:
                # Count the total number of sentences BEFORE filtering
                max_total += 1
                # Count the 'sentence number' to match with ELAN numbers: Compare if the file is from the same speaker or not
                current_speaker = line.split("\t")[-1].strip()
                last_speaker = lastline.split("\t")[-1].strip()
                if current_speaker == last_speaker:
                    sentence_n += 1
                else:
                    sentence_n = 1
                lastline = line
                # Extract duration information
                start = float(line.split("\t")[3].strip())
                end = float(line.split("\t")[5].strip())
                # Get the text column and lower case it. Add trailing whitespace for regex.
                column = line.split("\t")[6].lower()+" "
                # Ignore lines that contain "["; normally [redacted] text. (300 lines)
                if "[" in column: continue
                # Ignore the digit for the same reason
                if re.search('\d',column): continue
                # Replace non-speech sounds (in brackets) with whitespace; e.g. (laughs)
                column = re.sub("\(.*?\)", " ", column)
                # Replace most punctuation with whitespace; keep ' and -; e.g. don't and T-shirt
                column = re.sub("[^\w\d\s\'-]+", " ", column)
                # Replace incomplete words with whitespace: most are of the form "\w+- "; "for- " end of sentence still there
                column = re.sub("\w+- ", " ", column)
                # Now we can remove hyphens and apostrophes that are not surrounded by \w: https://stackoverflow.com/questions/61109658/remove-dash-from-string-but-not-from-middle-when-its-surrounded-by-a-z
                column = re.sub("((?<!\w)[\'-]|[\'-](?!\w))", "", column)

                # Adding space between Chinese characters and English words, so that they won't be mixed
                # Between English and Chinese
                column = re.sub(u'(?<=[a-zA-Z])(?=[\u4e00-\u9fff\u3400-\u4dbf])'," ",column)
                # Between Chinese and English
                column = re.sub(u'(?<=[\u4e00-\u9fff\u3400-\u4dbf])(?=[a-zA-Z])'," ",column)

                # Split tokens
                sent_mix = column.split()

                # Loop through tokens by id
                for i in range(0, len(sent_mix)):
                    # Remove tokens that contain at least 1 letter and 2 adjacent numbers; e.g. a144
                    if re.search("\d{2}", sent_mix[i]) and sent_mix[i].islower(): sent_mix[i] = ""
                    # Remove tokens of the form [a-z]\d EXCEPT m4 which is a valid Cantonese word
                    if re.match("[a-z]\d", sent_mix[i]) and sent_mix[i] != "m4": sent_mix[i] = ""

                # Filter empty tokens removed in the above loop
                sent_mix = list(filter(None, sent_mix))
                # Skip empty sentences
                if not sent_mix: continue


                # First round lang symbol
                # Set langs label (Chinese concatenated)
                langs_block = []
                for i in range(0,len(sent_mix)):
                    lang_block = "cn" if re.match(u'[\u4e00-\u9fff\u3400-\u4dbf]',sent_mix[i]) else "en"
                    langs_block.append(lang_block)

                # sanity check
                assert len(sent_mix) == len(langs_block)

                # Postagging - the Chinese one can be sent to post tagging as it is, but the English one needs combination.
                switch_ids = [0] + [i for i in range(1, len(langs_block)) if langs_block[i] != langs_block[i-1]]
                pos = [] # store pos-taggers
                sent_word = [] # store tokens in jyutping format
                sent_cn = [] # store Chinese characters
                sent_all = [] # store Jyuping for every character
                sent_cn_all = [] # store every Chinese character

                # Loop through the indices
                for i in range(0, len(switch_ids)):
                    # Get the language of the current chunk
                    lang = langs_block[switch_ids[i]]

                    # Get the same-language tokens in the chunk
                    if i != len(switch_ids)-1:
                        chunk = sent_mix[switch_ids[i]:switch_ids[i+1]]
                    # Last chunk
                    else:
                        chunk = sent_mix[switch_ids[i]:]

                    # Annotate the chunk with the correct version of spacy
                    if lang == "en":
                        # assign new name to chunk for the nlp will add information to it
                        chunk_process = spacy.tokens.Doc(nlp_en.vocab, chunk)
                        nlp_en(chunk_process)
                        # Get the POS tags
                        tags = [tok.pos_ for tok in chunk_process]
                        pos.extend(tags)
                        sent_word.extend(chunk)
                        sent_cn.extend(chunk)
                        sent_all.extend(chunk)
                        sent_cn_all.extend(chunk)
                    else: # For Chinese, we need to do loop
                        for i in range(0,len(chunk)):
                            # First convert simplied to traditional, otherwise it will not be sucessful converted to jyuping
                            chunk[i] = chinese_converter.to_traditional(chunk[i])
                            ## Output every character
                            chars = [item for item in chunk[i]]
                            sent_cn_all.extend(chars)
                            # segment for NLP
                            segment = pycantonese.segment(chunk[i])
                            ## Output the segmented character
                            sent_cn.extend(segment)
                            # Pos-tagging
                            postag_list = pycantonese.pos_tag(segment)
                            tags = [item[1] for item in postag_list]
                            pos.extend(tags)
                            
                            # Jyuping list
                            # To keep consistent with pos-tagging, we have to feed segmented list to jyuping
                            jyutping_word = [pycantonese.characters_to_jyutping(seg) for seg in segment]
                            # This returns pairs of characters and jyutping
                            # Each seg is already a list, so double loop to extract jyutping
                            jyutping_word = [item[1] for group in jyutping_word for item in group]
                            # We want to split the strings like 'daan6hai6'
                            # We first concatenate all strings, then adding space after the number
                            # 'all' includes jyuping for every character, this is to calculate the distribution
                            jyutping_all = [re.sub(r'(?<=\d)', r' ',word) for word in jyutping_word]
                            jyutping_all_strip = [word.strip() for word in jyutping_all]
                            jyutping_all_split = [word.split() for word in jyutping_all_strip]
                            #### flatten this list after the last of each group has been extracted
                            jyutping_all_atom = list(flat(jyutping_all_split))

                            sent_word.extend(jyutping_all_strip)
                            sent_all.extend(jyutping_all_atom)


                # Store token langs here
                langs = []
                # Loop through tokens in all remaining sentences
                for i in range(0,len(sent_word)):
                    # Get the token lang
                    if sent_cn[i] in fillers or pos[i] == "PROPN" or pos[i] == "INTJ":
                        lang = 'null'
                    elif re.search("\D\d$", sent_word[i]):
                        lang = 'cn'
                    else:
                        lang = 'en'
                    langs.append(lang)

                # Print sampes
                # print(sent_mix)
                # print(sent_cn)
                # print(sent_word)
                # print(pos)
                # print(langs)
                # print(sent_all)
                # print(sent_cn_all)

                # sanity check
                assert len(pos) == len(sent_word)  == len(sent_cn) == len(langs) #== len(sent)
                assert len(sent_all) == len(sent_cn_all)

                # Calculate the total number of tokens
                total_tokens += len(sent_cn)

                # Now we export the corpus for manual examination
                corpus_tsv.write(str(max_total)  + "\t" + line.split()[-1] + "\t" + str(sentence_n) + "\t" + str(start) + "\t" + str(end) + "\t" "sent_mix" + "\t" + " | ".join(sent_mix)+"\n")
                corpus_tsv.write(str(max_total) + "\t" + line.split()[-1] + "\t" + str(sentence_n) + "\t" + str(start) + "\t" + str(end) + "\t" "sent_cn" + "\t" + " | ".join(sent_cn)+"\n")
                corpus_tsv.write(str(max_total) + "\t" + line.split()[-1] + "\t" + str(sentence_n) + "\t" + str(start) + "\t" + str(end) + "\t" "sent_word" + "\t" + " | ".join(sent_word)+"\n")
                corpus_tsv.write(str(max_total) + "\t" + line.split()[-1] + "\t" + str(sentence_n) + "\t" + str(start) + "\t" + str(end) + "\t" "pos" + "\t" + " | ".join(pos)+"\n")
                corpus_tsv.write(str(max_total) + "\t" + line.split()[-1] + "\t" + str(sentence_n) + "\t" + str(start) + "\t" + str(end) + "\t" "langs" + "\t" + " | ".join(langs)+"\n")
                corpus_tsv.write(str(max_total) + "\t" + line.split()[-1] + "\t" + str(sentence_n) + "\t" + str(start) + "\t" + str(end) + "\t" "sent_all" + "\t" + " | ".join(sent_all)+"\n")
                corpus_tsv.write(str(max_total) + "\t" + line.split()[-1] + "\t" + str(sentence_n) + "\t" + str(start) + "\t" + str(end) + "\t" "sent_cn_all" + "\t" + " | ".join(sent_cn_all)+"\n")

    ####################### Correct any tokens here ###################
                # The change below is only a demonstration to show how to manually modify some incorrect pronunciation.But chaning the jyutping for '咧' does not influence the result here as it is filter as 'language_neutral'.
                for i in range(0, len(sent_word)):
                    # change jyutping for 咧
                    if sent_cn[i] == "咧":
                        # we need to properly replace sent_word as well other well the dictionary won't match
                        sent_word[i] = re.sub("le2", "le1", sent_word[i])
                for i in range(0, len(sent_all)):
                    # change jyutping for 咧
                    if sent_cn_all[i] == "咧":
                        sent_all[i] == 'le1'
    ###################################################################

                # Count total number of sentences AFTER filtering
                filt_total += 1

                # Calculate the freq based on 'potential switch points':
                # For English words, every word is counted; for Chinese words, only word at boundary positions.
                for i in range(0,len(sent_word)):
                    tok = sent_word[i]
                    tok_cn = sent_cn[i] #as many Chinese forms as possible
                    # Keep track of English tokens
                    if langs[i] == "en":
                        if tok in en_dict: en_dict[tok] += 1
                        else: en_dict[tok] = 1
                    # Keep track of Cantonese tokens
                    elif langs[i] == "cn":
                        # The cn tokens can be stored as a whole (no need to decompose as tonal information)
                        if tok_cn in cn_char_dict: cn_char_dict[tok_cn] += 1
                        else: cn_char_dict[tok_cn] = 1
                        # check if there is only one character in the Chinese group or multiple
                        lentokens = len(sent_word[i].split())
                        # if only one word in the token, directly output the tone
                        if lentokens == 1:
                            # Get the tone number
                            tone = int(tok[-1])
                            # Save the tok in tone_dict
                            if tone in tone_dict: tone_dict[tone] += 1
                            else: tone_dict[tone] = 1
                            # Save the token in cn_dict
                            if tok in cn_dict: cn_dict[tok] += 1
                            else: cn_dict[tok] = 1
                        else: # if more than one word, then both first and last information are collected.
                            tok_first = sent_word[i].split()[0]
                            tok_last = sent_word[i].split()[-1]
                            tone_first = int(tok_first[-1])
                            tone_last = int(tok_last[-1])
                            if tone_first in tone_dict: tone_dict[tone_first] += 1
                            else: tone_dict[tone_first] = 1
                            if tone_last in tone_dict: tone_dict[tone_last] += 1
                            else: tone_dict[tone_last] = 1
                            if tok_first in cn_dict: cn_dict[tok_first] += 1
                            else: cn_dict[tok_first] = 1
                            if tok_last in cn_dict: cn_dict[tok_last] += 1
                            else: cn_dict[tok_last] = 1
                    else:
                        if tok_cn in filler_dict: filler_dict[tok_cn] +=1
                        else: filler_dict[tok_cn] = 1

                # Increment mixed total
                if len(set(langs)) != 1: mixed_total += 1

                # Now we proceed to three-way switch
                # Add explicit index to each language tag
                langs_ids = [(langs[i], i) for i in range(0, len(langs))]
                # Group by the language
                lang_groups = [(lang, list(group)) for lang, group in groupby(langs_ids, lambda x: x[0])]
                cne = 0
                enc = 0
                cn_null_en = [0]*len(sent_word)
                en_null_cn = [0]*len(sent_word)
                info_cne_null = ["none"] * len(sent_word)
                info_cne_en = ["none"] * len(sent_word)
                info_enc_null = ["none"] * len(sent_word)
                info_enc_en = ["none"] * len(sent_word)
                # Only sentences with at least 3 groups could be enc or cne
                if len(lang_groups) > 2 and len(set(langs)) == 3:
                    # Loop through the groups
                    for i in range(0, len(lang_groups)-2):
                        # CN -> Null -> EN
                        if lang_groups[i][0] == "cn" and lang_groups[i+1][0] == "null" and lang_groups[i+2][0] == "en":
                            cne = 1
                            cne_count += 1
                            # Useful info
                            cn_null_en[lang_groups[i][1][-1][-1]] = 1 # for regression analysis
                            cn_token = sent_word[lang_groups[i][1][-1][-1]]
                            # extract tone infor at the first
                            cn_token_last = cn_token.split()[-1]
                            tone = int(cn_token_last[-1])
                            cn_token_char = sent_cn[lang_groups[i][1][-1][-1]]
                            null_tokens = sent_word[lang_groups[i+1][1][0][-1]:lang_groups[i+1][1][-1][-1]+1]
                            null_tokens_char = sent_cn[lang_groups[i+1][1][0][-1]:lang_groups[i+1][1][-1][-1]+1]
                            info_cne_null[lang_groups[i][1][-1][-1]] = null_tokens_char # for regression analysis
                            en_token = sent_word[lang_groups[i+2][1][0][-1]]
                            info_cne_en[lang_groups[i][1][-1][-1]] = en_token # for regression analysis
                            #print(cn_token, cn_token_char, cn_token_last, "--", null_tokens, null_tokens_char, "--", en_token)
                            # Add to dictionary
                            if cn_token_last in cn_nulls_en_cn_dict: cn_nulls_en_cn_dict[cn_token_last] += 1
                            else: cn_nulls_en_cn_dict[cn_token_last] = 1
                            if cn_token_char in cn_nulls_en_cn_char_dict: cn_nulls_en_cn_char_dict[cn_token_char] += 1
                            else: cn_nulls_en_cn_char_dict[cn_token_char] = 1
                            # Dealing with null tokens: small loop
                            for item in null_tokens:
                                if item in cn_nulls_en_nulls_dict: cn_nulls_en_nulls_dict[item] += 1
                                else: cn_nulls_en_nulls_dict[item] = 1
                            for item in null_tokens_char:
                                if item in cn_nulls_en_nulls_char_dict: cn_nulls_en_nulls_char_dict[item] += 1
                                else: cn_nulls_en_nulls_char_dict[item] = 1
                            if en_token in cn_nulls_en_en_dict: cn_nulls_en_en_dict[en_token] += 1
                            else: cn_nulls_en_en_dict[en_token] = 1
                            if tone in cn_nulls_en_tone_dict: cn_nulls_en_tone_dict[tone] += 1
                            else: cn_nulls_en_tone_dict[tone] = 1
                            #print(list(lang_groups[i+1][1]))

                        # EN -> Null -> CN
                        if lang_groups[i][0] == "en" and lang_groups[i+1][0] == "null" and lang_groups[i+2][0] == "cn":
                            enc = 1
                            enc_count += 1
                            #print(sent_cn)
                            #print(langs)
                            # Useful info
                            en_null_cn[lang_groups[i+2][1][0][-1]] = 1
                            en_token = sent_word[lang_groups[i][1][-1][-1]]
                            null_tokens = sent_word[lang_groups[i+1][1][0][-1]:lang_groups[i+1][1][-1][-1]+1]
                            null_tokens_char = sent_cn[lang_groups[i+1][1][0][-1]:lang_groups[i+1][1][-1][-1]+1]
                            info_enc_null[lang_groups[i+2][1][0][-1]] = null_tokens_char # for regression analysis, attached to the chinese token
                            info_enc_en[lang_groups[i+2][1][0][-1]] = en_token # for regression analysis
                            cn_token = sent_word[lang_groups[i+2][1][0][-1]]
                            # extract tone infor at the first
                            cn_token_first = cn_token.split()[0]
                            tone = int(cn_token_first[-1])
                            # cn has group info
                            cn_token_char = sent_cn[lang_groups[i+2][1][0][-1]]
                            #print(en_token, "--", null_tokens, null_tokens_char,"--",cn_token, cn_token_char, cn_token_first)
                            if cn_token_first in en_nulls_cn_cn_dict: en_nulls_cn_cn_dict[cn_token_first] += 1
                            else: en_nulls_cn_cn_dict[cn_token_first] = 1
                            if cn_token_char in en_nulls_cn_cn_char_dict: en_nulls_cn_cn_char_dict[cn_token_char] += 1
                            else: en_nulls_cn_cn_char_dict[cn_token_char] = 1
                            for item in null_tokens:
                                if item in en_nulls_cn_nulls_dict: en_nulls_cn_nulls_dict[item] += 1
                                else: en_nulls_cn_nulls_dict[item] = 1
                            for item in null_tokens_char:
                                if item in en_nulls_cn_nulls_char_dict: en_nulls_cn_nulls_char_dict[item] += 1
                                else: en_nulls_cn_nulls_char_dict[item] = 1
                            for item in null_tokens:
                                if item in en_nulls_cn_nulls_dict: en_nulls_cn_nulls_dict[item] += 1
                                else: en_nulls_cn_nulls_dict[item] = 1
                            for item in null_tokens_char:
                                if item in en_nulls_cn_nulls_char_dict: en_nulls_cn_nulls_char_dict[item] += 1
                                else: en_nulls_cn_nulls_char_dict[item] = 1
                            if en_token in en_nulls_cn_en_dict: en_nulls_cn_en_dict[en_token] += 1
                            else: en_nulls_cn_en_dict[en_token] = 1
                            if tone in en_nulls_cn_tone_dict: en_nulls_cn_tone_dict[tone] += 1
                            else: en_nulls_cn_tone_dict[tone] = 1

                # Sanity check ### Print list and check!
                assert len(cn_null_en) == len(en_null_cn)== len(sent_word) == len(info_cne_null)== len(info_cne_en) == len(info_enc_null)== len(info_enc_en)

                if cne ==1 or enc == 1: mixed_nulls_total +=1

                ec = 0
                ce = 0
                # Loop through tokens to look for switches
                for i in range(1, len(sent_word)):
                    # Look for cases where the language changes
                    if langs[i] != langs[i-1]:
                        # CN -> EN
                        if langs[i] == "en" and langs[i-1] == 'cn':
                           ce = 1
                           ce_count += 1
                           # Save the en token
                           if sent_word[i] in cn_en_en_dict: cn_en_en_dict[sent_word[i]] += 1
                           else: cn_en_en_dict[sent_word[i]] = 1
                           # Save the cn token (last token in the group)
                           tok_last = sent_word[i-1].split()[-1]
                           if tok_last in cn_en_cn_dict: cn_en_cn_dict[tok_last] += 1
                           else: cn_en_cn_dict[tok_last] = 1
                           # Save the tone
                           tone = int(tok_last[-1])
                           if tone in cn_en_tone_dict: cn_en_tone_dict[tone] += 1
                           else: cn_en_tone_dict[tone] = 1
                           # Save the cn char token - no need to decompose
                           if sent_cn[i-1] in cn_en_cn_char_dict: cn_en_cn_char_dict[sent_cn[i-1]] +=1
                           else: cn_en_cn_char_dict[sent_cn[i-1]] = 1
                        # EN -> CN
                        elif langs[i] == 'cn' and langs[i-1] == 'en':
                           ec = 1
                           ec_count += 1
                           # Save the en token
                           if sent_word[i-1] in en_cn_en_dict: en_cn_en_dict[sent_word[i-1]] += 1
                           else: en_cn_en_dict[sent_word[i-1]] = 1
                           # Save the cn token (first token in the group)
                           tok_first = sent_word[i].split()[0]
                           if tok_first in en_cn_cn_dict: en_cn_cn_dict[tok_first] += 1
                           else: en_cn_cn_dict[tok_first] = 1
                           # Save the tone
                           tone = int(tok_first[-1])
                           if tone in en_cn_tone_dict: en_cn_tone_dict[tone] += 1
                           else: en_cn_tone_dict[tone] = 1
                           # Save the cn char token - no need to decompose
                           if sent_cn[i] in en_cn_cn_char_dict: en_cn_cn_char_dict[sent_cn[i]] += 1
                           else: en_cn_cn_char_dict[sent_cn[i]] = 1

                if ce ==1 or ec == 1:
                    mixed +=1
                    # Now we export the corpus for manual examination
                    corpus_switch_tsv.write(str(max_total) + "\t" + line.split()[-1] + "\t" + str(sentence_n) + "\t" + str(start) + "\t" + str(end) + "\t" "sent_mix" + "\t" + " | ".join(sent_mix)+"\n")
                    corpus_switch_tsv.write(str(max_total) + "\t" + line.split()[-1] + "\t" + str(sentence_n) + "\t" + str(start) + "\t" + str(end) + "\t" "sent_cn" + "\t" + " | ".join(sent_cn)+"\n")
                    corpus_switch_tsv.write(str(max_total) + "\t" + line.split()[-1] + "\t" + str(sentence_n) + "\t" + str(start) + "\t" + str(end) + "\t" "sent_word" + "\t" + " | ".join(sent_word)+"\n")
                    corpus_switch_tsv.write(str(max_total) + "\t" + line.split()[-1] + "\t" + str(sentence_n) + "\t" + str(start) + "\t" + str(end) + "\t" "pos" + "\t" + " | ".join(pos)+"\n")
                    corpus_switch_tsv.write(str(max_total) + "\t" + line.split()[-1] + "\t" + str(sentence_n) + "\t" + str(start) + "\t" + str(end) + "\t" "langs" + "\t" + " | ".join(langs)+"\n")
                    corpus_switch_tsv.write(str(max_total) + "\t" + line.split()[-1] + "\t" + str(sentence_n) + "\t" + str(start) + "\t" + str(end) + "\t" "sent_all" + "\t" + " | ".join(sent_all)+"\n")
                    corpus_switch_tsv.write(str(max_total) +"\t" + line.split()[-1] +  "\t" + str(sentence_n) +  "\t" + str(start) + "\t" + str(end) +"\t" "sent_cn_all" + "\t" + " | ".join(sent_cn_all)+"\n")


                # Write to output
                for i in range(0, len(sent_word)):
                    # Store output items here
                    out = []
                    out.append(sent_word[i]) # Column 1: word item with pinyin
                    out.append(sent_cn[i]) # Column 2: word item with characters
                    out.append(langs[i]) # Column 3: lang label
                    out.append(pos[i]) # Column 4: Post-tagging
                    if langs[i] != "cn":
                        out.append("-") # Column 5: initial tone
                        out.append("-") # Column 6: last tone
                    else:
                        out.append(sent_word[i].split()[0][-1]) # Column 5: initial tone
                        out.append(sent_word[i].split()[-1][-1]) # Column 6: last tone
                    # Attach true switching infomation
                    if langs[i] == 'cn':
                        if i!=0 and langs[i-1] == 'en': #EN-> CN
                            out.append("1") # Column 7: if EN-CN
                            out.append(sent_word[i-1]) # Column 8: record the EN before
                        else:
                            out.append("0") # Column 7:
                            out.append("0") # Column 8:
                        if i != len(sent_word)-1 and langs[i+1] == "en":
                            out.append("1") # Column 9: if CN-EN
                            out.append(sent_word[i+1]) # Column 10: record the EN after
                        else:
                            out.append("0") # Column 9
                            out.append("0") # Column 10
                    else:
                        out.append("0") # Column 7
                        out.append("0") # Column 8
                        out.append("0") # Column 9
                        out.append("0") # Column 10
                    out.append(str(cn_null_en[i])) # Column 11: if contains cn_null_en
                    out.append(str(info_cne_null[i])) # Column 12: nulls string
                    out.append(str(info_cne_en[i]))# Column 13: en string
                    out.append(str(en_null_cn[i])) # Column 14: if contains en_null_cn
                    out.append(str(info_enc_null[i])) # Column 15: nulls string
                    out.append(str(info_enc_en[i]))# Column 16: en string

                    # Column 17: Also output source information
                    out.append(line.split()[-1])
                    # Column 18: Sentence info
                    out.append(str(max_total))
                    # Column 19: word in sentence info
                    out.append(str(i))
                    # Write to output
                    out_tsv.write("\t".join(out)+"\n")
                out_tsv.write("\n")

    # Calculate tone distribution normalised by word frequency
    # Full dictionary for freq
    for tok, cnt in cn_dict.items():
        freq = cnt/sum(cn_dict.values())
        out = []
        out.append(tok)
        out.append(str(freq))
        out.append("cn_dict")
        freq_tsv.write("\t".join(out)+ "\n")
    for tok, cnt in en_dict.items():
        freq = cnt/sum(en_dict.values())
        out = []
        out.append(tok)
        out.append(str(freq))
        out.append("en_dict")
        freq_tsv.write("\t".join(out)+ "\n")
    freq_tsv.write("\n")
    for tok, cnt in cn_char_dict.items():
        freq = cnt/sum(cn_char_dict.values())
        out = []
        out.append(tok)
        out.append(str(freq))
        out.append("cn_char_dict")
        freq_tsv.write("\t".join(out)+ "\n")
    freq_tsv.write("\n")
    for tok, cnt in filler_dict.items():
        freq = cnt/sum(filler_dict.values())
        out = []
        out.append(tok)
        out.append(str(freq))
        out.append("filler_dict")
        freq_tsv.write("\t".join(out)+ "\n")
    freq_tsv.write("\n")
    for tok, cnt in tone_dict.items():
        print(str(tok), str(cnt))
        freq = cnt/sum(tone_dict.values())
        out = []
        out.append(str(tok))
        out.append(str(freq))
        out.append("tone")
        freq_tsv.write("\t".join(out)+ "\n")
    freq_tsv.write("\n")
    # CN- > EN
    cn_en_tone_norm_dict = {}
    for tok, cnt in cn_en_cn_dict.items():
        # Get the tone
        tone = int(tok[-1])
        # Add the tone to the dict
        if tone not in cn_en_tone_norm_dict: cn_en_tone_norm_dict[tone] = 0
        # Calculate word frequency in the corpus
        freq = cn_dict[tok]/sum(cn_dict.values())
        # Weight the tone at a switch point by frequency
        cn_en_tone_norm_dict[tone] += freq*cnt
    # EN- > CN
    en_cn_tone_norm_dict = {}
    for tok, cnt in en_cn_cn_dict.items():
        # Get the tone
        tone = int(tok[-1])
        # Add the tone to the dict
        if tone not in en_cn_tone_norm_dict: en_cn_tone_norm_dict[tone] = 0
        # Calculate word frequency in the corpus
        freq = cn_dict[tok]/sum(cn_dict.values())
        # Weight the tone at a switch point by frequency
        en_cn_tone_norm_dict[tone] += freq*cnt

    # CN- > nulls -> EN
    cn_nulls_en_tone_norm_dict = {}
    for tok, cnt in cn_nulls_en_cn_dict.items():
        # Get the tone
        tone = int(tok[-1])
        # Add the tone to the dict
        if tone not in cn_nulls_en_tone_norm_dict: cn_nulls_en_tone_norm_dict[tone] = 0
        # Calculate word frequency in the corpus
        freq = cn_dict[tok]/sum(cn_dict.values())
        # Weight the tone at a switch point by frequency
        cn_nulls_en_tone_norm_dict[tone] += freq*cnt

    # EN- > nulls -> CN
    en_nulls_cn_tone_norm_dict = {}
    for tok, cnt in en_nulls_cn_cn_dict.items():
        # Get the tone
        tone = int(tok[-1])
        # Add the tone to the dict
        if tone not in en_nulls_cn_tone_norm_dict: en_nulls_cn_tone_norm_dict[tone] = 0
        # Calculate word frequency in the corpus
        freq = cn_dict[tok]/sum(cn_dict.values())
        # Weight the tone at a switch point by frequency
        en_nulls_cn_tone_norm_dict[tone] += freq*cnt

    # Print results
    print("\nTotal sentences: "+str(max_total))
    print("Filtered sentences: "+str(filt_total))
    print("Mixed sentences (occurrence of different language labels): "+str(mixed_total)+" ("+str(round((mixed_total/filt_total)*100,2))+"%)")
    print("Mixed sentences (cne or enc): "+str(mixed_nulls_total)+" ("+str(round((mixed_nulls_total/filt_total)*100,2))+"%)\n")
    print("Mixed sentences (ce or ec): "+str(mixed)+" ("+str(round((mixed/filt_total)*100,2))+"%)\n")
    print("\n C->N->E count = "+str(cne_count))
    print("E->N->C count = "+str(enc_count))
    print("C->E count = "+str(ce_count))
    print("E->C count = "+str(ec_count))

    print("\nAverage number of tokens: "+str(round(total_tokens/filt_total,2)))

    print("Global Cantonese tone frequency:")
    for k, v in sorted(tone_dict.items()):
        print("\t".join([str(k), str(v)]))
    print("\nMost frequent CN words:")
    print(dict(sorted(cn_dict.items(), key=lambda x: x[1], reverse=True)[:50]))
    print("Most frequent CN characters:")
    print(dict(sorted(cn_char_dict.items(), key=lambda x: x[1], reverse=True)[:50]))
    print("Most frequent EN words:")
    print(dict(sorted(en_dict.items(), key=lambda x: x[1], reverse=True)[:50]))
    print("Most frequent filler words:")
    print(dict(sorted(filler_dict.items(), key=lambda x: x[1], reverse=True)[:50]))

    print("\nCN -> EN:")
    print("\t".join(["Tone", "Freq", "Dist", "Global", "Norm_f", "Norm_d"]))
    for k, v in sorted(cn_en_tone_dict.items()):
        print("\t".join([str(k), str(v),
            str(round((v/sum(cn_en_tone_dict.values()))*100,2))+"%",
            str(round((v/tone_dict[k])*100,2))+"%",
            str(round(cn_en_tone_norm_dict[k],2)),
            str(round((cn_en_tone_norm_dict[k]/sum(cn_en_tone_norm_dict.values()))*100,2))+"%",
            ]))
    print("\t".join(["Number of switch points:", str(sum(cn_en_tone_dict.values()))]))
    print("Most frequent CN switch-points:")
    print(dict(sorted(cn_en_cn_dict.items(), key=lambda x: x[1], reverse=True)[:50]))
    print("Most frequent EN switch-points:")
    print(dict(sorted(cn_en_en_dict.items(), key=lambda x: x[1], reverse=True)[:50]))
    print("Most frequent CN characters switch-points:")
    print(dict(sorted(cn_en_cn_char_dict.items(), key=lambda x: x[1], reverse=True)[:50]))

    print("\nEN -> CN:")
    print("\t".join(["Tone", "Freq", "Dist", "Global", "Norm_f", "Norm_d"]))
    for k, v in sorted(en_cn_tone_dict.items()):
        print("\t".join([str(k), str(v),
            str(round((v/sum(en_cn_tone_dict.values()))*100,2))+"%",
            str(round((v/tone_dict[k])*100,2))+"%",
            str(round(en_cn_tone_norm_dict[k],2)),
            str(round((en_cn_tone_norm_dict[k]/sum(en_cn_tone_norm_dict.values()))*100,2))+"%",
            ]))
    print("\t".join(["Number of switch points:", str(sum(en_cn_tone_dict.values()))]))
    print("Most frequent EN switch-points:")
    print(dict(sorted(en_cn_en_dict.items(), key=lambda x: x[1], reverse=True)[:50]))
    print("Most frequent CN switch-points:")
    print(dict(sorted(en_cn_cn_dict.items(), key=lambda x: x[1], reverse=True)[:50]))
    print("Most frequent CN characters switch-points:")
    print(dict(sorted(en_cn_cn_char_dict.items(), key=lambda x: x[1], reverse=True)[:50]))

    print("\nCN -> nulls -> EN:")
    print("\t".join(["Tone", "Freq", "Dist", "Global", "Norm_f", "Norm_d"]))
    for k, v in sorted(cn_nulls_en_tone_dict.items()):
        print("\t".join([str(k), str(v),
            str(round((v/sum(cn_nulls_en_tone_dict.values()))*100,2))+"%",
            str(tone_dict[k]),
            str(round((v/tone_dict[k])*100,2))+"%",
            str(round(cn_nulls_en_tone_norm_dict[k],2)),
            str(round((cn_nulls_en_tone_norm_dict[k]/sum(cn_nulls_en_tone_norm_dict.values()))*100,2))+"%",
            ]))
    print("\t".join(["Number of switch points:", str(sum(cn_nulls_en_tone_dict.values()))]))
    print("Most frequent CN switch-points (CN-nulls):")
    print(dict(sorted(cn_nulls_en_cn_dict.items(), key=lambda x: x[1], reverse=True)[:50]))
    print("Most frequent CN characters switch-points (CN-nulls):")
    print(dict(sorted(cn_nulls_en_cn_char_dict.items(), key=lambda x: x[1], reverse=True)[:50]))
    print("Most frequent EN switch-points (nulls-EN):")
    print(dict(sorted(cn_nulls_en_en_dict.items(), key=lambda x: x[1], reverse=True)[:50]))
    print("Most frequent nulls words in between")
    print(dict(sorted(cn_nulls_en_nulls_dict.items(), key=lambda x: x[1], reverse=True)[:50]))
    print("Most frequent nulls words characters")
    print(dict(sorted(cn_nulls_en_nulls_char_dict.items(), key=lambda x: x[1], reverse=True)[:50]))

    print("\nEN -> nulls -> CN:")
    print("\t".join(["Tone", "Freq", "Dist", "Global", "Norm_f", "Norm_d"]))
    for k, v in sorted(en_nulls_cn_tone_dict.items()):
        print("\t".join([str(k), str(v),
            str(round((v/sum(en_nulls_cn_tone_dict.values()))*100,2))+"%",
            str(tone_dict[k]),
            str(round((v/tone_dict[k])*100,2))+"%",
            str(round(en_nulls_cn_tone_norm_dict[k],2)),
            str(round((en_nulls_cn_tone_norm_dict[k]/sum(en_nulls_cn_tone_norm_dict.values()))*100,2))+"%",
            ]))
    print("\t".join(["Number of switch points:", str(sum(en_nulls_cn_tone_dict.values()))]))
    print("Most frequent CN switch-points (CN-nulls):")
    print(dict(sorted(en_nulls_cn_cn_dict.items(), key=lambda x: x[1], reverse=True)[:50]))
    print("Most frequent EN switch-points (nulls-EN):")
    print(dict(sorted(en_nulls_cn_en_dict.items(), key=lambda x: x[1], reverse=True)[:50]))
    print("Most frequent CN characters switch-points (CN-nulls):")
    print(dict(sorted(en_nulls_cn_cn_char_dict.items(), key=lambda x: x[1], reverse=True)[:50]))
    print("Most frequent nulls switch-points in between")
    print(dict(sorted(en_nulls_cn_nulls_dict.items(), key=lambda x: x[1], reverse=True)[:50]))
    print("Most frequent nulls words characters")
    print(dict(sorted(en_nulls_cn_nulls_char_dict.items(), key=lambda x: x[1], reverse=True)[:50]))
