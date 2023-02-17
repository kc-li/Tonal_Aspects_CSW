
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 10 23:24:34 2022

@author: Katrina Li
"""

from glob import glob
import re
import jieba
import spacy
from itertools import groupby
from pypinyin import pinyin, lazy_pinyin, Style, load_phrases_dict, load_single_dict

# Adjust the dictionary loading order
# Single pinyin data
from pypinyin_dict.pinyin_data import kxhc1983 
kxhc1983.load()
# most frequent single word
from pypinyin_dict.pinyin_data import kmandarin_8105
kmandarin_8105.load()
# Phrase pinyin data
# First load large_pinyin, hope some of them will be overwritten
from pypinyin_dict.phrase_pinyin_data import large_pinyin
large_pinyin.load()
from pypinyin_dict.phrase_pinyin_data import zdic_cibs
zdic_cibs.load()
from pypinyin_dict.phrase_pinyin_data import di
di.load()


# English spacy
nlp_en = spacy.load("en_core_web_sm", disable=["parser", "ner"])
# Chinese spacy
nlp_cn = spacy.load("zh_core_web_sm", disable=["parser", "ner"])

# Pinyin modification
load_single_dict({ord('咯'): 'lo5'})
load_single_dict({ord('咧'): 'lie5'})
#load_phrases_dict({'一直': [['yì'], ['zhí']]})
#load_phrases_dict({'一下': [['yí'], ['xià']]})


def flat(l):
    for k in l:
        if not isinstance(k, (list, tuple)):
            yield k
        else:
            yield from flat(k)


# Get all the filenames
hlvc_files = glob("./data_Mandarin/*.txt")

# Fillers list
fillers = ['okay', 'er','yeah','呃','嗯','hum','哦','oh','诶','eh','a','ah','啊','lah','啦','lor','loh','咯','hor','嘛','吗','嘞','c','p','j','n','u','e','s','nat','orh','ei','yah','ha','mah','呢']

# Output file for statistical analysis
out_tsv = open("./output/processed_man.tsv", "w")
pinyin_tsv = open("./output/pinyin_check.tsv", "w")
freq_tsv = open("./output/processed_man_freq.tsv", "w")
corpus_tsv = open("./output/processed_man_corpus.tsv", "w")
corpus_switch_tsv = open("./output/processed_switch_corpus.tsv", "w")

# Sentence totals
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
        for line in text:
            # Count total number of sentences BEFORE filtering
            max_total += 1
            # Remove the format symbols
            line = line.strip() # remove the new line at the end
            line = re.sub("<v-noise>","",line)
            # Remove space between Chinese characters, to falicitate pinyin conversion
            line = re.sub(u'(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])',"",line)
            # Convert to lower case and trailing sapce
            line = line.lower()+" "
            column = line.split()
            # Remove the first element which is the marker
            column.pop(0)
            sent_mix = column
            #print(sent_mix)

            # First round lang symbol
            # Set langs label (Chinese concatenated)
            langs_block = []
            for i in range(0,len(sent_mix)):
                lang_block = "cn" if re.match(u'[\u4e00-\u9fff\u3400-\u4dbf]',sent_mix[i]) else "en"
                langs_block.append(lang_block)
                
            # sanity check
            assert len(sent_mix) == len(langs_block)
            
            #Get the switch point indices (Add 0 to the start)
            switch_ids = [0] + [i for i in range(1, len(langs_block)) if langs_block[i] != langs_block[i-1]]
            pos = []
            sent_cn = []
            sent_word = [] #pinyin, contains all syllables
            sent_all = []
            sent_cn_all = []
            
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
                    chunk_process = spacy.tokens.Doc(nlp_en.vocab, chunk)
                    nlp_en(chunk_process)
                    sent_cn.extend(chunk)
                    sent_word.extend(chunk)
                    sent_all.extend(chunk)
                    sent_cn_all.extend(chunk)
                else: 
                    # jieba process string, not list
                    chunk = ''.join(chunk)
                    # Atomize all characters
                    chars = [item for item in chunk]
                    sent_cn_all.extend(chars)
                    # jiaba segmentation precedes the spacy model - this is reuiqred, and help control the number of segments
                    seg = jieba.lcut(chunk)
                    # add to the same list
                    sent_cn.extend(seg)
                    chunk_process = spacy.tokens.Doc(nlp_cn.vocab, seg)
                    nlp_cn(chunk_process)
                    
                    #Convert to pinyin, for each segmented part
                    for item in seg:
                        pinyin_list= pinyin(item, style=Style.TONE3, neutral_tone_with_five=True)
                        # Exclude polyphones
                        pinyin_word = [char[0] for char in pinyin_list]
                        # Keep the pinyin for each word and export
                        sent_word.append(' '.join(pinyin_word))

                        # Atomize all pinyin
                        sent_all.extend(pinyin_word)       
                #Get the POS tags
                tags = [tok.pos_ for tok in chunk_process]
                pos.extend(tags)
            
            # Store token langs here
            langs = []
            # Loop through tokens in all remaining sentences
            for i in range(0,len(sent_word)):
                # Get the token lang; Mandarin words in sent end with exactly 1 number; fillers include Chinese characters so needs to be processed in sent_cn
                if sent_cn[i] in fillers or pos[i] == "PROPN" or pos[i] == "INTJ":
                    lang = 'null'
                elif re.search("\D\d$", sent_word[i]):
                    lang = 'cn'
                else:
                    lang = 'en'
                langs.append(lang)
                             
            # print(sent_mix)            
            # print(sent_cn)  
            # print(sent_word)
            # print(pos)
            # print(langs)
            # print(sent_all)
            # print(sent_cn_all)
               
            # Sanity check to make sure we have the correct number of pos tags
            assert len(sent_cn) == len(sent_word) == len(pos) == len(langs)
            assert len(sent_all) == len(sent_cn_all)   
            
            # Calculate the total number of tokens
            total_tokens += len(sent_cn)
            
            # Now we export the corpus for manual examination
            corpus_tsv.write(str(max_total) + "\t" + line.split()[0] + "\t" + "sent_mix" + "\t" + " | ".join(sent_mix)+"\n")
            corpus_tsv.write(str(max_total) + "\t" + line.split()[0] + "\t" + "sent_cn" + "\t" + " | ".join(sent_cn)+"\n")
            corpus_tsv.write(str(max_total) + "\t" + line.split()[0] + "\t" + "sent_word" + "\t" + " | ".join(sent_word)+"\n")
            corpus_tsv.write(str(max_total) + "\t" + line.split()[0] + "\t" + "pos" + "\t" + " | ".join(pos)+"\n")
            corpus_tsv.write(str(max_total) + "\t" + line.split()[0] + "\t" + "langs" + "\t" + " | ".join(langs)+"\n")
            corpus_tsv.write(str(max_total) + "\t" + line.split()[0] + "\t" + "sent_all" + "\t" + " | ".join(sent_all)+"\n")
            corpus_tsv.write(str(max_total) + "\t" + line.split()[0] + "\t" + "sent_cn_all" + "\t" + " | ".join(sent_cn_all)+"\n")          
  
            # Count total number of sentences AFTER filtering
            filt_total += 1

            # Calculate the freq based on 'potential switch points':
            # For English words, every word is counted; for Chinese words, only word at boundary positions.
            for i in range(0,len(sent_word)):
                tok = sent_word[i]
                tok_cn = sent_cn[i]
                # Keep track of English tokens
                if langs[i] == "en":
                    if tok in en_dict: en_dict[tok] += 1
                    else: en_dict[tok] = 1
                # Keep track of Mandarin tokens
                elif langs[i] == "cn": 
                    # The cn tokens can be stored as a whole (no need to decompose as tonal information)
                    if tok_cn in cn_char_dict: cn_char_dict[tok_cn] += 1
                    else: cn_char_dict[tok_cn] = 1                    
                    # check if there is only one character in the Chinese group or multiple
                    lentokens = len(sent_word[i].split())
                    # if only one word in the token
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

            # Three-way switches
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
                    # EN -> Null -> CN
                    if lang_groups[i][0] == "en" and lang_groups[i+1][0] == "null" and lang_groups[i+2][0] == "cn":
                        enc = 1
                        enc_count += 1
                        # Useful info
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
                       # Save the cn char token
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
                       # Save the cn char token
                       if sent_cn[i] in en_cn_cn_char_dict: en_cn_cn_char_dict[sent_cn[i]] += 1
                       else: en_cn_cn_char_dict[sent_cn[i]] = 1        

            if ce ==1 or ec == 1: 
                mixed +=1
                # Now we export the corpus for manual examination
                corpus_switch_tsv.write(str(max_total) + "\t" + line.split()[0] + "\t" + "sent_mix" + "\t" + " | ".join(sent_mix)+"\n")
                corpus_switch_tsv.write(str(max_total) + "\t" + line.split()[0] + "\t" + "sent_cn" + "\t" + " | ".join(sent_cn)+"\n")
                corpus_switch_tsv.write(str(max_total) + "\t" + line.split()[0] + "\t" + "sent_word" + "\t" + " | ".join(sent_word)+"\n")
                corpus_switch_tsv.write(str(max_total) + "\t" + line.split()[0] + "\t" + "pos" + "\t" + " | ".join(pos)+"\n")
                corpus_switch_tsv.write(str(max_total) + "\t" + line.split()[0] + "\t" + "langs" + "\t" + " | ".join(langs)+"\n")
                corpus_switch_tsv.write(str(max_total) + "\t" + line.split()[0] + "\t" + "sent_all" + "\t" + " | ".join(sent_all)+"\n")
                corpus_switch_tsv.write(str(max_total) + "\t" + line.split()[0] + "\t" + "sent_cn_all" + "\t" + " | ".join(sent_cn_all)+"\n") 

            # Write to output
            for i in range(0, len(sent_word)):
                # Store output items here
                out = []
                out.append(sent_word[i])
                out.append(sent_cn[i])
                out.append(langs[i])
                out.append(pos[i])
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
                out.append(line.split()[0])
                # Column 18: Sentence info
                out.append(str(max_total))
                # Column 19: word in sentence info
                out.append(str(i))
                # Column 20: filename (knowing which language dominated)
                out.append(str(filename))
                # Write to output
                out_tsv.write("\t".join(out)+"\n")
            out_tsv.write("\n") # This line probably shouldn't be here!!!


# Output the most freuqent Chinese characters and the pinyin, for manual inspection
checkpinyin_dict = dict(sorted(cn_char_dict.items(), key=lambda x: x[1], reverse=True)[:500])
for item, freq in checkpinyin_dict.items():
    pinyin_list= pinyin(item, style=Style.TONE3, neutral_tone_with_five=True)
    # Exclude polyphones
    pinyin_word = [char[0] for char in pinyin_list]
    out2 = []
    out2.append(item)
    out2.append(str(pinyin_word))
    out2.append(str(freq))
    pinyin_tsv.write("\t".join(out2)+"\n")
pinyin_tsv.write("\n")



# Calculate tone distribution normalised by word frequency
# Full dictionary for freq
for tok, cnt in cn_dict.items():
    freq = cnt/sum(cn_dict.values())
    out = []
    out.append(tok)
    out.append(str(freq))
    out.append("cn_dict")
    freq_tsv.write("\t".join(out)+ "\n")
freq_tsv.write("\n")
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
    print(tok, str(cnt))
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

print("Global Mandarin tone frequency:")
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
print("Most frequent CN characters switch-points:")
print(dict(sorted(cn_en_cn_char_dict.items(), key=lambda x: x[1], reverse=True)[:50]))
print("Most frequent EN switch-points:")
print(dict(sorted(cn_en_en_dict.items(), key=lambda x: x[1], reverse=True)[:50]))


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
print("Most frequent CN switch-points:")
print(dict(sorted(en_cn_cn_dict.items(), key=lambda x: x[1], reverse=True)[:50]))
print("Most frequent CN characters switch-points:")
print(dict(sorted(en_cn_cn_char_dict.items(), key=lambda x: x[1], reverse=True)[:50]))
print("Most frequent EN switch-points:")
print(dict(sorted(en_cn_en_dict.items(), key=lambda x: x[1], reverse=True)[:50]))

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

