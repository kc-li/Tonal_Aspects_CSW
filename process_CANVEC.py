#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 13:32:05 2022

@author: Katrina Li
"""
import os
os.chdir('./data_Vietnamese')
import re
import vPhon.vPhon
from itertools import groupby

# Filler words to ignore - change their language labels to neutral
fillers = ["yeah","okay","oh","ah","mhm"]
removefillers = ["'"]
# Tone map from VPhon to Tuc's tones
tone_map = {
    1: 2, 
    2: 3, 
    3: 5, 
    4: 4, 
    5: 1, 
    6: 6, 
    7: 1, #7-5-1
    8: 6}

# Sentence totals
max_total = 0
filt_total = 0
mixed_total = 0
mixed_nulls_total = 0
mixed = 0

vne_count = 0
env_count = 0
ve_count = 0
ev_count = 0

total_tokens = 0
# Token and tone type counts
en_dict = {}
vn_dict = {}
tone_dict = {}
tone_dict = {}
filler_dict = {}
# VN -> EN counts
vn_en_en_dict = {}
vn_en_vn_dict = {}
vn_en_tone_dict = {}
# EN -> VN counts
en_vn_en_dict = {}
en_vn_vn_dict = {}
en_vn_tone_dict = {}
# CN -> Nulls -> EN counts
vn_nulls_en_vn_dict = {}
vn_nulls_en_nulls_dict = {}
vn_nulls_en_en_dict = {}
vn_nulls_en_tone_dict = {}
# EN -> Nulls -> CN counts
en_nulls_vn_vn_dict = {}
en_nulls_vn_nulls_dict = {}
en_nulls_vn_en_dict = {}
en_nulls_vn_tone_dict = {}


freq_tsv = open("processed_vie_freq.tsv", "w")

# Open the CanVEC file
with open("CanVEC.csv") as text, open("processed_canvec.tsv", "w") as out_tsv:
    # Each utterance/clause spans 6 lines; store each group here
    clause = []
    # Loop throught the lines
    for line in text:
        # Strip whitespace and add the line to the clause
        clause.append(line.strip())
        # If there are not 6 lines in clause, go to the next line
        if len(clause) != 6: continue
        # Count the total number of clauses
        max_total += 1
        # Get the tokens (converted to lower-case), langs, pos tags and clause lang
        toks = clause[1].split(",")[-1].lower().split(" | ")
        tok_pos = clause[2].split(",")[-1].split(" | ")
        tok_langs = clause[3].split(",")[-1].split(" | ")
        clause_lang = clause[4].split(",")[-1]
        # Removefiller: remove the ''' symbol
        # Get the ids of filler tokens
        removefiller_ids = [i for i in range(0, len(toks)) if toks[i] in removefillers]
        # Update toks, tok_pos and tok_langs with fillers removed
        toks = [toks[i] for i in range(0, len(toks)) if i not in removefiller_ids]
        tok_pos = [tok_pos[i] for i in range(0, len(tok_pos)) if i not in removefiller_ids]
        tok_langs = [tok_langs[i] for i in range(0, len(tok_langs)) if i not in removefiller_ids]
        # Normal fillers: don't remove, only replace the language symbol
        # Get the ids of filler tokens
        filler_ids = [i for i in range(0, len(toks)) if toks[i] in fillers]
        # Update tok_langs
        for i in range(0, len(tok_langs)):
            if i in filler_ids:
                tok_langs[i] = "@non"
            if tok_pos[i] == "INTJ":
                tok_langs[i] = "@non"
        # Save tok tones here
        tok_tones = []

        # Calculate the total number of tokens
        total_tokens += len(toks)

        # Fill the global token and tone dictionaries
        for i in range(0, len(toks)):
            # English tokens
            if tok_langs[i] == "@eng":
                # Token frequency
                if toks[i] in en_dict: en_dict[toks[i]] += 1
                else: en_dict[toks[i]] = 1
                # Set tone as 0
                tok_tones.append(0)
            # Vietnamese tokens
            elif tok_langs[i] == "@vie":
                # Token frequency
                if toks[i] in vn_dict: vn_dict[toks[i]] += 1
                else: vn_dict[toks[i]] = 1
                
                # Get the phonetic representation of the token
                ipa = vPhon.vPhon.main([toks[i]], dialect="s", eight=True, nosuper=True, tokenize=True)
                # Split the IPA representation into syllables
                tone = ipa.split("_")
                # Loop through syllables
                for j in range(0, len(tone)):
                    # If the syllable ends with a digit tone number
                    if tone[j] and tone[j][-1].isdigit():
                        # Convert the tone number to an int and map to Tuc
                        tone[j] = tone_map[int(tone[j][-1])]
                        # Save it in the tone_dict and tok_tones
                        if tone[j] in tone_dict: tone_dict[tone[j]] += 1
                        else: tone_dict[tone[j]] = 1
                    # Otherwise, set tone as 0
                    else: tone[j] = 0
                # Append tone to tok_tones
                if len(tone) == 1: tok_tones.append(tone[0])
                # Multisyllabic tokens are a list
                else: tok_tones.append(tone)
                
            # Language neutral tones
            elif tok_langs[i] == "@non":
                # Token frequency
                if toks[i] in filler_dict: filler_dict[toks[i]] += 1
                else: filler_dict[toks[i]] = 1
                # Set tone as 0
                tok_tones.append(0)
            # Some tokens are labelled @univ, but we ignore these and just set a null tone
            else: tok_tones.append(0)
        
        # print(toks)
        # print(tok_pos)    
        # print(tok_tones)
        # print(tok_langs)
        # Make sure we have assigned a tone to each token
        assert len(toks) == len(tok_tones)


        # count three-way transitions
        # Add explicit index to each language tag
        langs_ids = [(tok_langs[i], i) for i in range(0, len(tok_langs))]
        # Group by the language
        lang_groups = [(lang, list(group)) for lang, group in groupby(langs_ids, lambda x: x[0])]
        vne = 0
        env = 0
        vn_null_en = [0]*len(toks)
        en_null_vn = [0]*len(toks)
        info_vne_null = ["none"] * len(toks)
        info_vne_en = ["none"] * len(toks)
        info_env_null = ["none"] * len(toks)
        info_env_en = ["none"] * len(toks)
        # Only sentences with at least 3 groups could be enc or cne
        if len(lang_groups) > 2 and len(set(tok_langs)) == 3:
            # Loop through the groups
            for i in range(0, len(lang_groups)-2):
                # VN -> Null -> EN
                if lang_groups[i][0] == "@vie" and lang_groups[i+1][0] == "@non" and lang_groups[i+2][0] == "@eng":
                    vne = 1
                    vne_count += 1
                    # Useful info
                    vn_null_en[lang_groups[i][1][-1][-1]] = 1 # for regression analysis
                    vn_token = toks[lang_groups[i][1][-1][-1]]
                    
                    # Get the phonetic representation of the token
                    ipa = vPhon.vPhon.main(vn_token, dialect="s", eight=True, nosuper=True, tokenize=True)
                    # Split the IPA representation into syllables, and only keep the last one
                    tone = ipa.split("_")[-1]
                    if tone and tone[-1].isdigit():
                        tone = tone_map[int(tone[-1])]
                        if tone in vn_nulls_en_tone_dict: vn_nulls_en_tone_dict[tone] += 1
                        else: vn_nulls_en_tone_dict[tone] = 1
                    # Otherwise, set tone = 0
                    else: tone = 0
                    
                    null_tokens = toks[lang_groups[i+1][1][0][-1]:lang_groups[i+1][1][-1][-1]+1]
                    info_vne_null[lang_groups[i][1][-1][-1]] = null_tokens # for regression analysis
                    en_token = toks[lang_groups[i+2][1][0][-1]]
                    info_vne_en[lang_groups[i][1][-1][-1]] = en_token # for regression analysis
                    # Add to dictionary
                    if vn_token in vn_nulls_en_vn_dict: vn_nulls_en_vn_dict[vn_token] += 1
                    else: vn_nulls_en_vn_dict[vn_token] = 1
                    if en_token in vn_nulls_en_en_dict: vn_nulls_en_en_dict[en_token] += 1
                    else: vn_nulls_en_en_dict[en_token] = 1
                    # Dealing with null tokens: small loop
                    for item in null_tokens:
                        if item in vn_nulls_en_nulls_dict: vn_nulls_en_nulls_dict[item] += 1
                        else: vn_nulls_en_nulls_dict[item] = 1
                # EN -> Null -> CN
                if lang_groups[i][0] == "@eng" and lang_groups[i+1][0] == "@non" and lang_groups[i+2][0] == "@vie":
                    env = 1
                    env_count += 1

                    # Useful info
                    en_null_vn[lang_groups[i+2][1][0][-1]] = 1
                    en_token = toks[lang_groups[i][1][-1][-1]]
                    null_tokens = toks[lang_groups[i+1][1][0][-1]:lang_groups[i+1][1][-1][-1]+1]
                    info_env_null[lang_groups[i+2][1][0][-1]] = null_tokens # for regression analysis, attached to the chinese token
                    info_env_en[lang_groups[i+2][1][0][-1]] = en_token # for regression analysis
                    vn_token = toks[lang_groups[i+2][1][0][-1]]
                    
                    # Get the phonetic representation of the token
                    ipa = vPhon.vPhon.main(vn_token, dialect="s", eight=True, nosuper=True, tokenize=True)
                    # Split the IPA representation into syllables, and only keep the initial one
                    tone = ipa.split("_")[0]
                    if tone and tone[-1].isdigit():
                        tone = tone_map[int(tone[-1])]
                        if tone in en_nulls_vn_tone_dict: en_nulls_vn_tone_dict[tone] += 1
                        else: en_nulls_vn_tone_dict[tone] = 1
                    # Otherwise, set tone = 0
                    else: tone = 0                    

                    if vn_token in en_nulls_vn_vn_dict: en_nulls_vn_vn_dict[vn_token] += 1
                    else: en_nulls_vn_vn_dict[vn_token] = 1
                    for item in null_tokens:
                        if item in en_nulls_vn_nulls_dict: en_nulls_vn_nulls_dict[item] += 1
                        else: en_nulls_vn_nulls_dict[item] = 1
                    for item in null_tokens:
                        if item in en_nulls_vn_nulls_dict: en_nulls_vn_nulls_dict[item] += 1
                        else: en_nulls_vn_nulls_dict[item] = 1
                    if en_token in en_nulls_vn_en_dict: en_nulls_vn_en_dict[en_token] += 1
                    else: en_nulls_vn_en_dict[en_token] = 1

        # Sanity check ### Print list and check!
        assert len(vn_null_en) == len(en_null_vn)== len(toks) == len(info_vne_null)== len(info_vne_en) == len(info_env_null)== len(info_env_en)

        if vne ==1 or env == 1: mixed_nulls_total +=1

        # From here on, only look at mixed clauses
        if clause_lang == "@mix":
            # Increment mixed total (count the ve transition)
            mixed_total += 1
            # Loop through tokens again to look for switches
            for i in range(1, len(toks)):
                # VN -> EN
                if tok_langs[i] == "@eng" and tok_langs[i-1] == "@vie":
                    ve_count += 1
                    # Save the en token
                    if toks[i] in vn_en_en_dict: vn_en_en_dict[toks[i]] += 1
                    else: vn_en_en_dict[toks[i]] = 1
                    # Save the vn token
                    if toks[i-1] in vn_en_vn_dict: vn_en_vn_dict[toks[i-1]] += 1
                    else: vn_en_vn_dict[toks[i-1]] = 1
                    # Save the tone if it is not 0, which is usually an error (rare)
                    if type(tok_tones[i-1]) == list:
                        if tok_tones[i-1][-1] == 0: continue
                        if tok_tones[i-1][-1] in vn_en_tone_dict: 
                            vn_en_tone_dict[tok_tones[i-1][-1]] += 1
                        else: vn_en_tone_dict[tok_tones[i-1][-1]] = 1
                    else:
                        if tok_tones[i-1] == 0: continue
                        if tok_tones[i-1] in vn_en_tone_dict: 
                            vn_en_tone_dict[tok_tones[i-1]] += 1
                        else: vn_en_tone_dict[tok_tones[i-1]] = 1
                # EN -> VN
                elif tok_langs[i] == "@vie" and tok_langs[i-1] == "@eng":
                    ev_count += 1
                    # Save the en token
                    if toks[i-1] in en_vn_en_dict: en_vn_en_dict[toks[i-1]] += 1
                    else: en_vn_en_dict[toks[i-1]] = 1
                    # Save the cn token
                    if toks[i] in en_vn_vn_dict: en_vn_vn_dict[toks[i]] += 1
                    else: en_vn_vn_dict[toks[i]] = 1
                    # Save the tone if it is not 0, which is usually an error (rare)
                    if type(tok_tones[i]) == list:
                        if tok_tones[i][0] == 0: continue
                        if tok_tones[i][0] in en_vn_tone_dict: 
                            en_vn_tone_dict[tok_tones[i][0]] += 1
                        else: en_vn_tone_dict[tok_tones[i][0]] = 1
                    else:
                        if tok_tones[i] == 0: continue
                        if tok_tones[i] in en_vn_tone_dict: 
                            en_vn_tone_dict[tok_tones[i]] += 1
                        else: en_vn_tone_dict[tok_tones[i]] = 1
                        

        # Write to output
        for i in range(0, len(toks)):
            # Store output items here
            out = []
            out.append(toks[i]) #Column1: token
            out.append(tok_langs[i]) #Column2: language symbol
            out.append(tok_pos[i]) #Column3: post-tagging
            if tok_langs[i] != "@vie": out.append("-") #Column4 tone
            else: out.append(str(tok_tones[i])) #Column4 tone
            # Attach true switching infomation
            if tok_langs[i] == '@vie':
                if i!=0 and tok_langs[i-1] == '@eng': #EN-> VN
                    out.append("1") # Column 5: if EN-VN
                    out.append(toks[i-1]) # Column 6: record the EN before
                else:
                    out.append("0") # Column 5:
                    out.append("0") # Column 6:
                if i != len(toks)-1 and tok_langs[i+1] == "@eng":
                    out.append("1") # Column 7: if VN-EN
                    out.append(toks[i+1]) # Column 8: record the EN after
                else:
                    out.append("0") # Column 7
                    out.append("0") # Column 8
            else:
                out.append("0") # Column 5
                out.append("0") # Column 6
                out.append("0") # Column 7
                out.append("0") # Column 8
            out.append(str(vn_null_en[i])) # Column 9: if contains cn_null_en
            out.append(str(info_vne_null[i])) # Column 10: nulls string
            out.append(str(info_vne_en[i]))# Column 11: en string
            out.append(str(en_null_vn[i])) # Column 12: if contains en_null_cn
            out.append(str(info_env_null[i])) # Column 13: nulls string
            out.append(str(info_env_en[i]))# Column 14: en string

            # Column 15: Also output source information (speaker)
            out.append(clause[0].split(",")[0])
            # Column 16: Generation information
            out.append(clause[0].split(",")[1])
            # Column 17: Sentence info
            out.append(str(max_total))
            # Column 18: word in sentence info
            out.append(str(i))
            # Write to output
            out_tsv.write("\t".join(out)+"\n")
        out_tsv.write("\n")

        # Reset the clause for the next one (ou)
        clause = []

# Calculate tone distribution normalised by word frequency
# Full dictionary for freq
for tok, cnt in vn_dict.items():
    freq = cnt/sum(vn_dict.values())
    out = []
    out.append(tok)
    out.append(str(freq))
    out.append("vn")
    freq_tsv.write("\t".join(out)+ "\n")
for tok, cnt in en_dict.items():
    freq = cnt/sum(en_dict.values())
    out = []
    out.append(tok)
    out.append(str(freq))
    out.append("en")
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
# Extract tones: tones have been extracted in the correct number
for tok, cnt in tone_dict.items():
    print(str(tok), str(cnt))
    freq = cnt/sum(tone_dict.values())
    out = []
    out.append(str(tok))
    out.append(str(freq))
    out.append("tone")
    freq_tsv.write("\t".join(out)+ "\n")
freq_tsv.write("\n")

# VN- > EN
vn_en_tone_norm_dict = {}
for tok, cnt in vn_en_vn_dict.items():                       
    # Get the tone
    ipa = vPhon.vPhon.main(tok, dialect="s", eight=True, nosuper=True, tokenize=True)
    # Split the IPA representation into syllables, and only keep the last one
    tone = ipa.split("_")[-1]
    if tone and tone[-1].isdigit():
        tone = tone_map[int(tone[-1])]
    # Otherwise, set tone = 0
    else: tone = 0

    # Add the tone to the dict
    if tone not in vn_en_tone_norm_dict: vn_en_tone_norm_dict[tone] = 0
    # Calculate word frequency in the corpus
    freq = vn_dict[tok]/sum(vn_dict.values())
    # Weight the tone at a switch point by frequency
    vn_en_tone_norm_dict[tone] += freq*cnt
# EN- > VN
en_vn_tone_norm_dict = {}
for tok, cnt in en_vn_vn_dict.items():
    # Get the tone
    ipa = vPhon.vPhon.main(tok, dialect="s", eight=True, nosuper=True, tokenize=True)
    # Split the IPA representation into syllables, and only keep the initial one
    tone = ipa.split("_")[0]
    if tone and tone[-1].isdigit():
        tone = tone_map[int(tone[-1])]
    # Otherwise, set tone = 0
    else: tone = 0

    # Add the tone to the dict
    if tone not in en_vn_tone_norm_dict: en_vn_tone_norm_dict[tone] = 0
    # Calculate word frequency in the corpus
    freq = vn_dict[tok]/sum(vn_dict.values())
    # Weight the tone at a switch point by frequency
    en_vn_tone_norm_dict[tone] += freq*cnt

# VN- > nulls -> EN
vn_nulls_en_tone_norm_dict = {}
for tok, cnt in vn_nulls_en_vn_dict.items():
    # Get the tone
    ipa = vPhon.vPhon.main(tok, dialect="s", eight=True, nosuper=True, tokenize=True)
    # Split the IPA representation into syllables, and only keep the last one
    tone = ipa.split("_")[-1]
    if tone and tone[-1].isdigit():
        tone = tone_map[int(tone[-1])]
    # Otherwise, set tone = 0
    else: tone = 0
    # Add the tone to the dict
    if tone not in vn_nulls_en_tone_norm_dict: vn_nulls_en_tone_norm_dict[tone] = 0
    # Calculate word frequency in the corpus
    freq = vn_dict[tok]/sum(vn_dict.values())
    # Weight the tone at a switch point by frequency
    vn_nulls_en_tone_norm_dict[tone] += freq*cnt

# EN- > nulls -> VN
en_nulls_vn_tone_norm_dict = {}
for tok, cnt in en_nulls_vn_vn_dict.items():
    # Get the tone
    ipa = vPhon.vPhon.main(tok, dialect="s", eight=True, nosuper=True, tokenize=True)
    # Split the IPA representation into syllables, and only keep the initial one
    tone = ipa.split("_")[0]
    if tone and tone[-1].isdigit():
        tone = tone_map[int(tone[-1])]
    # Otherwise, set tone = 0
    else: tone = 0
    # Add the tone to the dict
    if tone not in en_nulls_vn_tone_norm_dict: en_nulls_vn_tone_norm_dict[tone] = 0
    # Calculate word frequency in the corpus
    freq = vn_dict[tok]/sum(vn_dict.values())
    # Weight the tone at a switch point by frequency
    en_nulls_vn_tone_norm_dict[tone] += freq*cnt

# Print results
print("\nTotal sentences: "+str(max_total))
# mixed_total here is ve or ev
print("Mixed sentences (ve or ev): "+str(mixed_total)+" ("+str(round((mixed_total/max_total)*100,2))+"%)\n")
print("Mixed sentences (cne or enc): "+str(mixed_nulls_total)+" ("+str(round((mixed_nulls_total/max_total)*100,2))+"%)\n")
print("\n V->N->E count = "+str(vne_count))
print("E->N->V count = "+str(env_count))
print("V->E count = "+str(ve_count))
print("E->V count = "+str(ev_count))

print("\nAverage number of tokens: "+str(round(total_tokens/max_total,2)))

print("\nGlobal Vietnamese tone frequency:")
for k, v in sorted(tone_dict.items()):
    print("\t".join([str(k), str(v)]))
print("\nMost frequent VN words:")
print(dict(sorted(vn_dict.items(), key=lambda x: x[1], reverse=True)[:50]))
print("Most frequent EN words:")
print(dict(sorted(en_dict.items(), key=lambda x: x[1], reverse=True)[:50]))
print("Most frequent filler words:")
print(dict(sorted(filler_dict.items(), key=lambda x: x[1], reverse=True)[:50]))


print("\nVN -> EN:")
print("\t".join(["Tone", "Freq", "Dist", "Global", "Norm_f", "Norm_d"]))
for k, v in sorted(vn_en_tone_dict.items()):
    print("\t".join([str(k), str(v),
        str(round((v/sum(vn_en_tone_dict.values()))*100,2))+"%",
        str(round((v/tone_dict[k])*100,2))+"%",
        str(round(vn_en_tone_norm_dict[k],2)),
        str(round((vn_en_tone_norm_dict[k]/sum(vn_en_tone_norm_dict.values()))*100,2))+"%",
        ]))
print("\t".join(["Number of switch points:", str(sum(vn_en_tone_dict.values()))]))
print("Most frequent VN switch-points:")
print(dict(sorted(vn_en_vn_dict.items(), key=lambda x: x[1], reverse=True)[:20]))
print("Most frequent EN switch-points:")
print(dict(sorted(vn_en_en_dict.items(), key=lambda x: x[1], reverse=True)[:20]))

print("\nEN -> VN:")
print("\t".join(["Tone", "Freq", "Dist", "Global", "Norm_f", "Norm_d"]))
for k, v in sorted(en_vn_tone_dict.items()):
    print("\t".join([str(k), str(v),
        str(round((v/sum(en_vn_tone_dict.values()))*100,2))+"%",
        str(round((v/tone_dict[k])*100,2))+"%",
        str(round(en_vn_tone_norm_dict[k],2)),
        str(round((en_vn_tone_norm_dict[k]/sum(en_vn_tone_norm_dict.values()))*100,2))+"%",
        ]))
print("\t".join(["Number of switch points:", str(sum(en_vn_tone_dict.values()))]))
print("Most frequent EN switch-points:")
print(dict(sorted(en_vn_en_dict.items(), key=lambda x: x[1], reverse=True)[:20]))
print("Most frequent VN switch-points:")
print(dict(sorted(en_vn_vn_dict.items(), key=lambda x: x[1], reverse=True)[:20]))

print("\nVN -> nulls -> EN:")
print("\t".join(["Tone", "Freq", "Dist", "Global", "Norm_f", "Norm_d"]))
for k, v in sorted(vn_nulls_en_tone_dict.items()):
    print("\t".join([str(k), str(v),
        str(round((v/sum(vn_nulls_en_tone_dict.values()))*100,2))+"%",
        str(tone_dict[k]),
        str(round((v/tone_dict[k])*100,2))+"%",
        str(round(vn_nulls_en_tone_norm_dict[k],2)),
        str(round((vn_nulls_en_tone_norm_dict[k]/sum(vn_nulls_en_tone_norm_dict.values()))*100,2))+"%",
        ]))
print("\t".join(["Number of switch points:", str(sum(vn_nulls_en_tone_dict.values()))]))
print("Most frequent CN switch-points (CN-nulls):")
print(dict(sorted(vn_nulls_en_vn_dict.items(), key=lambda x: x[1], reverse=True)[:50]))
print("Most frequent EN switch-points (nulls-EN):")
print(dict(sorted(vn_nulls_en_en_dict.items(), key=lambda x: x[1], reverse=True)[:50]))
print("Most frequent nulls words in between")
print(dict(sorted(vn_nulls_en_nulls_dict.items(), key=lambda x: x[1], reverse=True)[:50]))

print("\nEN -> nulls -> CN:")
print("\t".join(["Tone", "Freq", "Dist", "Global", "Norm_f", "Norm_d"]))
for k, v in sorted(en_nulls_vn_tone_dict.items()):
    print("\t".join([str(k), str(v),
        str(round((v/sum(en_nulls_vn_tone_dict.values()))*100,2))+"%",
        str(tone_dict[k]),
        str(round((v/tone_dict[k])*100,2))+"%",
        str(round(en_nulls_vn_tone_norm_dict[k],2)),
        str(round((en_nulls_vn_tone_norm_dict[k]/sum(en_nulls_vn_tone_norm_dict.values()))*100,2))+"%",
        ]))
print("\t".join(["Number of switch points:", str(sum(en_nulls_vn_tone_dict.values()))]))
print("Most frequent CN switch-points (CN-nulls):")
print(dict(sorted(en_nulls_vn_vn_dict.items(), key=lambda x: x[1], reverse=True)[:50]))
print("Most frequent EN switch-points (nulls-EN):")
print(dict(sorted(en_nulls_vn_en_dict.items(), key=lambda x: x[1], reverse=True)[:50]))
print("Most frequent nulls switch-points in between")
print(dict(sorted(en_nulls_vn_nulls_dict.items(), key=lambda x: x[1], reverse=True)[:50]))
