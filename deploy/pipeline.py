import pandas as pd
import numpy as np
import pickle
from features import get_fragments
from word_embedding import WordEmbedding
from normalizer import run_normalize
import re 
# from sklearn.svm import SVC

import sklearn

txt = "Si Maya ay pangalawa sa tatlong magkakapatid. Nakatira sila sa abang tahanan ng Barangay San Isidro, lungsod ng Heneral Santos. Mapagmahal ang kanyang mga magulang na sina Mang Mando at Aling Memay. Mahilig maglaro si Maya sa kanyang manika, na bigay ng kanyang Ante Anna. Araw ng Sabado, naglalaro ang magkakapatid, “Kuya Miguel, maglalaro tayo ng manika ko”, tawag ni Maya sa kanyang nakatatandang kapatid. Bog,Bog,Bog “Sandali lamang naglalaro pa kami ni Marco ng basketball”, tugon naman ng kanyang Kuya Miguel. Si Marco ay ang kanilang nakababatang kapatid. Mag-isa na lamang naglaro si Maya sa kanyang manika na pinangalanan niyang Meya. “Mmmmm….mmmmm, tulog na!tulog na! Meya kong mahal”. Awit ni Maya habang inuugoy ang kanyang manika. Inilapag niya ito sa loob ng kahon bilang kama ng kanyang manika. Pumunta si Maya sa kusina upang uminom ng tubig. Nakita niya sa labas ang dalawa niyang kapatid na naglalaro ng bola. “Kuya Miguel, ipasa mo rin sa akin ang bola”, nakangiting sigaw ni Maya. “Eto, saluin mo ang bola Maya”. Tugon ni Kuya Miguel. “Wow, ang galing mo Ate Maya, nasalo mo”, papuri ni Marco sa kanyang ate. Lumabas sila sa kanilang bahay upang maglaro pa ng habol-habolan. Dumating ang kanyang Nanay Memay mula sa palengke. “Mano po Inay”, patakbong lumapit ang magkakapatid. “Maaari po bang maglaro pa kami sa labas, Inay?  Paalam ni Kuya Miguel kay Aling Memay. “O, sige mga anak, basta maging maingat at huwag mag-aaway,” paalala ni Aling Memay. “Huli ka, Marco!”  “Ako naman habulin mo Marco,” tugon ni Maya.  “Mag-iingat kayo, baka madapa kayo,” paalala ng kanilang Kuya Miguel. Sa kusina, inaayos ni Aling Memay ang kanyang pinamili. Meron siyang mga prutas; mangga, at mansanas.   Gulay at manok.  Nagligpit siya ng mga kalat sa loob ng sala, itinabi ang kahon at ipinasok sa bodega.Matapos magluto, tinawag na niya ang kanyang mga anak upang kumain ng meryenda. “Nandito na po kami Inay”, bungad ni Maya. Binigyan sila ng tinapay at juice ni Aling Memay.   “Salamat po, Inay,” nabusog po kami. Sabay sabi nina Kuya Miguel, Maya at Marco.” Kinabukasan. “Magandang umaga, Maya,” masayang bati ng kanyang manika. “Magandang umaga naman Meya,” tugon ni Maya sa kanyang Manika. “Halika maglaro tayo ng tagu-taguan,” sambit ni Meya Manika. “Isa, dalawa, tatlo, magtago kana.”   Hinanap ni Maya si Meya Manika, sumilip sya sa ilalim ng kama, “Huli ka!” ngunit walang Meya.  Pumunta naman siya sa kanang parte ng kanilang bahay ang sala, tahimik at wala parin si Meya. “Aaaaahhh! alam ko na,” tinungo ni Maya ang silid nila.  Dahan-dahan niyang binuksan ang pinto. “Meya, Meya nasaan ka?  Mahuhuli na kita.”  Malumanay na sambit ni Maya. Subalit wala paring Meya Manika. “Naku! Saan kaya siya nagtago? Lumabas ito ng bahay, tiningnan sa likod ng kanilang bahay, sa ilalim ng mga tanim, gayon paman wala paring Meya Manika. Nagsimula na siyang mag-alala. “Nasaan kana Meya? Lumabas kana?” Paiyak na sambit ni Maya.  “Inay, Itay, nawawala po si Meya.”  “Paanong nawala? Tanong ni Mang Mando. “Naglalaro lang po kami ng tagu-taguan, kanina ko pa po siya hinahanap, pero hindi ko siya mahanap.”  Malungkot na sagot ni Maya.“Huhuhuhu, tulungan po ninyo ako, hanapin natin si Meya.”  Iyak nang iyak si Maya.  Lumapit na rin ang kanyang Kuya Miguel at bunsong kapatid na si Marco. “Huwag ka nang umiyak Maya, tutulungan ka naming.”  Nagmungkahi ang kanyang Inay na si Aling Memay na mag imprinta sila ng larawan ni Meya Manika, ipaskil sa labas ng kanilang bahay at sa kanilang plasa. “Heto na ang larawan ni Meya, idikit natin sa labas.”   Ang wika ni Aling Memay. Subalit hindi pa rin nila nahanap si Memay. Iyak nang iyak parin si Maya. “Maya, Maya, gumising ka, nananaginip ka.”  “Bakit kaba umiiyak? “Inay nawawala po si Meya.”  Ang malungkot na sagot ni Maya. “Saan mo ba iniwan ang iyong manika?   Tanong naman ni Aling Nena. “Hindi ko po alam, Inay”. Dali-daling bumangon si Maya. Lumabas ng kanyang silid at hinanap si Maya. “Pumunta siya sa sala, sa kusina, sa banyo, at sa bahaging likuran ng kanilang bahay.  Subalit wala talaga si Meya Manika.  “Isipin mo nang mabuti Maya, saan ka huling naglaro ng iyong manika?   Tanong muli ni Aling Memay. “Kahapon po dito sa loob ng bahay.”  “Miguel, Marco, hali nga kayo,” tawag ni Aling Memay sa dalawang magkakapatid.  Lumapit kaagad sila at nagtanong.  “Bakit po Inay?  Nakita niyo ba ang manika ni Maya?  “Hindi po,” magalang na sagot ni Kuya Miguel.  “Ikaw Marco?  Tanong naman ni Aling Memay kay Marco. “Hindi rin po, Inay.”  “Isipin mo nang mabuti Maya, saan mo inilagay ang iyong manika? “Kahapon po pinapatulog ko po siya, ah, naaalala ko na po pinasok ko siya sa isang karton na ginawa kong kama niya.” “Dito ko po sa sahig nilapag kahapon.”  Maligsing tugon ni Maya sa kanyang Inay.  Dali-daling tumalikod si Aling Memay, at nagtungo sa bodega. “Ito ba na kahon Maya?   Tanong ni Maya.   “Upo , Inay yan nga po.”  Sabik na sagot ni Maya. “Heto, buksan mo,” wika ng kanyang inay. Nang buksan ni Maya, nagningning ang kanyang mga mata nang makita sa loob ng kahon si Meya Manika. “Maraming-maraming salamat po Inay.”  Niyakap nang mahigpit ni Maya si Meya Manika. Masayang-masaya din ang kanyang mga kapatid. Pinaalalahanan ni Aling Memay ang kanyang mga anak na, “Kapag kayo ay maglalaro, siguraduhing iligpit sa tamang lalagyan ang inyong mga laruan. Pahalagahan ninyo ang ano mang bagay na nasa inyo,” wika ng kanilang Inay.  “Opo Inay, patawad po, nakalimutan kong iligpit ang aking manika,” malumanay na sagot ni Maya.  “Opo Inay!” sagot din ni Kuya Miguel at Marco. Hinalikan at niyaakap muli Maya si Meya Manika. "
new_txt = run_normalize(txt) 
print(new_txt)

# Result:
# kinuha|VBOF ko|PRS ito|PRO sa|CCT refrigerator|FW $ ang | DTC food | FW na | CCP kinakain | VBTR ni | DTP Martha | NNP ay | LM masarap | JJD

word_embedder = WordEmbedding(new_txt)
print(word_embedder)

# Result
    # ang: {'traditional': CC: 3 SC: 1 IP: False, 'lexical': LX: other IF: False}
    # food: {'traditional': CC: 4 SC: 1 IP: False, 'lexical': LX: other IF: True}
    # ni: {'traditional': CC: 2 SC: 1 IP: False, 'lexical': LX: other IF: False}
    # ay: {'traditional': CC: 2 SC: 1 IP: False, 'lexical': LX: other IF: False}
    # kinakain: {'traditional': CC: 8 SC: 3 IP: True, 'lexical': LX: verb IF: False}
    # Martha: {'traditional': CC: 6 SC: 2 IP: True, 'lexical': LX: noun IF: False}
    # na: {'traditional': CC: 2 SC: 1 IP: False, 'lexical': LX: other IF: False}
    # ito: {'traditional': CC: 3 SC: 1 IP: False, 'lexical': LX: other IF: False}
    # kinuha: {'traditional': CC: 6 SC: 3 IP: True, 'lexical': LX: verb IF: False}
    # masarap: {'traditional': CC: 7 SC: 2 IP: True, 'lexical': LX: adj IF: False}
    # sa: {'traditional': CC: 2 SC: 1 IP: False, 'lexical': LX: other IF: False}
    # refrigerator: {'traditional': CC: 12 SC: 5 IP: True, 'lexical': LX: other IF: True}
    # ko: {'traditional': CC: 2 SC: 1 IP: False, 'lexical': LX: other IF: False}


def sentence_length(all_words):
    total_sentences = 0
    total_words = 0
    sentence = []

    for word in all_words:
        if word == "$":
            if sentence:
                total_sentences += 1
                total_words += len(sentence)
                sentence = []
        elif word != "#":
            sentence.append(word)

    if sentence:
        total_sentences += 1
        total_words += len(sentence)

    avg_sentence_length = total_words / total_sentences if total_sentences > 0 else 0
    return avg_sentence_length




all_words = new_txt.split()
sent_len = sentence_length(all_words)

print(all_words)

# Result:
# ['ang|DTC', 'food|FW', 'na|CCP', 'kinakain|VBTR', 'ni|DTP', 'Martha|NNP', 'ay|LM',
#     'masarap|JJD', '#', 'kinuha|VBOF', 'ko|PRS', 'ito|PRO', 'sa|CCT', 'refrigerator|FW', '$']

content = []
for word in all_words:
    match = re.match(r"(.+)\|(.+)", word)
    if match:
        parsed_word, _ = match.groups()  # Extract the word before "|"
        content.append(parsed_word)
    elif word in {"$", "#"}:  # Retain special markers if not tagged
        content.append(word)

print(content)

# Result:
# ['ang', 'food', 'na', 'kinakain', 'ni', 'Martha', 'ay', 'masarap', '#', 'kinuha', 'ko', 'ito', 'sa', 'refrigerator', '$']

entries = get_fragments(content, sent_len, word_embedder.keyvalues())
print(entries)
print()
models = [1, 2, 3, 4, 5, 6]
feature_set = ['sent_len', 'word_len', 'syll_num', 'poly_num','noun_tr', 'verb_tr', 'type_tr', 'lex_density', 'lex_foreign']

for model_v in models:
    with open(f'svm/svm_{model_v}_B_-1.pkl', 'rb') as f:
        model = pickle.load(f)

    X_new = pd.DataFrame([
        entries[0].get_features(),
        #  entries[1].get_features(), 
        #  entries[2].get_features(),
         
    ], columns=feature_set)

    print(X_new)

    predictions = model.predict(X_new)
    # print(inputs)
    probabilities = model.predict_proba(X_new)
    print(model.classes_)
    print(f"Grade {model_v} = Probabilities:\n {probabilities}")
