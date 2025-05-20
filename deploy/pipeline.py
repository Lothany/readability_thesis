import json
import re
import pandas as pd
import numpy as np
from Features import get_fragments
from WordEmbedding import WordEmbedding
from normalizer import run_normalize
import re 
import pickle
from Features import split_into_fragments
from Fragment import Fragment

def fragments_to_json(fragments, target):
    return [
        {
            "content": {
                "original": fragment.show(),
                "grade_scores": fragment.to_dict()
            },
            "target": {
                "level": target,
                "grade_score": fragment.to_dict()[target]
            }
        }
        for fragment in fragments
    ]

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

def run_pipeline(text, target):
    # txt = 'May isang tribong nagngagalang “Bolig”. Ang tribong ito noon paman ay palipat-lipat na ng lugar at kung saan na nakakarating hanggang sa napadpad sila sa lugar na sobrang init at mala-desyerto, ang bayan ng Uga. Walang masyadong tumatagal sa lugar na ito dahil nga sa sobrang lakas ng sikat ng araw at hindi dinadaanan ng ulan. Wala rin masyadong pinagkukunan ng tubig. Ngunit ang Tribong  Bolig ay napagod na sa kakalakad at paghahanap ng matitirahan kaya pinili na lang muna nilang manatili sa lugar na ito. Ang tribong ito ay may kasamang mga bata at isa na dito si ay si “Imbok”. Si Imbok ay isang mabait at masayahing   bata kaya naman ay sya gustong-gusto ng kanyang mga katribo. Maliban dito siya rin ay isang mataba ang mapagbiro kaya sya ay nagbibigay ng ngiti sa kanyang mga kasama. “Magandang umaga ginang Mirasol?” bati ni Imbok. “Oyy, magandang umaga Imbok?” Tugon ni Aling Mirasol “Mas lalo po kayong gumaganda ngayon.” Palabirong sabi ni Imbok. “Di naman. Ikaw talaga Imbok napaka palabiro mo talaga.” Kinikilig na tugon ni Aling Mirasol. Sunod ay dumaan naman si Imbok kay Mang Bergo. “Magandang umaga, Mang Bergo!” “Magandang umaga din sa iyo, Imbok.” “Ang bango naman po niyang niluluto nyo, sigurado ako masarap yan.” Ang sabi ni Imbok. “Hmmnn.  Hindi mo pa nga natitikman! Pero dahil mabait ka at pinapatawa mo ako, sige bibigyan kita.” Sabi ni Mang Bergo. “Wow! Maraming Salamat po!” Masayang sabi ni Imbok. Unang pananatili ng mga Tribong Bolig sa Uga ay maayos pa dahil meron silang imbak na tubig kung kayat naman hindi sila gaanong nahihirapan. Ngunit di nagtagal ay unti-unti nang nauubos ang kanilang tubig at maliban pa dito mas sobrang umiinit ang lugar kaya naman ay nagsisimula na nilang maranasan ang hirap. “Naku! Wala nang tubig! Pano na itong nilututo ko” “Ayyy, Ano ba yan! Wla nang tubig na natira. Di pa ako tapos maligo.” “Uhaw-uhaw na ako ngunit walang tubig na maiinom.” Nagrereklamo na ang mga tao dahil sa kakulangan ng tubig at nagsisimula na silang manghina  dahil hindi sila nakakainom ng tubig. Dahil dito ay nalungkot si Imbok sa kung ano ang kanilang nararanasan at kanyang nakikita. Gustong-gusto niya tulungan ang mga katribo ngunit wla syang magawa.  Naghanap ng paraan si Imbok. Pumunta sya kung saan-saang parte ng Uga para maghanap ng pinanggagalingan ng tubig. Sa tagal nyang paghahanap ay napagod sya at tuyong-tuyong. Humanap siya nang masisilungan. Meron siyang nakitang punong kahoy at doon muna sumilong. Nang nainitan ito at gustong magpahangin ay umakyat ito sa tuktog ng puno. Nagpahinga ito at tiningnan ang lawak ng lugar. Sa kanyang pagtitingin ay may naaninag siyang parang kumikinang sa malayo. Bumaba si Imbok at nagmadaling tumakbo papunta sa kung saan niya nakita iyong kumikinang-kinang.  Nang narating na ito ni Imbok ay nabighani sya sa kanyang mga nakita. Nakakita siya ng kung anu-anong mga bagay. Mga bagay na sa loob nga bahay lamang nakikita. Plato, kutsara, salamin at kung anu-ano pa na kumikinang. Ang tanging hindi lamang kumikinang dito ay isang bagay at yun ay ang Tabo.  Nilapitan ito ni Imbok. Nang ito’y kanyang hinawakan ay biglang lumabas ang tubig didto hanggang sa mapuno. Dahil uhaw-uhaw na si Imbok  kumuha sa ng tubig dito at uninom. Ang nakabibigla ay muling na namang umagas ang tubig dito at napuno ang tabo.  Tuwang-tuwa siya sa kanyang nakita at nahanap. Naisip niya na ito ang solusyon sa kanilang problema na  kawalan ng tubig. Nagmadali siyang umuwi upang ibalita ito sa kanyang mga katribo. Tumakbo siyang tuwang-tuwa habang bitbit ang tabo na hindi nauubusan ng tubig. Malapit nang makarating si Imbok sa kaniyang mga katribo. Malayo pa lang siya ay natanaw na sya ng kanyang mga kasama. “Si imbok ba yun?” “Parang si Imbok nga. Ba’t parang nagmamadali ata sya? Nagtatakang tanong ng kanyang mga kasama. “Mga kasama, mga kasama” sigaw ni Imbok habang kumakaway. Halos lahat na ng mga katribo ang nakapansin sa kanya at inaantay nga nila ang pagdating ni Imbok. Nang si Imbok ay nakarating na, masaya at agad-agad niyang ibinahagi sa mga kasama ang kanyang nahanap. “Mga kasama meron akong magandang balita sa inyo. Malulutas na ang ating suliranin.” Masayang sinabi ni Imbok. Masaya ang mga kasama niya sa narinig pero nagtataka kung papano malulutas ni Imbok kanilang problema. “Magandang balita iyan Imbok. Pero papano?” sabi ng isa niyang kasama. “Baka naman binibiro mo na naman kami?” kasunod na sinabi na isa pa niyang kasama. “Itong tabo!” Itinaas ni Imbok ang tabo at ipinakita sa mga katribo.” Tumawa ang kanyang mga katribo at hindi sya pinaniwalaan. “HAHAHA.HAHAHAH.HAHAAH” Tawang-tawa ang kanyang mga kasama. “Ito talagang si Imbok palaging nagbibiro. Sabi ng kanyang mga kasama. “Hindi po ako nagbibiro.” Ipinakita ni Imbok kung paano ang pagpapadaloy ng tubig sa tabo. Kumuha sya ng isang timba at ibinuhos ang tubig na laman ng tabo.  Nagpatuloy ang pagdaloy ang tubig mula sa tabo at hindi tumitigil.  Nabigla ang mga kasama ni Imbok sa Nakita at napagtanto nilang totoo nga ang sinsabi ni Imbok. “Hala! Totoo nga.” Namamanhang sabi ng kanyang mga kasama. “Sa wakas ay may solusyon narin tayo sa ating problema.” “Salamat Imbok. Dahil dito ay may tubig na tayo at di na tayo mahihirapan pa.” Tuwang-tuwa ang Tribong Bolig dahil sa Tabong nahanap ni Imbok. Hindi na sila mahihirapan pa.  Nasolusyan nga ang problema ng Tribong Uga sa kawalan ng tubig. Nakakapagligo na sila araw-araw. Nakapaglalaba ng damit. Nakakainom nga maraming tubig. Si imbok naman ay nasiyahan sa kanyang ginawa. Hindi na nya nakitang nahihirapan pa ang ang kanyang mga katribo. Nagpatuloy ang kabutihan ng kanyang puso. Hindi niya ipinagdadamot kung sino man ang nangangailangan nito. Buong puso syang nagbibigay.  Ang tribong Bolig ay naghanap ng permanenting kinalalagyan ng tabo kaya naman ay gumawa sila ng isang malaking balon sa gitna ng kanilang tinitirahan nang sa ganun ay madali na lamang para sa kanila ang pagkuha ng tubig. Kung meron mang mga dayo na dumadaan sa lugar na ito ay, nagbabahagi sila ng tubig ng walang bayad.  Simula noon ay hindi na umalis pa ang Tribong bolig sa lugar na ito. Ang mala-desyertong lugar ng Uga ay isa na sa lugar na pinanangalinagn ng tubig. Ito ay napalilibutan na ng mga anyong tubig at mula dito ay umaagos ang tubig papunta sa ibang biyan. Naging kilala narin ang Bayang Uga na mayaman sa tubig.'
    txt = text
    normalized_txt = run_normalize(txt)
    # print(normalized_txt) #----

    # Result:
    # kinuha|VBOF ko|PRS ito|PRO sa|CCT refrigerator|FW $ ang | DTC food | FW na | CCP kinakain | VBTR ni | DTP Martha | NNP ay | LM masarap | JJD

    word_embedder = WordEmbedding(normalized_txt)
    # print(word_embedder) #----

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

    all_words = normalized_txt.split()
    sent_len = sentence_length(all_words)

    # print(all_words) #----

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

    # print(content) #----

    # Result:
    # ['ang', 'food', 'na', 'kinakain', 'ni', 'Martha', 'ay', 'masarap', '#', 'kinuha', 'ko', 'ito', 'sa', 'refrigerator', '$']

    basis = split_into_fragments(txt)
    entries = get_fragments(content, sent_len, word_embedder.keyvalues())
    models = [1, 2, 3, 4, 5, 6]
    feature_set = ['sent_len', 'word_len', 'syll_num', 'poly_num','noun_tr', 'verb_tr', 'type_tr', 'lex_density', 'lex_foreign']

    output_fragments = []
    for index, entry in enumerate(entries):
        output_fragment = Fragment(basis[index], index)
        for model_v in models:
            with open(f'svm/svm_{model_v}_B_-1.pkl', 'rb') as f:
                model = pickle.load(f)

            input_feature = pd.DataFrame([entry.get_features(),], columns=feature_set)
            probabilities = model.predict_proba(input_feature)
            output_fragment.set_score(model_v, float(probabilities[0][1])*100)
        output_fragments.append(output_fragment)

    json_output = fragments_to_json(output_fragments, target)

    return json_output
