import os
import pickle
import re


def load_wordset(pkl_path):
    if os.path.exists(pkl_path):
        with open(pkl_path, 'rb') as f:
            return pickle.load(f)
    return set()

def save_wordset(wordset, pkl_path):
    with open(pkl_path, 'wb') as f:
        pickle.dump(wordset, f)


def is_likely_foreign(word):
    rare_letters = set("qvxzc")
    return any(char in rare_letters for char in word.lower())

def looks_filipino(word):
    prefixes = ('mag', 'nag', 'ma', 'ka', 'pag', 'napaka', 'pina', 'ipinag', 'kumaka', 'ikina')
    suffixes = ('an', 'in', 'ng', 'han')
    return word.startswith(prefixes) or word.endswith(suffixes)

def has_reduplication(word):
    if '-' in word:
        parts = word.split('-')
        return len(parts) == 2 and parts[0] == parts[1]
    return False

def heuristic_score(word):
    score = 0
    if is_likely_foreign(word):
        score += 2
    if looks_filipino(word):
        score -= 1
    if has_reduplication(word):
        score -= 1
    return score

def classify_word(word, known_filipino, known_foreign, session_filipino, session_foreign, identify_only = False):
    word = word.lower()
    if word in known_filipino or word in session_filipino:
        return False
    if word in known_foreign or word in session_foreign:
        return True

    score = heuristic_score(word)
    if score >= 2:
        print(f"🤖 '{word}' classified as Foreign (score {score})")
        session_foreign.add(word)
        return True
    elif score <= -1:
        print(f"🤖 '{word}' classified as Filipino (score {score})")
        session_filipino.add(word)
        return False
    else:
        if not identify_only:
            print(f"❓ Heuristic uncertain for word: '{word}' (score {score})")
            user_input = input(
                "Press Enter if Filipino, type 'y' if Foreign: ").strip().lower()
            if user_input == 'y':
                session_foreign.add(word)
                return True
            else:
                session_filipino.add(word)
                return False
        else:
            print(f"--- Unable to classify if the {word} is foreign")
            return False

def process_file(file_path, known_filipino, known_foreign, session_filipino, session_foreign):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read().lower()
        words = re.findall(
            r'\b[a-zA-ZñÑ]+(?:-[a-zA-ZñÑ]+)?\b', text)  # Tokenize
        seen = set()
        for word in words:
            if word not in seen:
                seen.add(word)
                classify_word(word, known_filipino, known_foreign,
                              session_filipino, session_foreign)

def reassign_word(word, filipino_words, foreign_words):
    if word in filipino_words:
        filipino_words.remove(word)
        foreign_words.add(word)
        print(f"✅ '{word}' moved from Filipino to Foreign.")
    elif word in foreign_words:
        foreign_words.remove(word)
        filipino_words.add(word)
        print(f"✅ '{word}' moved from Foreign to Filipino.")
    else:
        print(f"⚠️ '{word}' not found in either Filipino or Foreign sets.")

def identify(word:str):
    filipino_words = load_wordset('filipino_words.pkl')
    foreign_words = load_wordset('foreign_words.pkl')

    session_filipino = set()
    session_foreign = set()
    print(f"-- Identifying the word {word} if it's foreign")
    return classify_word(word, filipino_words, foreign_words, session_filipino, session_foreign, True)

def mass_identify(filepath:str):
    filipino_words = load_wordset('filipino_words.pkl')
    foreign_words = load_wordset('foreign_words.pkl')

    session_filipino = set()
    session_foreign = set()

    input_path = filepath
    if not os.path.exists(input_path):
        print(f"⚠️ File '{input_path}' not found.")
        exit()

    print(f"📄 Processing file: {input_path}")
    process_file(input_path, filipino_words, foreign_words, session_filipino, session_foreign)

    print("\n✅ Finished scanning file.")
    print("Type 'save' to persist your labels or 'exit' to quit.\n")

    while True:
        cmd = input("Command: ").strip().lower()
        if cmd == 'save':
            if session_filipino:
                filipino_words.update(session_filipino)
                save_wordset(filipino_words, 'filipino_words.pkl')
                session_filipino.clear()
                print("✅ Filipino words saved.")
            if session_foreign:
                foreign_words.update(session_foreign)
                save_wordset(foreign_words, 'foreign_words.pkl')
                session_foreign.clear()
                print("✅ Foreign words saved.")
            if not session_filipino and not session_foreign:
                print("Nothing new to save.")
        elif cmd == 'edit':
            word = input("Enter a word to re-assign: ").strip().lower()

            if word == 'cancel':
                print("Cancelling...")
                break

            reassign_word(word, filipino_words, foreign_words)
        elif cmd == 'exit':
            if session_filipino or session_foreign:
                print(
                    "⚠️ You have unsaved changes. Type 'save' before exiting if you want to keep them.")
            break
        else:
            print("Unknown command. Type 'save' or 'exit'.")
