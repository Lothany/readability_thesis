from preprocess import test
import re

if __name__ == "__main__":
    # test()
    word = "Hello"
    consonant_vowel = r'([b-df-hjklmnp-rtvwz][aeiou])'
    diphthongs = r'(ai|ei|oi|ui|au|ou)'
    consonant_clusters = r'(ng|br|tr|st)'
    pattern = re.compile(f'({consonant_clusters}|{diphthongs}|{consonant_vowel})')
    matches = pattern.findall(word.lower())
    print(len(matches))
