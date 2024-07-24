import pandas as pd
import numpy as np
from hazm import *
import time
import math

start_time = time.time()

punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~؛»،«'''

normalizer = Normalizer()
word_tokenizer = WordTokenizer()
token_splitter = TokenSplitter()

book_train = pd.read_csv("books_train.csv")
book_test = pd.read_csv("books_test.csv")
stop_word = pd.read_csv("sw.csv")

sw = []
for i in range(len(stop_word)):
    sw.append(stop_word.loc[i][0])

society = {}  
job = {}  
novel = {}  
islam = {}  
child_stroy = {}  
short_story = {}  


def recognize_category(category: str) -> dict:
    if category == "جامعه‌شناسی":
        return society
    elif category == "کلیات اسلام":
        return islam
    elif category == "داستان کودک و نوجوانان":
        return child_stroy
    elif category == "مدیریت و کسب و کار":
        return job
    elif category == "داستان کوتاه":
        return short_story
    elif category == "رمان":
        return novel
    return None


def calculate_probebility_words(bow: dict) -> float:
    sum = 0
    uniqe_word = len(bow)
    for i in bow:
        sum = sum + bow.get(i)
    for j in bow:
        bow[j] = math.log((bow.get(j) + 100) / (sum + 100 * uniqe_word), 0.5)
    return math.log((100 / sum + 100 * uniqe_word), 0.5)


def analyze_probabilities(result_category: int, real_category: str) -> bool:
    if ((result_category == 0 and real_category == "جامعه‌شناسی") or
        (result_category == 1 and real_category == "مدیریت و کسب و کار") or
        (result_category == 2 and real_category == "رمان") or
        (result_category == 3 and real_category == "کلیات اسلام") or
        (result_category == 4 and real_category == "داستان کودک و نوجوانان") or
            (result_category == 5 and real_category == "داستان کوتاه")):
        return True
    return False


for row in range(len(book_train)):
    description = normalizer.normalize(book_train.loc[row][1])
    name = normalizer.normalize(book_train.loc[row][0])
    l1 = word_tokenizer.tokenize(description)
    l1.extend(word_tokenizer.tokenize(name))
    dictionary = recognize_category(book_train.loc[row][2])
    if dictionary == None:
        print("error\n")
    for i in l1:
        if i not in punctuations and "۱" >= i and i <= "۹" and i not in sw:
            if dictionary.get(i) == None:
                dictionary.update({i: 1})
            else:
                dictionary[i] = dictionary[i] + 1

for row in range(len(book_test)):
    description = normalizer.normalize(book_test.loc[row][1])
    name = normalizer.normalize(book_test.loc[row][0])
    l1 = word_tokenizer.tokenize(description)
    l1.extend(word_tokenizer.tokenize(name))
    l2 = []
    for i in l1:
        if i not in punctuations and "۱" >= i and i <= "۹" and i not in sw:
            l2.append(i)
    book_test.loc[row][1] = l2

not_existing_word = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
not_existing_word[0] = calculate_probebility_words(society)
not_existing_word[1] = calculate_probebility_words(job)
not_existing_word[2] = calculate_probebility_words(novel)
not_existing_word[3] = calculate_probebility_words(islam)
not_existing_word[4] = calculate_probebility_words(child_stroy)
not_existing_word[5] = calculate_probebility_words(short_story)

true_recognize = 0
for i in range(len(book_test)):
    p = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    for j in book_test.loc[i][1]:
        if j in society:
            p[0] += society.get(j)
        else:
            p[0] += not_existing_word[0]

        if j in job:
            p[1] += job.get(j)
        else:
            p[1] += not_existing_word[1]

        if j in novel:
            p[2] += novel.get(j)
        else:
            p[2] += not_existing_word[2]

        if j in islam:
            p[3] += islam.get(j)
        else:
            p[3] += not_existing_word[3]

        if j in child_stroy:
            p[4] += child_stroy.get(j)
        else:
            p[4] += not_existing_word[4]

        if j in short_story:
            p[5] += short_story.get(j)
        else:
            p[5] += not_existing_word[5]

    if analyze_probabilities(p.argmax(), book_test.loc[i][2]):
        true_recognize += 1

print(true_recognize, " books were recognized true\naccuracy: ",
      true_recognize*100/450, "%")
print("Time: ", "%s" % (time.time() - start_time), " sec")
