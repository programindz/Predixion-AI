from collections import defaultdict
import itertools
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize

CONVERSATION = """
RA: नमस्ते श्री कुमार, मैं एक्स वाई जेड फाइनेंस से बोल रहा हूं। आपके लोन के बारे में बात करनी थी।
B: हां, बोलिए। क्या बात है?
RA: सर, आपका पिछले महीने का EMI अभी तक नहीं आया है। क्या कोई समस्या है?
B: हां, थोड़ी दिक्कत है। मेरी नौकरी चली गई है और मैं नया काम ढूंढ रहा हूं।
RA: ओह, यह तो बुरा हुआ। लेकिन सर, आपको समझना होगा कि लोन का भुगतान समय पर करना बहुत जरूरी है।
B: मैं समझता हूं, लेकिन अभी मेरे पास पैसे नहीं हैं। क्या कुछ समय मिल सकता है?
RA: हम समझते हैं आपकी स्थिति। क्या आप अगले हफ्ते तक कुछ भुगतान कर सकते हैं?
B: मैं कोशिश करूंगा, लेकिन पूरा EMI नहीं दे पाऊंगा। क्या आधा भुगतान चलेगा?
RA: ठीक है, आधा भुगतान अगले हफ्ते तक कर दीजिए। बाकी का क्या प्लान है आपका?
B: मुझे उम्मीद है कि अगले महीने तक मुझे नया काम मिल जाएगा। तब मैं बाकी बकाया चुका दूंगा।
RA: ठीक है। तो हम ऐसा करते हैं - आप अगले हफ्ते तक आधा EMI जमा कर दीजिए, और अगले महीने के 15 तारीख तक बाकी का भुगतान कर दीजिए। क्या यह आपको स्वीकार है?
B: हां, यह ठीक रहेगा। मैं इस प्लान का पालन करने की पूरी कोशिश करूंगा।
RA: बहुत अच्छा। मैं आपको एक SMS भेज रहा हूं जिसमें भुगतान की डिटेल्स होंगी। कृपया इसका पालन करें और समय पर भुगतान करें।
B: ठीक है, धन्यवाद आपके समझने के लिए।
RA: आपका स्वागत है। अगर कोई और सवाल हो तो मुझे बताइएगा। अलविदा।
B: अलविदा।
"""


'''
A function to create a dictionary which stores keys as the speakers and values as the list of utterances
of the speaker in the conversation
'''
def create_conversation_dictionary(conversation):
    new_conversation = conversation.split('\n')
    speaker_words = defaultdict(list)

    for each_conv in new_conversation:
        if each_conv == "":
            continue
        speaker, utterance = each_conv.split(":")
        speaker_words[speaker].append(utterance)

    return speaker_words

SPEAKER_WORDS = create_conversation_dictionary(CONVERSATION)
# print(SPEAKER_WORDS,"\n")


# These stop words define a set of most common words in hindi language which can be ignored for analysis part
STOP_WORDS = set([
    "और", "का", "की", "को", "है", "यह", "हमें", "आप", "नहीं", "लेकिन", "क्योंकि", 
    "से", "में", "पर", "जब", "तो", "यहाँ", "वह", "हों", "सभी", "?","-" ,","
])


'''
A function to generate score for each utterance of both speakers in the conversation which 
represents presence of different/valuable words. Return value will be in sorted order i.e high score to low score
'''
def get_score(speaker_words):
    frequency_words = defaultdict(int)
    sentence_score = defaultdict(int)

    for speaker in speaker_words.values():
        for utterance in speaker:
            words = word_tokenize(utterance)
            for word in words:
                if word not in STOP_WORDS:
                    frequency_words[word] +=1
    # print(frequency_words, "\n")

    for speaker in speaker_words.values():
        for utterance in speaker:
            words = word_tokenize(utterance)
            for word in words:
                sentence_score[utterance] += frequency_words.get(word, 0)
    # print(sentence_score)
    sentence_score = dict(sorted(sentence_score.items(), key=lambda x:x[1], reverse=True))
    return sentence_score

sentence_scores = get_score(SPEAKER_WORDS)

# To get the k most scored utterances which consist of variety of words
K = 0
print("Top ", K, " Valuable Utterances: ")
for _ in range(K):
    print(next(iter(sentence_scores)))



####### SENTIMENT ANALYSIS  ############

POSITIVE_WORDS = set([
    "धन्यवाद", "अच्छा", "उम्मीद", "ठीक", "पालन", "स्वीकार"
])
NEGATIVE_WORDS = set([
    "समस्या", "दिक्कत", "बुरा", "नहीं", "पैसे नहीं", "नहीं आया", "चली"
])

'''
A function which determines the sentiments of an utterance either from
RA or Borrower based on presence of positive, negative and neutral words
'''
def get_sentiments(speaker_words):
    sentence_sentiment = dict()

    for speaker in speaker_words.values():
        for utterance in speaker:
            sentiment_weight = 0
            words = word_tokenize(utterance)

            for word in words:
                if word in POSITIVE_WORDS:
                    sentiment_weight += 1
                elif word in NEGATIVE_WORDS:
                    sentiment_weight -= 1
                else:
                    pass

            if sentiment_weight >= 1:
                sentence_sentiment[utterance] = 'positive'
            elif sentiment_weight <= -1:
                sentence_sentiment[utterance] = 'negative'
            else:
                sentence_sentiment[utterance] = 'neutral'

    return sentence_sentiment

sentiments = get_sentiments(SPEAKER_WORDS)
print("Receiving Agent Sentiment Change: ", list(sentiments.values())[:8],
    "\nBorrower Sentiment Change: ", list(sentiments.values())[8:])