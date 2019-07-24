from pandas import DataFrame, read_csv
import pandas as pd
from nltk.stem.wordnet import WordNetLemmatizer
from pattern3.text.en import singularize
import matplotlib.pyplot as plt
import math
import re

file = './data/train.csv'
Data = pd.read_csv(file)

emails = Data['text']
statuses = Data['type']
emailsNumber = len(emails)
#---------------------------------------------
def hasLink(email) :
    links = []
    links += re.findall('www\.(?:[-\w.]|(?:%[\da-fA-F]))+', email)
    links += re.findall('https?://(?:[-\w.]|(?:%[\da-fA-F]))+', email)
    links += re.findall('[-\w.]?\.com', email)
    return 1 if (len(links) != 0) else 0

emailHasLinks = [hasLink(email) for email in emails]
#---------------------------------------------
def hasLongNumber(email) :
    return 1 if len(re.findall('[0-9\-]{3}', email)) != 0 else 0

emailHasLongNumber = [hasLongNumber(email) for email in emails]
#---------------------------------------------
trashCharsWeights = [0 for i in range(emailsNumber)]

for index in range(emailsNumber) :
    email = emails[index].lower()
    formedEmail = ""
    for character in email :
        if character >= 'a' and character <= 'z' :
            formedEmail += character
        else :
            formedEmail += ' '
            trashCharsWeights[index] += 1
    trashCharsWeights[index] /= len(emails[index].split()) if trashCharsWeights[index] != 0 else 1
    emails[index] = formedEmail
#---------------------------------------------
hamEmails = []
spamEmails = []
hamTrashCharsWeights = []
spamTrashCharsWeights = []
hamHasLinks = []
spamHasLinks = []
hamHasLongNumber = []
spamHasLongNumber = []

for index in range(emailsNumber) :
    if statuses[index] == 'spam' :
        spamEmails.append(emails[index])
        spamTrashCharsWeights.append(trashCharsWeights[index])
        spamHasLinks.append(emailHasLinks[index])
        spamHasLongNumber.append(emailHasLongNumber[index])
    else :
        hamEmails.append(emails[index])
        hamTrashCharsWeights.append(trashCharsWeights[index])
        hamHasLinks.append(emailHasLinks[index])
        hamHasLongNumber.append(emailHasLongNumber[index])

hamsNumber = len(hamEmails)
spamsNumber = len(spamEmails)
#---------------------------------------------
# plt.hist([hamHasLinks,spamHasLinks], 2, color = ['g','r'], label = ['ham emails','spam emails'])
# plt.title('Link feature')
# plt.xlabel('feature (has link)')
# plt.ylabel('number of emails with feature')
# plt.legend()
# plt.show()

# plt.hist([hamHasLongNumber,spamHasLongNumber], 2, color = ['g','r'], label = ['ham emails','spam emails'])
# plt.title('LongNumber feature')
# plt.xlabel('feature (has link)')
# plt.ylabel('number of emails with feature')
# plt.legend()
# plt.show()

# plt.hist([hamTrashCharsWeights,spamTrashCharsWeights], 50, color = ['g','r'], label = ['ham emails','spam emails'])
# plt.title('Trash characters feature')
# plt.xlabel('weight of trash characters')
# plt.ylabel('number of emails with same trash weight')
# plt.legend()
# plt.show()
#---------------------------------------------
def linkBayesian(feature) :
    P_spam = spamsNumber/(spamsNumber+hamsNumber)
    numberOfHamsWithLiks = hamHasLinks.count(1)
    numberOfSpamsWithLiks = spamHasLinks.count(1)
    spamEmailsWithFeature = numberOfSpamsWithLiks if feature else spamsNumber - numberOfSpamsWithLiks
    hamEmailsWithFeature = numberOfHamsWithLiks if feature else hamsNumber - numberOfHamsWithLiks
    P_feature_if_spam = spamEmailsWithFeature / spamsNumber
    P_feature_if_ham = hamEmailsWithFeature / hamsNumber
    P_feature = P_feature_if_spam + P_feature_if_ham
    return 100*(P_feature_if_spam*P_spam/P_feature) if P_feature != 0 else 0
#---------------------------------------------
hamLinkBayesianPs = [linkBayesian(feature) for feature in hamHasLinks]
spamLinkBayesianPs = [linkBayesian(feature) for feature in spamHasLinks]

# plt.hist([hamLinkBayesianPs,spamLinkBayesianPs], 10, color = ['g','r'], label = ['ham emails','spam emails'])
# plt.title('Link feature')
# plt.xlabel('spam chance %')
# plt.ylabel('number of emails with same chance')
# plt.legend()
# plt.show()
#---------------------------------------------
def longNumberBayseian(feature) :
    P_spam = spamsNumber/(spamsNumber+hamsNumber)
    numberOfHamsWithLongNumber = hamHasLongNumber.count(1)
    numberOfSpamsWithLongNumber = spamHasLongNumber.count(1)
    spamEmailsWithFeature = numberOfSpamsWithLongNumber if feature else spamsNumber - numberOfSpamsWithLongNumber
    hamEmailsWithFeature = numberOfHamsWithLongNumber if feature else hamsNumber - numberOfHamsWithLongNumber
    P_feature_if_spam = spamEmailsWithFeature / spamsNumber
    P_feature_if_ham = hamEmailsWithFeature / hamsNumber
    P_feature = P_feature_if_spam + P_feature_if_ham
    return 100*(P_feature_if_spam*P_spam/P_feature) if P_feature != 0 else 0
#---------------------------------------------
hamLongNumberBayesianPs = [longNumberBayseian(feature) for feature in hamHasLongNumber]
spamLongNumberBayesianPs = [longNumberBayseian(feature) for feature in spamHasLongNumber]

# plt.hist([hamLongNumberBayesianPs,spamLongNumberBayesianPs], 10, color = ['g','r'], label = ['ham emails','spam emails'])
# plt.title('LongNumber feature')
# plt.xlabel('spam chance %')
# plt.ylabel('number of emails with same chance')
# plt.legend()
# plt.show()
#---------------------------------------------
def trashCharBayesian(feature) :
    P_spam = spamsNumber/(spamsNumber+hamsNumber)
    
    spamEmailsWithFeature = 0
    for spamTrashCharsWeight in spamTrashCharsWeights :
        if spamTrashCharsWeight >= feature:
            spamEmailsWithFeature += 1

    hamEmailsWithFeature = 0
    for hamTrashCharsWeight in hamTrashCharsWeights :
        if hamTrashCharsWeight >= feature:
            hamEmailsWithFeature +=  1
            
    P_feature_if_spam = spamEmailsWithFeature / spamsNumber
    P_feature_if_ham = hamEmailsWithFeature / hamsNumber
    P_feature = P_feature_if_spam + P_feature_if_ham
    return 100*(P_feature_if_spam*P_spam/P_feature) if P_feature != 0 else 0
#---------------------------------------------
hamTrashWeightsBayesianPs = [trashCharBayesian(feature) for feature in hamTrashCharsWeights]
spamTrashWeightsBayesianPs = [trashCharBayesian(feature) for feature in spamTrashCharsWeights]

# plt.hist([hamTrashWeightsBayesianPs,spamTrashWeightsBayesianPs], 50, color = ['g','r'], label = ['ham emails','spam emails'])
# plt.title('TrashWeight feature')
# plt.xlabel('spam chance %')
# plt.ylabel('number of emails with same chance')
# plt.legend()
# plt.show()
#---------------------------------------------
hamLengths = [len(email) for email in hamEmails]
spamLengths = [len(email) for email in spamEmails]

# plt.hist([hamLengths, spamLengths], 20, color = ['g','r'], label = ['ham emails','spam emails'])
# plt.title('Length feature')
# plt.xlabel('length')
# plt.ylabel('emails with same length')
# plt.legend()
# plt.show()
#---------------------------------------------
def lengthBayesian(feature) :
    P_spam = spamsNumber/(spamsNumber+hamsNumber)
    
    spamEmailsWithFeature = 0
    for length in spamLengths :
        if abs(feature - length) <= 1 : 
            spamEmailsWithFeature += 1

    hamEmailsWithFeature = 0
    for length in hamLengths :
        if abs(feature - length) <= 1 : 
            hamEmailsWithFeature += 1
    
    P_feature_if_spam = spamEmailsWithFeature / spamsNumber
    P_feature_if_ham = hamEmailsWithFeature / hamsNumber
    P_feature = P_feature_if_spam + P_feature_if_ham
    return 100*(P_feature_if_spam*P_spam/P_feature) if P_feature != 0 else 0
#---------------------------------------------
hamLengthBayesianPs = [lengthBayesian(feature) for feature in hamLengths]
spamLengthBayesianPs = [lengthBayesian(feature) for feature in spamLengths]

# plt.hist([hamLengthBayesianPs,spamLengthBayesianPs], 50, color = ['g','r'], label = ['ham emails','spam emails'])
# plt.title('Length feature')
# plt.xlabel('spam chance %')
# plt.ylabel('number of emails with same chance')
# plt.legend()
# plt.show()
#---------------------------------------------
wordsValue = dict()
# wordsRepeat = dict()

def updateWordsValue(emails, status) :
    for email in emails :
        for word in email.split() :
            if len(word) < 2 : continue
            word = WordNetLemmatizer().lemmatize(word,'v')
            word = singularize(word)
            if word not in wordsValue :
                wordsValue[word] = 0
                # wordsRepeat[word] = 0
            wordsValue[word] += 1/hamsNumber if status == 'ham' else -1/spamsNumber
            # wordsRepeat[word] += 1

updateWordsValue(hamEmails, 'ham')
updateWordsValue(spamEmails, 'spam')
#---------------------------------------------
# commonWords = set()
# commonWordsEdge = emailsNumber/10

# for word,value in wordsRepeat.items() :
#     if value > commonWordsEdge :
#         commonWords.add(word)
# print(commonWords)

# for word in commonWords :
#     wordsValue.pop(word)
# ---------------------------------------------
def setEmailsValue(emails) :
    values = []
    for email in emails :
        value = 0
        wordsProcessNumber = 0
        for word in email.split() :
            word = WordNetLemmatizer().lemmatize(word,'v')
            word = singularize(word)
            if word not in wordsValue : continue
            value += wordsValue[word]
            wordsProcessNumber += 1
        value /= wordsProcessNumber if wordsProcessNumber != 0 else 1
        values.append(value)
    return values

hamValues = setEmailsValue(hamEmails)
spamValues = setEmailsValue(spamEmails)

# plt.hist([hamValues,spamValues], 80, color = ['g','r'], label = ['ham emails','spam emails'])
# plt.title('Value feature')
# plt.xlabel('emails value')
# plt.ylabel('number of emails with same value')
# plt.legend()
# plt.show()
#---------------------------------------------
def valueBayesian(feature) :
    P_spam = spamsNumber/(spamsNumber+hamsNumber)
    
    spamEmailsWithFeature = 0
    for value in spamValues :
        if value <= feature:
            spamEmailsWithFeature += 1

    hamEmailsWithFeature = 0
    for value in hamValues :
        if value <= feature:
            hamEmailsWithFeature += 1
            
    P_feature_if_spam = spamEmailsWithFeature / spamsNumber
    P_feature_if_ham = hamEmailsWithFeature / hamsNumber
    P_feature = P_feature_if_spam + P_feature_if_ham
    return 100*(P_feature_if_spam*P_spam/P_feature) if P_feature != 0 else 0
#---------------------------------------------
hamValueBayesianPs = [valueBayesian(feature) for feature in hamValues]
spamValueBayesianPs = [valueBayesian(feature) for feature in spamValues]

# plt.hist([hamValueBayesianPs,spamValueBayesianPs], 50, color = ['g','r'], label = ['ham emails','spam emails'])
# plt.title('Value feature')
# plt.xlabel('spam chance %')
# plt.ylabel('number of emails with same chance')
# plt.legend()
# plt.show()
#---------------------------------------------
# hamBayesiansPs = [hamWordsBagBayesianPs,hamValueBayesianPs,hamLengthBayesianPs,hamTrashWeightsBayesianPs,hamLinkBayesianPs,hamLongNumberBayesianPs]
# spamBayesiansPs = [spamWordsBagBayesianPs,spamValueBayesianPs,spamLengthBayesianPs,spamTrashWeightsBayesianPs,spamLinkBayesianPs,spamLongNumberBayesianPs]

# hamLabels = ['ham words bag','ham values','ham length','ham trash charackter','ham link','ham long number']
# spamLabels = ['spam words bag','spam values','spam length','spam trash charackter','spam link','spam long number']

# plt.hist(hamBayesiansPs, 25, alpha=0.5, label = hamLabels)
# plt.hist(spamBayesiansPs, 25, alpha=0.5, label = spamLabels)
# plt.title('emails bayesians possibilitys')
# plt.xlabel('possibility')
# plt.ylabel('number of emails')
# plt.legend()
# plt.show()
#---------------------------------------------
# def avgOfFiveFeature(a, b, c, d, e, n) :
#     return [(a[i]+b[i]+c[i]+d[i]+e[i])/5  for i in range(n)]

# hamsAvgOfFiveFeature = avgOfFiveFeature(hamValueBayesianPs, hamLengthBayesianPs, hamTrashWeightsBayesianPs, hamLinkBayesianPs, hamLongNumberBayesianPs, hamsNumber)
# spamsAvgOfFiveFeature = avgOfFiveFeature(spamValueBayesianPs, spamLengthBayesianPs, spamTrashWeightsBayesianPs, spamLinkBayesianPs, spamLongNumberBayesianPs, spamsNumber)

# plt.hist(hamsAvgOfFiveFeature, 100, color = 'g', label = 'hams')
# plt.hist(spamsAvgOfFiveFeature, 100, color = 'r', label = 'spams')
# plt.title('emails spam possibilitys')
# plt.xlabel('possibility')
# plt.ylabel('number of emails')
# plt.legend()
# plt.show()
#---------------------------------------------
wordsRepeatInSpams = dict()
wordsRepeatInHams = dict()
wordsNumberOfHams = 0
wordsNumberOfSpams = 0

def updateWordsRepeat(emails, wordsRepeat, status) :
    global wordsNumberOfHams, wordsNumberOfSpams
    for email in emails :
        for word in email.split() :
            if len(word) < 2 : continue
            word = WordNetLemmatizer().lemmatize(word,'v')
            word = singularize(word)
            if word not in wordsRepeat :
                wordsRepeat[word] = 0
            wordsRepeat[word] += 1
            if status == 'ham' : wordsNumberOfHams += 1
            else : wordsNumberOfSpams += 1
            
updateWordsRepeat(hamEmails, wordsRepeatInHams, 'ham')
updateWordsRepeat(spamEmails, wordsRepeatInSpams, 'spam')
#---------------------------------------------
# def deleteCommonWords(wordsRepeat, totalWordsNumber) :
#     readyToRemove = set()
#     for word in wordsRepeat :
#         if wordsRepeat[word] >= totalWordsNumber/100 :
#             readyToRemove.add(word)
#     for word in readyToRemove :
#         print(word, wordsRepeat[word])
#         wordsRepeat.pop(word)

# deleteCommonWords(wordsRepeatInHams, wordsNumberOfHams)
# deleteCommonWords(wordsRepeatInSpams, wordsNumberOfSpams)
#---------------------------------------------
def wordsBagBayesian(word) :
    P_spam = spamsNumber/(spamsNumber+hamsNumber)
    P_feature_if_spam = wordsRepeatInSpams[word] / wordsNumberOfSpams if word in wordsRepeatInSpams else 0
    P_feature_if_ham = wordsRepeatInHams[word] / wordsNumberOfHams if word in wordsRepeatInHams else 0
    P_feature = P_feature_if_spam + P_feature_if_ham
    return 100*(P_feature_if_spam * P_spam) / P_feature if P_feature != 0 else 0
#---------------------------------------------
def emailWordsBagBayesian(words) :
    emailWordsBagBayes = 0
    wordsCount = 0
    for word in words :
        if len(word) < 2 : continue
        word = WordNetLemmatizer().lemmatize(word,'v')
        word = singularize(word)
        emailWordsBagBayes += wordsBagBayesian(word)
        wordsCount += 1
    return emailWordsBagBayes / wordsCount if wordsCount != 0 else 0
#---------------------------------------------
hamWordsBagBayesianPs = [emailWordsBagBayesian(email.split()) for email in hamEmails]
spamWordsBagBayesianPs = [emailWordsBagBayesian(email.split()) for email in spamEmails]

# plt.hist([hamWordsBagBayesianPs,spamWordsBagBayesianPs], 50, color = ['g','r'], label = ['ham emails','spam emails'])
# plt.title('WordsBag feature')
# plt.xlabel('spam chance %')
# plt.ylabel('number of emails with same chance')
# plt.legend()
# plt.show()
#---------------------------------------------
def chanceFormol(value, wordsBag, length, trashNumber, link, longNumber) :
    return (wordsBag + (value + length + trashNumber + link + longNumber)/5)/2
#---------------------------------------------
def hamEmailsSpamChance(index) :
    return chanceFormol(hamValueBayesianPs[index], hamWordsBagBayesianPs[index], hamLengthBayesianPs[index], hamTrashWeightsBayesianPs[index], hamLinkBayesianPs[index], hamLongNumberBayesianPs[index])
def spamEmailsSpamChance(index) :
    return chanceFormol(spamValueBayesianPs[index], spamWordsBagBayesianPs[index], spamLengthBayesianPs[index], spamTrashWeightsBayesianPs[index], spamLinkBayesianPs[index], spamLongNumberBayesianPs[index])

hamEmailsSpamChances = [hamEmailsSpamChance(index) for index in range(hamsNumber)]
spamEmailsSpamChances = [spamEmailsSpamChance(index) for index in range(spamsNumber)]

# plt.hist([hamEmailsSpamChances,spamEmailsSpamChances], 100, color = ['g','r'], label = ['ham emails','spam emails'])
# plt.title('Total chance')
# plt.xlabel('spam chance %')
# plt.ylabel('number of emails with same chance')
# plt.legend()
# plt.show()
#---------------------------------------------
maxEdge = math.floor(max(max(hamEmailsSpamChances), max(spamEmailsSpamChances))) + 1
# print('max edge (max spam chance) :', maxEdge)

def wrongDetected(edge, status) :
    wrongs = 0
    if status == 'ham' :
        for chance in hamEmailsSpamChances :
            if chance >= edge :
                wrongs += 1
    else :
        for chance in spamEmailsSpamChances :
            if chance < edge :
                wrongs += 1
    return wrongs

wrongHamDetectedSpams = [wrongDetected(edge/100, 'ham') for edge in range(100*maxEdge+1)]
wrongSpamDetectedHams = [wrongDetected(edge/100, 'spam') for edge in range(100*maxEdge+1)]
def totalWrongDetected(index) :
    return wrongHamDetectedSpams[index] + wrongSpamDetectedHams[index]

# plt.plot(wrongHamDetectedSpams, color = 'g', label = 'ham emails')
# plt.plot(wrongSpamDetectedHams, color = 'r', label = 'spam emails')
# plt.plot([totalWrongDetected(index) for index in range(100*maxEdge+1)], color = 'y', label = 'total wrong detecteds')
# plt.title('Wrong detected')
# plt.xlabel('Split edge (0.01 %)')
# plt.ylabel('wrong detecteds')
# plt.legend()
# plt.show()
#---------------------------------------------
def correctDetectedSpams(index) :
    return spamsNumber - wrongSpamDetectedHams[index]
def detectedSpams(index) :
    return spamsNumber - wrongSpamDetectedHams[index] + wrongHamDetectedSpams[index]
def correctDetected(index) :
    return hamsNumber + spamsNumber - totalWrongDetected(index)

def recall(index) :
    return 100*correctDetectedSpams(index)/spamsNumber
def precision(index) :
    return 100*correctDetectedSpams(index)/detectedSpams(index) if detectedSpams(index) != 0 else 0
def accuracy(index) :
    return 100*correctDetected(index)/emailsNumber

recalls = [recall(index) for index in range(100*maxEdge+1)]
precisions = [precision(index) for index in range(100*maxEdge+1)]
accuracys = [accuracy(index) for index in range(100*maxEdge+1)]
performances = [(recalls[index]+precisions[index]+accuracys[index])/3 for index in range(100*maxEdge+1)]

# plt.plot(recalls, color = 'g', label = 'Recall')
# plt.plot(precisions, color = 'y', label = 'Precision')
# plt.plot(accuracys, color = 'b', label = 'Accuracy')
# plt.plot(performances, color = 'r', label = 'Performance')
# plt.title('Judge')
# plt.xlabel('edge of slice')
# plt.ylabel('percent * 100')
# plt.legend()
# plt.show()
#---------------------------------------------
bestPerformances = max(performances)
edgeOfSlice = performances.index(bestPerformances) / 100
print('best performance : %', bestPerformances, ', for edge : ', edgeOfSlice)
#---------------------------------------------
def isSpam(chance) :
    if chance >= edgeOfSlice : return True
    else : False
#---------------------------------------------
testFile = './data/test.csv'
testData = pd.read_csv(testFile)

testEmails = testData['text']
testOriginEmails = testEmails.copy()
testStatuses = testData['type']
testNumber = len(testEmails)
#---------------------------------------------
testHasLinks = [hasLink(testEmails[index]) for index in range(testNumber)]
testHasLongNumber = [hasLongNumber(testEmails[index]) for index in range(testNumber)]
testTrashCharsWeights = [0 for i in range(testNumber)]

for index in range(testNumber) :
    email = testEmails[index].lower()
    formedEmail = ""
    for character in email :
        if character >= 'a' and character <= 'z' :
            formedEmail += character
        else :
            formedEmail += ' '
            testTrashCharsWeights[index] += 1
    testTrashCharsWeights[index] /= len(testEmails[index].split()) if testTrashCharsWeights[index] != 0 else 1
    testEmails[index] = formedEmail

testEmailValues = setEmailsValue(testEmails)
testEmailLengths = [len(email) for email in testEmails]
#---------------------------------------------
# plt.hist(testEmailValues, 50, color = 'c')
# plt.title('test emails Values')
# plt.show()
# plt.hist(testEmailLengths, 50, color = 'c')
# plt.title('test emails Lengths')
# plt.show()
# plt.hist(testTrashCharsWeights, 50, color = 'c')
# plt.title('test emails TrashCharsWeights')
# plt.show()
# plt.hist(testHasLinks, 2, color = 'c')
# plt.title('test emails WithLinks')
# plt.show()
# plt.hist(testHasLongNumber, 2, color = 'c')
# plt.title('test emails WithLongNumber')
# plt.show()
#---------------------------------------------
testValuesBayesianPs = [valueBayesian(feature) for feature in testEmailValues]
testLengthsBayesianPs = [lengthBayesian(feature) for feature in testEmailLengths]
testTrashCharsBayesianPs = [trashCharBayesian(feature) for feature in testTrashCharsWeights]
testLinksBayesianPs = [linkBayesian(feature) for feature in testHasLinks]
testLongNumberBayesianPs = [longNumberBayseian(feature) for feature in testHasLongNumber]
testWordsBagBayesianPs = [emailWordsBagBayesian(email.split()) for email in testEmails]

# testBayesiansPs = [testWordsBagBayesianPs,testValuesBayesianPs,testLengthsBayesianPs,testTrashCharsBayesianPs,testLinksBayesianPs,testLongNumberBayesianPs]
# labels = ['words bag','value','length','trash charackter','link','long number']

# plt.hist(testBayesiansPs, 25, label = labels)
# plt.title('test emails bayesians possibilitys')
# plt.xlabel('possibility')
# plt.ylabel('number of emails')
# plt.legend()
# plt.show()
#---------------------------------------------
testSpamChances = [chanceFormol(testValuesBayesianPs[index], testWordsBagBayesianPs[index], testLengthsBayesianPs[index], testTrashCharsBayesianPs[index], testLinksBayesianPs[index], testLongNumberBayesianPs[index]) for index in range(testNumber)]

# plt.hist(testSpamChances, 25, color = 'c')
# plt.title('test emails spam possibilitys')
# plt.xlabel('spam possibility')
# plt.ylabel('number of emails with same chance')
# plt.show()
#---------------------------------------------
thdah = thdas = tsdah = tsdas = 0
# t : test, d : detected, a : as, h : ham, s : spam
# print('========================================================')
for index in range(testNumber) :
    if testStatuses[index] == 'ham' :
        if isSpam(testSpamChances[index]) :
            thdas += 1
            # print('++++++++++++++++++++++++++++++++++++++++++++++++++++++')
            # print(testOriginEmails[index])
            # print('++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        else :
            thdah += 1
    else :
        if isSpam(testSpamChances[index]) :
            tsdas += 1
        else :
            # print('------------------------------------------------------')
            # print(testOriginEmails[index])
            # print('------------------------------------------------------')
            tsdah += 1
# print('========================================================')

testRecall = tsdas / (tsdah + tsdas) if tsdah + tsdas != 0 else 0
testPrecision = tsdas / (thdas + tsdas) if thdas + tsdas != 0 else 0
testAccuracy = (thdah + tsdas) / testNumber
testPerformance = 100*(testRecall + testPrecision + testAccuracy)/3

print('Racall :', testRecall)
print('Precision :', testPrecision)
print('Accuracy :', testAccuracy)
print('Performance: %', testPerformance)