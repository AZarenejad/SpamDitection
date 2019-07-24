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
#---------------------------------------------
hamLengths = [len(email) for email in hamEmails]
spamLengths = [len(email) for email in spamEmails]
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
#---------------------------------------------
wordsValue = dict()

def updateWordsValue(emails, status) :
    for email in emails :
        for word in email.split() :
            if len(word) < 2 : continue
            word = WordNetLemmatizer().lemmatize(word,'v')
            word = singularize(word)
            if word not in wordsValue :
                wordsValue[word] = 0
            wordsValue[word] += 1/hamsNumber if status == 'ham' else -1/spamsNumber

updateWordsValue(hamEmails, 'ham')
updateWordsValue(spamEmails, 'spam')
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
#---------------------------------------------
maxEdge = math.floor(max(max(hamEmailsSpamChances), max(spamEmailsSpamChances))) + 1

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
#---------------------------------------------
bestPerformances = max(performances)
edgeOfSlice = performances.index(bestPerformances) / 100
print('best performance : %', bestPerformances, ', for edge : ', edgeOfSlice)
#---------------------------------------------
def isSpam(chance) :
    if chance >= edgeOfSlice : return True
    else : False
#---------------------------------------------
evalFile = './data/evaluate.csv'
evalData = pd.read_csv(evalFile)

evalEmails = evalData['text']
evalIds = evalData['id']
evalNumber = len(evalEmails)
#---------------------------------------------
evalHasLinks = [hasLink(evalEmails[index]) for index in range(evalNumber)]
evalHasLongNumber = [hasLongNumber(evalEmails[index]) for index in range(evalNumber)]
evalTrashCharsWeights = [0 for i in range(evalNumber)]

for index in range(evalNumber) :
    email = evalEmails[index].lower()
    formedEmail = ""
    for character in email :
        if character >= 'a' and character <= 'z' :
            formedEmail += character
        else :
            formedEmail += ' '
            evalTrashCharsWeights[index] += 1
    evalTrashCharsWeights[index] /= len(evalEmails[index].split()) if evalTrashCharsWeights[index] != 0 else 1
    evalEmails[index] = formedEmail

evalEmailValues = setEmailsValue(evalEmails)
evalEmailLengths = [len(email) for email in evalEmails]
#---------------------------------------------
evalValuesBayesianPs = [valueBayesian(feature) for feature in evalEmailValues]
evalLengthsBayesianPs = [lengthBayesian(feature) for feature in evalEmailLengths]
evalTrashCharsBayesianPs = [trashCharBayesian(feature) for feature in evalTrashCharsWeights]
evalLinksBayesianPs = [linkBayesian(feature) for feature in evalHasLinks]
evalLongNumberBayesianPs = [longNumberBayseian(feature) for feature in evalHasLongNumber]
evalWordsBagBayesianPs = [emailWordsBagBayesian(email.split()) for email in evalEmails]
#---------------------------------------------
evalSpamChances = [chanceFormol(evalValuesBayesianPs[index], evalWordsBagBayesianPs[index], evalLengthsBayesianPs[index], evalTrashCharsBayesianPs[index], evalLinksBayesianPs[index], evalLongNumberBayesianPs[index]) for index in range(evalNumber)]
#---------------------------------------------
evalTypePredict = ['spam' if isSpam(evalSpamChances[index]) else 'ham' for index in range(evalNumber)]
result = {'id' : evalIds, 'type' : evalTypePredict}
resultFile = DataFrame(result, columns = ['id', 'type'])
export_csv = resultFile.to_csv ('./data/output.csv', index = None, header=True)