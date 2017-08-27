import tensorflow as tf
import numpy as np
from datetime import date, timedelta
import datetime

dict = {}
window_width = 7
max_sentence_length = 10
def normalise_windows(window_data):
    normalised_data = []
    for window in window_data:
        #mean = np.mean(np.array(window).astype(np.float))
        #std = np.std(np.array(window).astype(np.float))
        window = np.array(window).astype(np.float).tolist()
        min_value = min(window)
        max_value = max(window)
        normalised_window = [ (float(p) - min_value)/(max_value - min_value) for p in window]
        normalised_data.append(normalised_window)
    return normalised_data  
def add_dictionary(news_data):
    
    text = news_data.split('||');
    text_vectors = text[1].split()
    for word in text_vectors:
        word = word.replace(".", "").lower()
        if  word in dict:
            dict[word] += 1
        else :
            dict[word] = 1

# def build_dict(news_data):
    # for d in news_data:
        # print (news_data)
        # d = d.split('||');
        # add_dictionary(d[1])

def get_word_ids():
    
    n_dict = {}
    i = 2
    for key, value in dict.items():
        n_dict[key] = i
        i = i+1
    return n_dict
    
def get_news(data, stt, end):
    
    result =  np.zeros([window_width, max_sentence_length], np.int32)
    i = 0
    for d in data:
        dt = date(int(d[0][0]), int(d[0][1]), int(d[0][2]))
        if (stt <= dt <= end):
            result[i] = d[1]
            i = i + 1
            if (i >= window_width):
                break

    return result

def words_to_ids(text):
    words = text.split()
    res = []
    res = np.zeros(max_sentence_length, np.int32)
    i = 0
    for w in words:
        try:
            res[i] = dict[w]
            i = i + 1
        except:
            i = i
    return res
    
def read_data(prices_path, news_path):
    
    #news format would by ...  2000-05-14 || e1 | R | e2
    
    f = open(prices_path, 'rb').read()
    data = f.decode().split('\n')
    
    f = open(news_path, 'r').read()
    news_data = f.split('\n')
    
    prices = []
    dates  = []
    
    news = []
    for i in range (len(data)):
        dates.append(data[i].split()[0].split("-"))
        prices.append(data[i].split()[1])
    
    for i in range (len(news_data)):
        add_dictionary(news_data[i])
        news.append([news_data[i].split("||")[0].split("-"), news_data[i].split("||")[1]])  
    
    windowed = []

    for i in range (len(prices) - window_width):
        windowed.append(prices[i:i+window_width])
        
    windowed = normalise_windows(windowed)
    windowed = np.array(windowed).astype(np.float)
	
    dict = get_word_ids()
    
    
    for i in range (len(news)):
        news[i][1] = words_to_ids(news[i][1])
        
    prices = np.array(prices).astype(np.float)
    input_date = np.array(dates).astype(np.int)
    
    labels = prices[window_width:]
    input_date = input_date[window_width:]
    
    tmp1 = []
    tmp1.append([1, 0])

    for i in range (1, len(labels) - 1):
        # increased from the prior price = [1, 0]
        if ( labels[i] > labels[i-1] ):
            tmp1.append([1, 0])
        else :
            tmp1.append([0, 1])
    
    labels = tmp1

    news_input = []
    
    for d in input_date:
        end = date(d[0], d[1], d[2])
        stt = end - timedelta(days=window_width)
        res = get_news(news, stt, end)
        news_input.append(res)
     
        
        
    # return array [date, array of sentence]
    return news_input, windowed, labels
    

#build_dict("test_dataset.txt")
r = read_data("close_price_msft.txt","title_events_extracted.txt")