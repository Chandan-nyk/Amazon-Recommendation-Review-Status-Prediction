#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#This project focusses on the following areas :

#Analysis of the dataset
#Understanding of the User's Rating Distribution
#Predict Recommend Status based on the subjective review provided by the user


# In[ ]:


#Cleaning the Datset


# In[ ]:


import pandas as pd
df = pd.read_csv(r"C:\Users\nayacha\Downloads\Consumer Reviews of Amazon Products/1429_1.csv")
df1_oasis = df.iloc[2816:3482,]
df2_fire_16gb = df.iloc[14448:15527,]
df3_paperwhite_4gb = df.iloc[17216:20392,]
df4_voyage = df.iloc[20392:20972,]
df5_paperwhite = df.iloc[20989:21019,]
#df.head(5)


# In[13]:


print(df1_oasis.shape)
print(df2_fire_16gb.shape)
print(df3_paperwhite_4gb.shape)
print(df4_voyage.shape)
print(df5_paperwhite.shape)


# In[14]:


df1_oasis.to_csv('Oasis.csv')
df2_fire_16gb.to_csv('Fire.csv')
df4_voyage.to_csv('Voyage.csv')


# In[15]:


frames = [df3_paperwhite_4gb,df5_paperwhite]
df4_voyage.to_csv('Voyage.csv')
kp = pd.concat(frames)
print(kp.head(5))
print(kp.tail(5))
kp = kp.reset_index()
print(kp.columns)
print(kp['reviews.rating'].describe())
kp.columns = ['Index','ID','Name','ASINS','Brand','Categories','Keys','Manufacturer','ReviewDate','ReviewDateAdded','ReviewDateSeen','PurchasedOnReview','RecommendStatus','ReviewID','ReviewHelpful','Rating','SourceURL','Comments','Title','UserCity','UserProvince','Username']
kp.columns
kp.head(5)
print(kp.columns.nunique())
kp = kp.drop(['ReviewID' , 'UserCity' , 'UserProvince','PurchasedOnReview'],axis = 1)
print(kp.columns.nunique())
print(kp.Rating.value_counts())
kp.Rating.value_counts()


# In[ ]:


#Analysing Data


# In[16]:


kp.RecommendStatus.nunique()
import matplotlib.pyplot as plt
kp.hist(column = 'Rating', by = 'RecommendStatus', color = 'Red')
plt.show()
print(kp.info())


# In[17]:


kp['Categories'] = 'Tablets'
kp['Name'] = 'Amazon Kindle Paperwhite'
print(kp.head(5))
print(kp.ReviewHelpful.value_counts())


# In[18]:


pd.DataFrame(kp[(kp.Rating==5)&(kp.RecommendStatus==False)]['Comments'])


# In[19]:


print(kp.Username.nunique())
print(kp.shape)
sum(kp['Username'].value_counts()>1)


# In[20]:


len(kp['Username'].value_counts()>1)


# In[21]:


kp.head(2)
kp = kp.drop('Keys',axis = 1)
print(kp.columns.nunique())
kp =kp.reset_index()
print(kp.head(2))


# In[ ]:


#Transforming Date Time


# In[22]:


kp.ReviewDate = pd.to_datetime(kp['ReviewDate'], dayfirst= True)
kp.ReviewDateAdded =pd.to_datetime(kp.ReviewDateAdded , dayfirst= True)
#kp.ReviewDateSeen = pd.to_datetime(kp.ReviewDateSeen, dayfirst = True)


# In[23]:


kp['ReviewDateSeen'] = kp['ReviewDateSeen'].str.split(',',expand = True).apply(lambda x:x.str.strip())
kp.ReviewDateSeen = pd.to_datetime(kp.ReviewDateSeen,dayfirst= True)   
print(kp.head(4))


# In[ ]:


#Likert Scale Analysis


# In[24]:


import numpy as np
promoters = sum(kp.Rating==5)
passive = sum(kp.Rating == 4)
detractors = sum(np.logical_and(kp.Rating >= 1, kp.Rating <=3))
respondents = promoters+passive+detractors
NPS_P = ((promoters - detractors)/respondents )*100
print(NPS_P)


# In[25]:


print(kp.tail(2))


# In[26]:


kp.plot(x = 'ReviewDate',y = 'Rating', kind = 'line',  figsize=(10,10))


# In[27]:


review_date = kp.ReviewDate
rating = kp.Rating
df_dr = pd.concat([review_date,rating],axis = 1)
print(df_dr.tail(5))
print(df_dr.shape)


# In[28]:


df_dr = df_dr.groupby(['ReviewDate','Rating']).size().unstack(fill_value = 0)
print(df_dr.loc['2017-02-04'])


# In[29]:


print(df_dr.head(5))


# In[30]:


df_dr.columns = ['A','B','C','Passive','Promoters']
df_dr['Detractors'] = df_dr['A'] + df_dr['B'] + df_dr['C']
df_dr.head(5)


# In[31]:


df_dr = df_dr.drop(labels = ['A','B','C'],axis = 1)
print(df_dr.head(5))


# In[32]:


df_dr['NPS'] = (df_dr['Promoters'] - df_dr['Detractors']) * 100 / (df_dr['Passive'] + df_dr['Promoters'] + df_dr['Detractors'])
print(df_dr.head(5))


# In[33]:


df_dr = df_dr.reset_index()
df_dr.plot( x = 'ReviewDate', y = 'NPS',kind = 'line', figsize=(10,10))


# In[34]:


df_dr.shape


# In[ ]:


#Sentiment Analysis - NLTK to find Compound Score


# In[35]:


kp.Name.nunique()
kp.head(2)


# In[36]:


data =  kp.drop(['Index','ID','Name','ASINS','Brand','Categories','Manufacturer','ReviewDateAdded','ReviewDateSeen','SourceURL'], axis = 1)
# Cleaned Dataset Now becomes
data = data.reset_index()
data.head(5)


# In[37]:


data = data.drop(['ReviewDate'], axis = 1)
data.columns


# In[38]:


def status(data):
    if(data == True):
        data = "Recommend"
        return data
    else:
        data = "Not Recommend"
        return data
    
data['RecommendStatus'] = pd.DataFrame(data['RecommendStatus'].apply(lambda x : status(x)))
data.head(5)


# In[41]:


dsa = data
dsa['feedback'] = dsa['Comments'] + dsa['Title']
dsa = dsa.drop(['Comments','Title'], axis = 1)


# In[42]:


dsa.head(5)


# In[43]:


import nltk


# In[44]:


from nltk.sentiment.vader import SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()

def polar_score(text):
    score = sid.polarity_scores(text)
    x = score['compound']
    return x


dsa['Compound_Score'] = dsa['feedback'].apply(lambda x : polar_score(x))


# In[45]:


dsa.head(5)


# In[46]:


dsa['length'] = dsa['feedback'].apply(lambda x: len(x) - x.count(" "))
dsa.head(2)


# In[ ]:


#Ideally people who'll Not Recommend the product, would have a lot to say against the features of the product


# In[47]:


import numpy as np
from matplotlib import pyplot
get_ipython().run_line_magic('matplotlib', 'inline')


# In[51]:


bins = np.linspace(0,200,40)
pyplot.hist(dsa[dsa['RecommendStatus'] == 'Not Recommend']['length'],bins,alpha  = 0.5,density = True, label = 'Not Recommend')
pyplot.hist(dsa[dsa['RecommendStatus'] == 'Recommend']['length'],bins,alpha = 0.5,density = True, label = 'Recommend')
pyplot.legend(loc = 'upper right')
pyplot.show()


# In[ ]:


#Using TF-IDF and Random Forest to predict Recommendation Status


# In[52]:


import string
import nltk
import re
stopword =  nltk.corpus.stopwords.words('english')
ps = nltk.PorterStemmer()


# In[53]:


def clean(text):
    no_punct = "".join([word.lower() for word in text if word not in string.punctuation])
    tokens = re.split('\W+',no_punct)
    text_stem = ([ps.stem(word) for word in tokens if word not in stopword])
    return text_stem


# In[54]:


from sklearn.feature_extraction.text import TfidfVectorizer
tf_idf = TfidfVectorizer(analyzer= clean)
Xtf_idfVector = tf_idf.fit_transform(dsa['feedback'])


# In[55]:


import pandas as pd

Xfeatures_data = pd.concat([dsa['Compound_Score'], dsa['length'], pd.DataFrame(Xtf_idfVector.toarray())], axis = 1)
Xfeatures_data.head(5)


# In[ ]:


#dataframe we would be applying Machine Learning to


# In[56]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(Xfeatures_data, dsa['RecommendStatus'], test_size = 0.2)

rf = RandomForestClassifier(n_estimators= 50, max_depth= 20, n_jobs= -1)
rf_model = rf.fit(X_train,y_train)
sorted(zip(rf.feature_importances_,X_train.columns), reverse = True)[0:10]


# In[ ]:


#Applying Grid Search to change hyper parameters and then applying RF


# In[57]:


def compute(n_est, depth):
    rf = RandomForestClassifier(n_estimators= n_est, max_depth= depth)
    rf_model = rf.fit(X_train, y_train)
    y_pred  = rf_model.predict(X_test)
    precision,recall,fscore,support  = score(y_test,y_pred, pos_label= 'Recommend', average = 'binary')
    print('Est: {}\ Depth: {}\ Precision: {}\ Recall: {}\ Accuracy: {}'.format(n_est, depth, round(precision,3), round(recall,3), (y_pred == y_test).sum()/ len(y_pred)))


# In[58]:


for n_est in [10,30,50,70]:
    for depth in [20,40,60,80,90]:
        compute(n_est,depth)
    


# In[ ]:


#Feature Engineering played a key role in boosting the model's performance matrix. 
#The length of the text and calculation of compound_score using sentiment analysis served as a basis to strike a balance between Precision & Recall (0.975 vs 1.0).
#further made the model robust enough to predict user's recommend status to 97.5%

#This concludes our Analysis of the Kindle Paperwhite.

