import sys
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import string
import re

from scipy.special import expit
from scipy import sparse
from scipy.sparse import csr_matrix

#from sklearn.model_selection import StratifiedKFold
from sklearn.cross_validation import StratifiedKFold

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import log_loss

from sklearn.cluster import KMeans

from sklearn.ensemble import RandomForestClassifier

from xgboost.sklearn import XGBClassifier
from xgboost.core import DMatrix
from xgboost.training import train, cv

ENGLISH_STOP_WORDS = frozenset([
    "a", "about", "above", "across", "after", "afterwards", "again", "against",  "all", "almost", "alone", "along", "already", "also", "although", "always",  "am", "among", "amongst", "amoungst", "amount", "an", "and", "another", "any", "anyhow", "anyone", "anything", "anyway", "anywhere", "are", "around", "as", "at", "back", "be", "became", "because", "become", "becomes", "becoming", "been", "before", "beforehand", "behind", "being", "below", "beside", "besides", "between", "beyond", "bill", "both", "bottom", "but", "by", "call", "can", "cannot", "cant", "co", "con", "could", "couldnt", "cry", "de", "describe", "detail", "do", "done", "down", "due", "during", "each", "eg", "eight", "either", "eleven", "else", "elsewhere", "empty", "enough", "etc", "even", "ever", "every", "everyone", "everything", "everywhere", "except", "few", "fifteen", "fifty", "fill", "find", "fire", "first", "five", "for", "former", "formerly", "forty", "found", "four", "from", "front", "full", "further", "get", "give", 
    "go", "had", "has", "hasnt", "have", "he", "hence", "her", "here", "hereafter",  "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his", "how", "however", "hundred", "i", "ie", "if", "in", "inc", "indeed", "interest", "into", "is", "it", "its", "itself", "keep", "last", "latter", "latterly", "least", "less", "ltd", "made", "many", "may", "me", "meanwhile", "might", "mill", "mine", "more", "moreover", "most", "mostly",    "move", "much", "must", "my", "myself", "name", "namely", "neither",    "never", "nevertheless", "next", "nine", "no", "nobody", "none", "noone",    "nor", "not", "nothing", "now", "nowhere", "of", "off", "often", "on",    "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our",    "ours", "ourselves", "out", "over", "own", "part", "per", "perhaps",    "please", "put", "rather", "re", "same", "see", "seem", "seemed",    "seeming", "seems", "serious", "several", "she", "should", "show", "side",    "since", "sincere", "six", "sixty", 
    "so", "some", "somehow", "someone",    "something", "sometime", "sometimes", "somewhere", "still", "such",    "system", "take", "ten", "than", "that", "the", "their", "them",    "themselves", "then", "thence", "there", "thereafter", "thereby",    "therefore", "therein", "thereupon", "these", "they", "thick", "thin",    "third", "this", "those", "though", "three", "through", "throughout",    "thru", "thus", "to", "together", "too", "top", "toward", "towards",    "twelve", "twenty", "two", "un", "under", "until", "up", "upon", "us",    "very", "via", "was", "we", "well", "were", "what", "whatever", "when",    "whence", "whenever", "where", "whereafter", "whereas", "whereby",    "wherein", "whereupon", "wherever", "whether", "which", "while", "whither",    "who", "whoever", "whole", "whom", "whose", "why", "will", "with",    "within", "without", "would", "yet", "you", "your", "yours", "yourself",    "yourselves"])

def cleaning_text(text):
    #sentence=sentence.lower()
    text = text.replace('<p><a  website_redacted', '')
    text = text.replace('!<br /><br />', '')
    text = text.replace('kagglemanager renthop com', '')
    text = re.sub('[^\w\s]',' ', text) #removes punctuations
    text = re.sub('\d+',' ', text) #removes digits
    text =' '.join([w for w in text.split() if not w in ENGLISH_STOP_WORDS]) # removes english stopwords
    #text=' '.join([w for w , pos in pos_tag(text.split()) if (pos == 'NN' or pos=='JJ' or pos=='JJR' or pos=='JJS' )])
    #selecting only nouns and adjectives
    text =' '.join([w for w in text.split() if not len(w)<=2 ]) #removes single lettered words and digits
    text = text.strip()
    return text

def cleaning_list(list):
    return [cleaning_text(x.lower()) for x in list]
    #return map(cleaning_text, list)

def feature_engineering(df_train, df_test, y_train):
    print('feature engineering', datetime.datetime.now())

    #for some reason listing_id improves the score

    #df_train.index = df_train['listing_id']
    #df_train = df_train.drop(['listing_id'], axis=1)

    #df_test.index = df_test['listing_id']
    #df_test = df_test.drop(['listing_id'], axis=1)

    #ignore_Index because use sort_index later.
    df_all = pd.concat((df_train, df_test), axis=0, ignore_index=True)

    ###
    ###date feature
    ###
    df_all['created'] = pd.to_datetime(df_all['created'])
    #year - all 2016
    #df['created_year'] = df['created'].dt.year
    df_all['created_month'] = df_all['created'].dt.month
    df_all['created_day'] = df_all['created'].dt.day
    df_all['created_day_of_year'] = df_all['created'].dt.strftime('%j').astype(int)
    df_all['created_hour'] = df_all['created'].dt.hour
    df_all['created_weekday'] = df_all['created'].dt.weekday
    df_all = df_all.drop(['created'], axis=1)

    '''
    #create_weekday categorical
    ohe = OneHotEncoder(sparse=False)
    df_all_ohe = ohe.fit_transform(df_all.created_weekday.reshape(-1, 1)) 	
    for i in range(df_all_ohe.shape[1]):
        df_all['ohe' + str(i)] = df_all_ohe[:, i]
    df_all = df_all.drop(['created_weekday'], axis=1)
    '''
    ###
    ### numeric features
    ###
    #adjust incorrect x/y
    x_mean = df_all.latitude.mean()
    y_mean = df_all.longitude.mean()

    df_all.loc[df_all.latitude < x_mean - 5, 'latitude'] = x_mean - 5
    df_all.loc[df_all.latitude > x_mean + 5, 'latitude'] = x_mean + 5
    df_all.loc[df_all.longitude < y_mean - 5, 'longitude'] = y_mean - 5
    df_all.loc[df_all.longitude > y_mean + 5, 'longitude'] = y_mean + 5

    '''
    #adjust incorrect x/y by percentile
    percentile = 0.1
    llimit = np.percentile(df_all.latitude.values, percentile)
    ulimit = np.percentile(df_all.latitude.values, 100 - percentile)
    df_all.loc[df_all['latitude']<llimit, 'latitude'] = llimit
    df_all.loc[df_all['latitude']>ulimit, 'latitude'] = ulimit
    llimit = np.percentile(df_all.longitude.values, percentile)
    ulimit = np.percentile(df_all.longitude.values, 100 - percentile)
    df_all.loc[df_all['longitude']<llimit, 'longitude'] = llimit
    df_all.loc[df_all['longitude']>ulimit, 'longitude'] = ulimit
    '''

    #log x/y
    df_all['logx'] = np.log(df_all['latitude'])
    df_all['logy'] = np.log(df_all['longitude'] + 100)

    #radius
    df_all['radius'] = np.log((df_all.latitude - x_mean)*(df_all.latitude - x_mean) + (df_all.longitude - y_mean)*(df_all.longitude - y_mean))

    #price
    #df_all.loc[df_all['price'] > 100000, 'price'] = 100000

    #log price
    #df_all['logprice'] = np.log(df_all.price)

    df_all["price_per_bed"] = df_all["price"]/df_all["bedrooms"] 
    df_all["room_dif"] = df_all["bedrooms"] - df_all["bathrooms"] 
    df_all["room_sum"] = df_all["bedrooms"] + df_all["bathrooms"] 
    df_all["price_per_room"] = df_all["price"]/df_all["room_sum"]

    df_all.loc[df_all.price_per_bed == np.inf, 'price_per_bed'] = 10000000
    df_all.loc[df_all.price_per_room == np.inf, 'price_per_room'] = 10000000

    df_all["photos_count"] = df_all["photos"].apply(len)
    df_all = df_all.drop(['photos'], axis=1)
    
    ###
    ###zones
    ###
    n_zones = 140
    x_min = df_all.logx.mean() - 0.004
    x_max = df_all.logx.mean() + 0.003
    y_min = df_all.logy.mean() - 0.003
    y_max = df_all.logy.mean() + 0.003

    df_all2 = df_all[(df_all.logx >= x_min) & (df_all.logx <= x_max) & (df_all.logy >= y_min) & (df_all.logy <= y_max)]
    kmeans = KMeans(n_clusters=n_zones, random_state=0).fit(df_all2[['logx', 'logy']])

    print('zones', df_all.shape)

    for i in range(n_zones):
        df_all['zone' + str(i)] = 0
        df_all.loc[df_all2.logx[kmeans.labels_ == i].index, 'zone' + str(i)] = 1

    print('zones', df_all.shape)

    ###
    ###description
    ###
    df_all['description'] = df_all['description'].apply(lambda x: cleaning_text(x))
    df_all["description_words_count"] = df_all["description"].apply(lambda x: 0 if len(x) == 0 else len(x.split(" ")))
    df_all['description_uppercase'] = df_all['description'].apply(lambda x: 1 if len(re.findall(r'[A-Z]', x))/(len(x) + 1) > 0.5 else 0)
    df_all['description'] = df_all['description'].apply(lambda x: x.lower())

    '''
    n_features2 = 100
    tfidf2 = CountVectorizer(stop_words='english', max_features=n_features2)
    tr_sparse2 = tfidf2.fit_transform(df_all[:df_train.shape[0]]['description'])
    te_sparse2 = tfidf2.transform(df_all[df_train.shape[0]:]['description'])
    '''
    df_all = df_all.drop(['description'], axis=1)

    ###
    ### features
    ###
    df_all["features_count"] = df_all["features"].apply(len)

    n_features = 2000
    df_all['features'] = df_all['features'].apply(lambda x: cleaning_list(x))
    df_all['features'] = df_all['features'].apply(lambda x: " ".join(["_".join(i.split(" ")) for i in x]))
    df_all['features'] = df_all['features'].apply(lambda x: x.lower())
    tfidf = CountVectorizer(stop_words='english', max_features=n_features)
    tr_sparse = tfidf.fit_transform(df_all[:df_train.shape[0]]['features'])
    te_sparse = tfidf.transform(df_all[df_train.shape[0]:]['features'])

    df_all = df_all.drop(['features'], axis=1)

    ###
    ###display and street address
    ###
    df_all['display_address'] = df_all.display_address.str.replace('Avenue', '')
    df_all['display_address'] = df_all.display_address.str.replace(' Ave', '')
    df_all['display_address'] = df_all.display_address.str.replace('Street', '')
    df_all['display_address'] = df_all.display_address.str.replace('St.', '')
    df_all['display_address'] = df_all.display_address.str.replace(' St', '')
    df_all['display_address'] = df_all.display_address.str.rstrip()

    df_all['street_address'] = df_all.street_address.str.replace('Avenue', '')
    df_all['street_address'] = df_all.street_address.str.replace(' Ave', '')
    df_all['street_address'] = df_all.street_address.str.replace('Street', '')
    df_all['street_address'] = df_all.street_address.str.replace('St.', '')
    df_all['street_address'] = df_all.street_address.str.replace(' St', '')
    df_all['street_address'] = df_all.street_address.str.rstrip()

    #keep only the first int from street_address - not a good idea, just the number without street is useless
    #df_all['street_address'] = df_all.street_address.apply(lambda x: x.split(" ")[0])

    ###
    ###categorical features
    ###
    #cannot make them ohe - too many distinct values
    ohe_features = ['building_id', 'display_address', 'manager_id', 'street_address'] 
    for f in ohe_features: 
        le = LabelEncoder() 
        df_all[f] = le.fit_transform(df_all[f]) 

    ###
    ###building_id, manager_id
    ###
    value_counts = df_all['building_id'].value_counts()
    df_all = pd.merge(df_all, pd.DataFrame(value_counts), left_on='building_id', right_index=True).sort_index()
    df_all = df_all.drop(['building_id_x'], axis=1)    
    df_all.loc[df_all.building_id == 0, 'building_id_y'] = 0

    value_counts = df_all['manager_id'].value_counts()
    df_all = pd.merge(df_all, pd.DataFrame(value_counts), left_on='manager_id', right_index=True).sort_index()
    df_all = df_all.drop(['manager_id_x'], axis=1)    
    df_all.loc[df_all.manager_id == 0, 'manager_id_y'] = 0

    print(df_all.shape)

    #done
    X_train = df_all[:df_train.shape[0]]
    X_test = df_all[df_train.shape[0]:]

    X_train = pd.concat((X_train, pd.DataFrame(tr_sparse.todense())), axis=1)
    X_test = pd.concat((X_test, pd.DataFrame(te_sparse.todense())), axis=1)

    #X_train = csr_matrix(np.hstack([X_train, tr_sparse.todense()]))
    #X_test = csr_matrix(np.hstack([X_test, te_sparse.todense()]))
    #X_train = csr_matrix(np.hstack([X_train, tr_sparse.todense(), tr_sparse2.todense()]))
    #X_test = csr_matrix(np.hstack([X_test, te_sparse.todense(), te_sparse2.todense()]))

    print('Train', X_train.shape)
    print('Test', X_test.shape)

    print('feature engineering done', datetime.datetime.now())
    return X_train, X_test

def feature_engineering_extra(df_train, df_test, y_train):
    temp = pd.concat([df_train.manager_id, pd.get_dummies(y_train)], axis = 1).groupby('manager_id').mean()
    temp.columns = ['high_frac', 'medium_frac', 'low_frac']
    #this is equivalent of number of reviews
    temp['manager_listings'] = df_train.groupby('manager_id').count().iloc[:,1]
    #this is equivalent to star rating (0, 1 or 2 stars)
    temp['manager_skill'] = temp['high_frac']*2 + temp['medium_frac']
    #lower the rating for fewer listings
    #temp['manager_skill'] = temp.manager_skill*expit((temp.manager_listings - 1)/4)
    #temp['manager_skill'] = temp.manager_skill*expit(temp.manager_listings/4)

    #use mean for managers with < 20 listings in train. TBD: explain why
    unranked_managers_ixes = temp['manager_listings'] < 20
    ranked_managers_ixes = ~unranked_managers_ixes
    mean_values = temp.loc[ranked_managers_ixes, ['high_frac', 'medium_frac', 'low_frac', 'manager_skill']].mean()
    temp.loc[unranked_managers_ixes, ['high_frac', 'medium_frac', 'low_frac', 'manager_skill']] = mean_values.values

    temp = temp['manager_skill']
    
    #join
    df_train = df_train.merge(temp.reset_index(), how='left', left_on='manager_id', right_on='manager_id')
    #manager with no listing - give them default 0.5 rating
    #df_all2 = df_all2.fillna(0.5)
    #df_all2 = df_all2.fillna(0)
    
    #remove manager_id - score is worse
    #df_train = df_train.drop(['manager_id'], axis=1)    
       
    #join
    df_test = df_test.merge(temp.reset_index(), how='left', left_on='manager_id', right_on='manager_id')
    #manager with no listing - give them default 0.5 rating
    #df_all2 = df_all2.fillna(0.5)
    #df_all2 = df_all2.fillna(0)
    #use mean for managers with no listings. TBD: explain why
    new_manager_ixes = df_test['manager_skill'].isnull()
    df_test.loc[new_manager_ixes, 'manager_skill'] = mean_values['manager_skill']
        
    #remove manager_id - score is worse
    #df_test = df_test.drop(['manager_id'], axis=1)    
        
    '''
    temp = pd.concat([df_all[:df_train.shape[0]].building_id, pd.get_dummies(y_train)], axis = 1).groupby('building_id').mean()
    temp.columns = ['high_frac', 'low_frac', 'medium_frac']
    #this is equivalent of number of reviews
    temp['building_listings'] = df_all[:df_train.shape[0]].groupby('building_id').count().iloc[:,1]
    #this is equivalent to star rating (0, 1 or 2 stars)
    temp['building_skill'] = temp['high_frac']*2 + temp['medium_frac']
    #lower the rating for fewer listings
    #temp['building_skill'] = temp.building_skill*expit((temp.building_listings - 1)/4)
    #temp['building_skill'] = temp.building_skill*expit(temp.building_listings/4)
        
    #use mean for buildings with < 20 listings in train. TBD: explain why
    unranked_managers_ixes = temp['building_listings'] < 20
    ranked_managers_ixes = ~unranked_managers_ixes
    mean_values = temp.loc[ranked_managers_ixes, ['high_frac','low_frac', 'medium_frac','building_skill']].mean()
    temp.loc[unranked_managers_ixes,['high_frac','low_frac', 'medium_frac','building_skill']] = mean_values.values

    temp = temp['building_skill']
    #join
    df_all2 = df_all.merge(temp.reset_index(), how='left', left_on='building_id', right_on='building_id')
    #building with no listing - give them default 0.5 rating
    #df_all2 = df_all2.fillna(0.5)
    #df_all2 = df_all2.fillna(0)
    #use mean for buidlings with no listings. TBD: explain why
    df_all2 = df_all2.fillna(mean_values['building_skill'])
    df_all['building_skill'] = df_all2['building_skill']

    #remove building_id?
    #df_all = df_all.drop(['building_id'], axis=1)    
    '''

    return df_train, df_test

def my_cv(clf, X_train, y_train):
    early_stopping_rounds = 100

    xgb_options = clf.get_xgb_params()
    xgb_options['num_class'] = 3
    xgb_options.update({"eval_metric":'mlogloss'})
    train_dmatrix = DMatrix(csr_matrix(X_train), label=y_train)

    folds = StratifiedKFold(y_train, n_folds=5, shuffle=True)
    cv_results = cv(xgb_options, train_dmatrix, clf.n_estimators, early_stopping_rounds=early_stopping_rounds, verbose_eval=False, show_stdv=False, folds=folds)

    return cv_results.values[-1][0], cv_results.shape[0]

def my_find_n_estimators(clf, X_train, y_train, n_estimators):
    xgb_options = clf.get_xgb_params()
    xgb_options['num_class'] = 3
    xgb_options.update({"eval_metric":'mlogloss'})
    train_dmatrix = DMatrix(csr_matrix(X_train), label=y_train)

    folds = StratifiedKFold(y_train, n_folds=5, shuffle=True)
    #cv_results = cv(xgb_options, train_dmatrix, clf.n_estimators, verbose_eval=False, show_stdv=False, folds=folds)
    cv_results = cv(xgb_options, train_dmatrix, n_estimators, verbose_eval=False, show_stdv=False, folds=folds)

    return cv_results.values[-1][0], cv_results['test-mlogloss-mean'].values

is_tt = 0
is_tt_rf = 0
is_cv = 0
is_gs = 0
is_find_n = 0
is_submit = 0
if __name__ == '__main__':
    if sys.argv[1] == 'tt':
        print('is_tt')
        is_tt = 1
    elif sys.argv[1] == 'tt_rf':
        print('is_tt_rf')
        is_tt_rf = 1
    elif sys.argv[1] == 'cv':
        print('is_cv')
        is_cv = 1
    elif sys.argv[1] == 'gs':
        print('is_gs')
        is_gs = 1
    elif sys.argv[1] == 'find_n':
        print('is_find_n')
        is_find_n = 1
    elif sys.argv[1] == 'submit':
        print('is_submit')
        is_submit = 1
    else:
        abc += 1

    np.random.seed(0)

    train_file = 'Dataset\\train.json'
    test_file = 'Dataset\\test.json'
    print('load data', datetime.datetime.now())
    df_train = pd.read_json(train_file)
    df_test = pd.read_json(test_file)
    print(df_train.shape)
    print(df_test.shape)
    print('load data done', datetime.datetime.now())

    target_num_map = {'high':0, 'medium':1, 'low':2}
    y_train = np.array(df_train['interest_level'].apply(lambda x: target_num_map[x]))
    df_train = df_train.drop(['interest_level'], axis=1)

    if is_tt == 1:
        X_train, X_test = feature_engineering(df_train, df_test, y_train)
    
        early_stopping_rounds = 100
        #learning_rate, max_depth, ss, cs, gamma, min_child_weight, reg_lambda, reg_alpha = 0.1, 6, 0.7, 0.7, 0, 1, 1, 0
        learning_rate, max_depth, ss, cs, gamma, min_child_weight, reg_lambda, reg_alpha = 0.1, 4, 0.8, 0.8, 0, 1, 1, 0
        clf = XGBClassifier(max_depth=max_depth, learning_rate=learning_rate, n_estimators=5000, objective='multi:softprob', subsample=ss, colsample_bytree=cs, gamma=gamma, min_child_weight=min_child_weight, reg_lambda=reg_lambda, reg_alpha=reg_alpha)

        scores2 = []
        for i in range(10):
            folds = StratifiedKFold(y_train, n_folds=5, shuffle=True)
            scores = []
            iterations = []
            for train_index, test_index in folds:
                X_train2, X_test2 = X_train.loc[train_index], X_train.loc[test_index]
                y_train2, y_test2 = y_train[train_index], y_train[test_index]

                X_train2, X_test2 = feature_engineering_extra(X_train2, X_test2, y_train2)

                X_train2 = csr_matrix(X_train2.values)
                X_test2 = csr_matrix(X_test2.values)

                clf.fit(X_train2, y_train2, eval_set=[(X_test2, y_test2)], eval_metric='mlogloss', early_stopping_rounds=early_stopping_rounds, verbose=False)
                #print(round(clf.booster().best_score, 6), int(clf.booster().best_ntree_limit))
                scores.append(round(clf.booster().best_score, 6))
                iterations.append(int(clf.booster().best_ntree_limit))

            scores = np.array(scores)
            iterations = np.array(iterations)
            score = scores.mean()
            scores2.append(score)
            print('score, std, iterations', score, scores.std(), iterations.mean())

        scores = np.array(scores2)
        scores = np.delete(scores, [scores.argmax(), scores.argmin()])
        print('score, std', scores.mean(), scores.std())

    if is_tt_rf == 1:
        X_train, X_test = feature_engineering(df_train, df_test, y_train)
    
        clf = RandomForestClassifier(n_estimators=1000, n_jobs=-1)

        scores2 = []
        for i in range(10):
            folds = StratifiedKFold(y_train, n_folds=5, shuffle=True)
            scores = []
            for train_index, test_index in folds:
                X_train2, X_test2 = X_train.loc[train_index], X_train.loc[test_index]
                y_train2, y_test2 = y_train[train_index], y_train[test_index]

                X_train2, X_test2 = feature_engineering_extra(X_train2, X_test2, y_train2)

                clf.fit(X_train2, y_train2)
                y_pred = clf.predict_proba(X_test2)
                score = log_loss(y_test2, y_pred)
                scores.append(round(score, 6))

            scores = np.array(scores)
            score = scores.mean()
            scores2.append(score)
            print('score, std', score, scores.std())

        scores = np.array(scores2)
        scores = np.delete(scores, [scores.argmax(), scores.argmin()])
        print('score, std', scores.mean(), scores.std())

    if is_cv == 1:
        X_train, X_test = feature_engineering(df_train, df_test, y_train)
    
        #learning_rate, max_depth, ss, cs, gamma, min_child_weight, reg_lambda, reg_alpha = 0.1, 6, 0.7, 0.7, 0, 1, 1, 0
        learning_rate, max_depth, ss, cs, gamma, min_child_weight, reg_lambda, reg_alpha = 0.1, 4, 0.8, 0.8, 0, 1, 1, 0
        clf = XGBClassifier(max_depth=max_depth, learning_rate=learning_rate, n_estimators=5000, objective='multi:softprob', subsample=ss, colsample_bytree=cs, gamma=gamma, min_child_weight=min_child_weight, reg_lambda=reg_lambda, reg_alpha=reg_alpha)
    
        scores = []
        for i in range(10):
            score, iterations = my_cv(clf, X_train, y_train)
            print('score', score, 'iterations', iterations)
            scores.append(score)

        scores = np.array(scores)
        scores = np.delete(scores, [scores.argmax(), scores.argmin()])
        print('my_cv mean, std', scores.mean(), scores.std())

    if is_find_n == 1:
        X_train, X_test = feature_engineering(df_train, df_test, y_train)
    
        #learning_rate, max_depth, ss, cs, gamma, min_child_weight, reg_lambda, reg_alpha = 0.1, 6, 0.7, 0.7, 0, 1, 1, 0
        learning_rate, max_depth, ss, cs, gamma, min_child_weight, reg_lambda, reg_alpha = 0.1, 4, 0.8, 0.8, 0, 1, 1, 0
        clf = XGBClassifier(max_depth=max_depth, learning_rate=learning_rate, n_estimators=5000, objective='multi:softprob', subsample=ss, colsample_bytree=cs, gamma=gamma, min_child_weight=min_child_weight, reg_lambda=reg_lambda, reg_alpha=reg_alpha)
    
        df = pd.DataFrame()
        for i in range(10):
            score, results = my_find_n_estimators(clf, X_train, y_train, 1000)
            print('iteration', i, 'score', score)
            df['column' + str(i)] = results

        print('score', df.sum(axis=1).min()/10)
        print('iteration', df.sum(axis=1).argmin() + 1)

    if is_gs == 1:
        learning_rate, max_depth, ss, cs, gamma, min_child_weight, reg_lambda, reg_alpha = 0.1, 6, 0.7, 0.7, 0, 1, 1, 0
        clf = XGBClassifier(max_depth=max_depth, learning_rate=learning_rate, n_estimators=5000, objective='multi:softprob', subsample=ss, colsample_bytree=cs, gamma=gamma, min_child_weight=min_child_weight, reg_lambda=reg_lambda, reg_alpha=reg_alpha)

        scores_out = []

        for n_features in range(50, 1201, 50):
            np.random.seed(0)
            print('features', n_features)
            X_train, X_test = feature_engineering(df_train, df_test, y_train, n_features)

            scores = []
            for i in range(10):
                score, _ = my_cv(clf, X_train, y_train)
                print('score', score)
                scores.append(score)

            scores = np.array(scores)
            scores = np.delete(scores, [scores.argmax(), scores.argmin()])
            print('my_cv mean, std', scores.mean(), scores.std())

            scores_out.append(scores.mean())

        for i in range(len(scores_out)):
            print(scores_out[i])

    if is_submit == 1:    
        X_train, X_test = feature_engineering(df_train, df_test, y_train)

        learning_rate, max_depth, ss, cs, gamma, min_child_weight, reg_lambda, reg_alpha = 0.1, 6, 0.7, 0.7, 0, 1, 1, 0
        #clf = XGBClassifier(max_depth=max_depth, learning_rate=learning_rate, n_estimators=351, objective='multi:softprob', subsample=ss, colsample_bytree=cs, gamma=gamma, min_child_weight=min_child_weight, reg_lambda=reg_lambda, reg_alpha=reg_alpha)
        clf = XGBClassifier(max_depth=max_depth, learning_rate=learning_rate, n_estimators=328, objective='multi:softprob', subsample=ss, colsample_bytree=cs, gamma=gamma, min_child_weight=min_child_weight, reg_lambda=reg_lambda, reg_alpha=reg_alpha)

        #learning_rate, max_depth, ss, cs, gamma, min_child_weight, reg_lambda, reg_alpha = 0.1, 4, 0.8, 0.8, 0, 1, 1, 0
        #clf = XGBClassifier(max_depth=max_depth, learning_rate=learning_rate, n_estimators=862, objective='multi:softprob', subsample=ss, colsample_bytree=cs, gamma=gamma, min_child_weight=min_child_weight, reg_lambda=reg_lambda, reg_alpha=reg_alpha)

        clf.fit(csr_matrix(X_train), y_train)
        y_predict = clf.predict_proba(csr_matrix(X_test))
        df_out = pd.DataFrame(y_predict)
        df_out.columns = ["high", "medium", "low"]
        df_out["listing_id"] = df_test.listing_id.values
        df_out.to_csv("Output/results.csv", index=False)

    print('done', datetime.datetime.now())

