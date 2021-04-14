import pandas as pd
import json
import nltk

def clean():
    print("clean! \n")
    data = {}
    with open('Data/homework.json') as f:
        i = 0
        for line in f:
            d = json.loads(line)
            data[i] = d
            i+=1

    df = pd.DataFrame.from_dict(data, orient = 'index')

    #remove nonsense column
    df = df.drop(columns = ['emotion_9'])

    # remove weird boolean rows
    df['drop'] = False
    for i in range(9):
        df[f'emotion_{i}'] = pd.to_numeric(df[f'emotion_{i}'],errors = 'coerce')
        df.loc[df[f'emotion_{i}'].isin([0.,1.])==False,"drop"] = True

    # get rid of dumb text
    drops = ["fnord","-1","cat","-2",""]
    df.loc[df[f'headline'].isin(drops),"drop"] = True
    df.loc[df[f'summary'].isin(drops),"drop"] = True
    df = df.loc[df['drop']==False]

    #sum ratings on story level
    df2 = df.drop(columns = ["drop","worker_id"]).groupby(["headline","summary"]).sum().reset_index()
    sc = df2['summary'].value_counts().reset_index()
    summary_drops = list(sc.loc[sc["summary"]>1,"index"]) #only keep single summary-headline
    df3 = df2.loc[~df2['summary'].isin(summary_drops)] # remove duplicates

    #get modal rating for story
    rename_dic = dict(zip([f"emotion_{x}" for x in range(9)],[x for x in range(9)]))
    df3 = df3.rename(columns = rename_dic)
    df3['emotion']=df3[[x for x in range(9)]].idxmax(axis=1)

    #stem the text
    df3 = df3.reset_index().rename(columns = {"emotion":"labels","index":"idx"})
    df3[["idx","headline","summary","labels"]].to_csv("Data/clean_unstemmed.csv",index=False)

    ps = nltk.stem.PorterStemmer()
    for var in ["headline","summary"]:
        df3[var] = df3[var].apply(lambda x: ps.stem(x))

    #serialize
    df3[["idx","headline","summary","labels"]].to_csv("Data/clean.csv",index=False)
    df3[["idx","headline","summary","labels"]].loc[:1000,:].to_csv("Data/clean_test.csv",index=False)