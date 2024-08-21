'''
Created on Jul 16, 2023

@author: slawek

'''
OUTPUT_WINDOW=48
MANDATORY_TESTING_QUANTS=[0.5, 0.001, 0.01, 0.05, 0.1, 0.25, 0.75, 0.9, 0.95, 0.99, 0.999]
NUM_OF_TESTING_QUANTS=100 #apart from MANDATORY_TESTING_QUANTS 


#when it comes to testing

#for each time series
  forecsSavingPath=outputDir+"e"+str(iEpoch)+"w"+str(workerNumber)+"_"+str(series)+".pickle"  #workerNo starts with 1. Naming convention important.
  oneSeriesForecs=[]
  quants=MANDATORY_TESTING_QUANTS+list(np.random.uniform(size=NUM_OF_TESTING_QUANTS))
  #create batch with the same time series repeated len(quants), but we will append one quant from the quants list
  #generate forecast. Now save it
  for horizon in range(OUTPUT_WINDOW):
    qForecs1=forec[:len(MANDATORY_TESTING_QUANTS),horizon].astype(float) #do not sort the mandatory quantiles, assumption is let's see the real thing, and also they are far apart, unlikely to cross
    qForecs2=sorted(forec[len(MANDATORY_TESTING_QUANTS):,horizon].astype(float)) #sort forecsts for the 100 random quantiles
    qForec=np.concatenate((qForecs1, qForecs2), axis=0)
    
    if workerNumber==1: # or workerNumber==4: #I am running ensemble, so each worker will save its results. But worker no 1 is special, it saves also actuals and quantiles
      actu=float(actuals[0,horizon])
      actuas=[actu]*len(quants)
      
      quants1=MANDATORY_TESTING_QUANTS
      quants2=sorted(quants[len(MANDATORY_TESTING_QUANTS):])  #sort 100 random quantiles
      quants12=quants1+quants2

      save_df=pd.DataFrame(zip(quants12, actuas, list(qForec)), columns=["quants","actuals","forec"+str(workerNumber)]) #for each horizon, save_df has len(quants) rows and 3 (workerNo=1) or 1 column
    else:
      save_df=pd.DataFrame(list(qForec), columns=["forec"+str(workerNumber)])  

    save_df=save_df.astype(np.float32)
    oneSeriesForecs.append(save_df)
    
  oneSeriesForec = pd.concat(oneSeriesForecs, axis=0)
  with open(forecsSavingPath, 'wb') as handle:
    pickle.dump(oneSeriesForec, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    
"""
e.g oneSeriesForecas of worker 1 is Pandas dataframe that may look like this: (Column names are important!)
       quants     actuals      forec1
0    0.500000  799.599976  798.159241
1    0.001000  799.599976  730.221924
2    0.010000  799.599976  752.763184
3    0.050000  799.599976  775.873230
4    0.100000  799.599976  782.751343
..        ...         ...         ...
106  0.957208         NaN  914.610657
107  0.963054         NaN  916.142090
108  0.964627         NaN  916.569092
109  0.985264         NaN  923.161133
110  0.997581         NaN  929.123047

[1939392 rows x 3 columns]

So this is how your output should look like if you use only one worker.

If you use more then, e.g. oneSeriesForecs of a worker 9 may look like this
         forec9
0    777.986572
1    706.660095
2    726.332275
3    748.451599
4    756.430725
..          ...
106  893.801697
107  897.119873
108  899.212097
109  901.271179
110  903.122009

[1939392 rows x 1 columns]


They are matching by rows, we have 11+100=111 quantiles (so rows) for each horizon for each series, so 111*48 rows for a single forecast for a series. 
1939392/(111*48)=364 so we do 364 daily forecast in testing

""