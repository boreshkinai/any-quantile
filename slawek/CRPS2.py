'''
Created on Apr 1, 2023

@author: slawek

Read test output files and calculate some probabilistic forecast metrics including CRPS
Added plot of relative frequencies of mandatory quantiles
'''

import glob
import numpy as np
import pandas as pd
import os
import pickle
#import properscoring as ps
import matplotlib.pyplot as plt

OUTPUT_DIR="results/MHLV/"
SHORT_RUN="NBEATSAQFILM-maxnorm=False-loss=MQNLoss" 
outputDir=OUTPUT_DIR+SHORT_RUN+"/"
EPOCH=1
USE_MEDIAN_AGG=False #median or mean aggregation over forecasts


if __name__ == '__main__' or __name__ == 'builtins':
  #e6w5_ITH4.pickle
  filePatterns=outputDir+"e"+str(EPOCH)+"*.pickle"
  filePaths= glob.glob(filePatterns)
  len(filePaths)
  #f=filePaths[1]
  series_s=set()
  for f in filePaths:
    start=f.rfind('_')
    end=f.rfind('.pickle')
    series=f[start+1:end]
    series_s.add(series)
  print('There are',len(series_s),'series saved, total number of files:',len(filePaths))
  
  MANDATORY_TESTING_QUANTS=[0.5, 0.001, 0.01, 0.05, 0.1, 0.25, 0.75, 0.9, 0.95, 0.99, 0.999]
  pbLosses=[]; relFreq=[]; normPbLosses=[]
  for i in range(len(MANDATORY_TESTING_QUANTS)+1):
    pbLosses.append([]) #the last one for all quantiles
    relFreq.append([])
    normPbLosses.append([])
    

  for series in series_s:
    filePatterns=outputDir+"e"+str(EPOCH)+"*_"+series+".pickle"
    filePaths= glob.glob(filePatterns)
    filePaths=sorted(filePaths)
    
    print('reading', len(filePaths),'files for',series)
    f=filePaths[0]
    for iif, f in enumerate(filePaths):
      #print('reading data from:',f)
      with open(f, 'rb') as handle:
        seriesByWorker_df = pd.read_pickle(handle)
      #type(seriesByWorker)  
      if iif==0: #file for the first worker contains quants and actuals  
        oneSeries_df=seriesByWorker_df.copy()
      else:
        if seriesByWorker_df.shape[1]>1: #in case we are debugging alignment
          assert np.all(seriesByWorker_df['quants']==oneSeries_df['quants'])
          assert np.all(seriesByWorker_df['actuals']==oneSeries_df['actuals'])
          seriesByWorker_df=seriesByWorker_df.iloc[:,-1]
        oneSeries_df=pd.concat([oneSeries_df,seriesByWorker_df],axis=1)
        
    if USE_MEDIAN_AGG:
      oneSeries_df['aggForec']=np.median(oneSeries_df.iloc[:,2:], axis=1)
    else:
      oneSeries_df['aggForec']=np.mean(oneSeries_df.iloc[:,2:], axis=1)
    
    for iquant, quant in enumerate(MANDATORY_TESTING_QUANTS):
      oneSeries_df.columns
      actuals=oneSeries_df['actuals'].loc[oneSeries_df['quants']==quant]
      forecs=oneSeries_df['aggForec'].loc[oneSeries_df['quants']==quant]
      diff=actuals-forecs
      pbLoss=np.maximum(diff*quant, diff*(quant-1))
      pbLosses[iquant].append(np.mean(pbLoss))
      normPbLosses[iquant].append(np.mean(pbLoss)/np.mean(actuals))
      lowerThanQuant=actuals<=forecs
      rf=np.sum(lowerThanQuant)/len(lowerThanQuant)
      relFreq[iquant].append(rf)
      oneSeries_df['actuals'].loc[oneSeries_df['quants']==quant]=np.nan #so the mandatory quants will not be used for CRPS calculation, below 
    #len(avgActuals[0])==35
  
    #all
    diff=oneSeries_df['actuals']-oneSeries_df['aggForec']
    quant=oneSeries_df['quants']
    pbLoss=np.maximum(diff*quant, diff*(quant-1))
    #assert np.all(pbLoss>0)
    pbLosses[-1].append(np.mean(pbLoss))
    normPbLosses[-1].append(np.mean(pbLoss)/np.mean(oneSeries_df['actuals']))
    
    

  print('Quants, Mean qLosses, normalized qLosses[%], relative freqs:')
  avgPbLosses=[]; relFreqs=[]
  for iquant, quant in enumerate(MANDATORY_TESTING_QUANTS):
    avgPbLosses.append(np.mean(pbLosses[iquant]))
    relFreqs.append(np.mean(relFreq[iquant]))
    print(quant,'\t',f'{avgPbLosses[-1]:10.3}',f'{np.mean(normPbLosses[iquant])*100:15.3}',f'{relFreqs[-1]:15.3}')
  print('Approx CRPS:',f'{2*np.mean(pbLosses[-1]):.4}')    
  print('Avg normalized quantile loss [%]:',f'{100*np.nanmean(normPbLosses[-1]):.4}')

  #plt.scatter(MANDATORY_TESTING_QUANTS, avgPbLosses, c ="blue")
  #plt.title('quantile loss')
  
  plt.close()
  plt.scatter(MANDATORY_TESTING_QUANTS, relFreqs)
  plt.plot((0,1),(0,1))
  plt.xlabel("quantiles")
  plt.ylabel("relative freq")
  plt.show()  
  
  
"""
example testing results
median aggregation
run              epoch,   CRPS,   CRPSnorm
T162,             5        194    0.86
                  6        197    0.86                  
T162_4layers      5        199    0.865
                  6        195    0.855
                  
                  
mean aggregation
T162,             4        205    0.895
                  5        193    0.85
                  6        199    0.88                
                        

MHIRES
run              epoch,   CRPS,   CRPSnorm
T23_BETA(0.3)     6        0.0329  13.09
                  7        0.0329  13.1
                  8        0.0328  13.05

"""
