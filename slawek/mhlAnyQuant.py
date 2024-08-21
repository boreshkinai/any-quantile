'''
Created in Apr-May, 2023
@author: Slawek Smyl

mhl, any quantile
The idea is that a trained (and saved, and restored) system can produce any quantile on demand.
In validation mode (FINAL=False), save forecasts to an ODBC database, in Test mode (FINAL=True) to files.
The program can also save the nets (set SAVE_NETS=True) and then potentially one could write a serving program (not provided) that restores them and produces requested quantiles.
This result is demonstrated in testing mode, where at the end of each epoch we generate randomly NUM_OF_TESTING_QUANTS and save the forecasts (to be later processed by CRPS.py).

internal ver. no.: 162
'''

  
from typing import List, Tuple, Optional  
import random
import datetime as dt
import numpy as np
import pandas as pd
import sys
import pickle
pd.set_option('display.max_rows',50_000)
pd.set_option('display.max_columns', 400)
pd.set_option('display.width',200)
np.set_printoptions(threshold=100_000)

import os
if not 'MKL_NUM_THREADS' in os.environ:
  os.environ['MKL_NUM_THREADS'] = '1'  #conservatively. You should have at least NUM_OF_NETS*MKL_NUM_THREADS cores
  print("MKL_NUM_THREADS not set, setting to 1")
if not 'OMP_NUM_THREADS' in os.environ:  
  os.environ['OMP_NUM_THREADS'] = '1'
  print("OMP_NUM_THREADS not set, setting to 1")
import torch
device="cpu"  #to be overwritten from the command line
from torch import Tensor
torch.set_printoptions(threshold=10_000)

print("pytorch version:"+torch.__version__)
print("numpy version:"+np.version.version)
print("curr dir:",os.getcwd())
print('sys.path:',sys.path)
DEBUG_AUTOGRAD_ANOMALIES=False
torch.autograd.set_detect_anomaly(DEBUG_AUTOGRAD_ANOMALIES)

OUTPUT_DIR="g:/temp/"
DATA_DIR="E:/progs/data/electra4/"

RUN='162 NUM_OF_UPDATES_PER_EPOCH=4000 BETA=0.3 S3[2][4][7][14][28] 150,70 LR=3e-3 {5:/3, 6:/10, 7:/30} batch=2 {4:5}'
SHORT_RUN="162" #inside OUTPUT_DIR for storing variable importance

FINAL=False
if FINAL:
  print('Testing mode')
  RUN='T'+RUN
  SHORT_RUN='T'+SHORT_RUN
  EPOCH_TO_START_SAVING_FORECASTS=4
  
  MANDATORY_TESTING_QUANTS=[0.5, 0.001, 0.01, 0.05, 0.1, 0.25, 0.75, 0.9, 0.95, 0.99, 0.999]
  NUM_OF_TESTING_QUANTS=100 #apart from MANDATORY_TESTING_QUANTS 
  
  SEED_FOR_TESTING=17; stateOfRNG=None
  now=dt.datetime.now()
  hour=int(now.strftime('%H'))
  if hour!=23: #we do not want to sample always the same validation series
    dayOfYear=int(now.strftime('%j'))
    SEED_FOR_TESTING+=dayOfYear #when sampling for validation, all workers need to be synchronized
    
  USE_ODBC=False
else:
  BEGIN_VALIDATION_TS=dt.datetime(2016,1,1)  #validation 2016-2017, test 2018
  print('Validation mode')
  EPOCH_TO_START_SAVING_FORECASTS=3
  RUN='V'+RUN
  SHORT_RUN='V'+SHORT_RUN
  USE_ODBC=True
  import pyodbc
  dbConn = pyodbc.connect(r'DSN=slawek') 
  dbConn.autocommit = False    
  cursor = dbConn.cursor()

  
DEBUG=False
if DEBUG:
  RUN='d'+RUN
  SHORT_RUN='d'+SHORT_RUN
  print('Debug mode')
  EPOCH_TO_START_SAVING_FORECASTS=1
  
USE_SIMPLE_ATTENTION=False
PER_SERIES_MULTIP=30
PASS_INPUT_TO_EVERY_LAYER=True
SAVE_NETS=False
FIRST_EPOCH_TO_SAVE_NETS=4
SAVE_VAR_IMPORTANCE=False
CONTEXT_SIZE_PER_SERIES=3
DISPLAY_SM_WEIGHTS=False
STATE_SIZE=150
H_SIZE=70

BETA=0.3
EPOCH_POW=0.85
NUM_OF_UPDATES_PER_EPOCH=4000
if DEBUG:
  NUM_OF_UPDATES_PER_EPOCH/=25
DILATIONS=[[2],[4],[7],[14],[28]] #moving by 1 day
CELLS_NAME=["S3Cell","S3Cell","S3Cell","S3Cell","S3Cell","S3Cell"]
saveVarImportance=SAVE_VAR_IMPORTANCE and CELLS_NAME[0]=="S3Cell"
INITIAL_LEARNING_RATE=3e-3
NUM_OF_TRAINING_STEPS=50
MAX_NUM_TRIES=1000

STEP_SIZE=24
SEASONALITY_IN_DAYS=7
SEASONALITY=STEP_SIZE*SEASONALITY_IN_DAYS #hourly
INPUT_WINDOW=SEASONALITY 

maxDilations=int(np.max(DILATIONS))
WARMUP_MULTIP=2
TRAINING_WARMUP_STEPS=int(maxDilations*WARMUP_MULTIP)+INPUT_WINDOW
TESTING_WARMUP_STEPS=TRAINING_WARMUP_STEPS



DATES_ENCODE_SIZE=7+31+52 #day of week, day of month, week number. We move by 1 day
DATES_EMBED_SIZE=4
OUTPUT_WINDOW=48 
INPUT_SIZE0=INPUT_WINDOW+DATES_EMBED_SIZE+1+OUTPUT_WINDOW #+log(normalizer)+seasonality

#percentiles below are used in validation mode, and have to correspond to the percentile columns of the forecast table (electra18)
#So, if you change PERCENTILES, you also need to regenerate electra18 table, see in comments at the end 
PERCENTILES=[50] #50 has to be first
for i in range(1,100,7):
  if i!=50:
    PERCENTILES.append(i)
SORTED_PERCENTILES=sorted(PERCENTILES) #only for database operations
SORTED_PERCENTILES_str=[str(x).replace(".","x") for x in SORTED_PERCENTILES] #only for database
QUANTS= [x/100. for x in PERCENTILES]
NUM_OF_QUANTS=len(QUANTS)
QUANTS_a=np.array(QUANTS)  
QUANTS_a=np.expand_dims(QUANTS_a, axis=1)#QUANTS_a.shape
QUANTS_t=torch.tensor(QUANTS)

INITIAL_BATCH_SIZE=2 
BATCH_SIZES={4:5} #at which epoch to change it to what
NUM_OF_EPOCHS=8
LEARNING_RATES={5:INITIAL_LEARNING_RATE/3, 6:INITIAL_LEARNING_RATE/10, 7:INITIAL_LEARNING_RATE/30}  #
OUTPUT_SIZE=OUTPUT_WINDOW+2 #levSm, sSm
NoneT=torch.FloatTensor([-1e38])  #jit does not like Optional etc.
smallNegative=-1e-35
LEV_SM0=-3.5
S_SM0=0.3


#following values are default, typically overwritten from the command line params
workerNumber=1

#################################################################

interactive=True #default, may be overwritten later, used by trouble() function
def trouble(msg):
  if interactive:
    raise Exception(msg)
  else:
    print(msg)
    import pdb; pdb.set_trace()
    
#lev and seasonality smoothing used only at the seasonality warming area
class PerSeriesParams(torch.nn.Module):
  def __init__(self, series):
    super(PerSeriesParams, self).__init__()

    tep=torch.nn.Parameter(torch.tensor(LEV_SM0, device=device))
    self.register_parameter("initLevSm_"+str(series),tep)
    self.initLevSm =tep
    
    tep=torch.nn.Parameter(torch.tensor(S_SM0, device=device))
    self.register_parameter("initSSm_"+str(series),tep)
    self.initSSm=tep
    
    tep=torch.nn.Parameter(torch.ones(NUM_OF_CONTEXT_SERIES*CONTEXT_SIZE_PER_SERIES, device=device))
    self.register_parameter("cModifier_"+str(series),tep)
    self.contextModifier =tep
      


#dat=testDates[2]
#input is a single date
def datesToMetadata(dat): 
  ret=torch.zeros([DATES_ENCODE_SIZE], device=device)
  dayOfWeek=dat.weekday() #Monday is 0 and Sunday is 6
  ret[dayOfWeek]=1
  
  dayOfYear=dat.timetuple().tm_yday
  week=min(51,dayOfYear//7)
  ret[7+week]=1
  
  dayOfMonth=dat.day
  ret[7+52+dayOfMonth-1]=1  #Between 1 and the number of days in the given month
  return ret


#batch=[0,2]
class Batch:
  def __init__(self, batch, isTraining=True):
    self.series=batch
    batchSize=len(batch)
    
    if isTraining:
      warmupSteps=TRAINING_WARMUP_STEPS
      startingIndex=random.choice(startingIndices)  
      startupArea=train_t[startingIndex-warmupSteps*STEP_SIZE:startingIndex, batch]
      numTries=0
      while torch.any(torch.isnan(startupArea)): #so we will ensure we do not start too early for AL
        startingIndex=random.choice(startingIndices)
        startupArea=train_t[startingIndex-warmupSteps*STEP_SIZE:startingIndex, batch]  
        numTries+=1
        if numTries>MAX_NUM_TRIES:
          trouble("numTries>MAX_NUM_TRIES, need to find a better algorithm :-)")
      reallyStartingIndex=startingIndex-warmupSteps*STEP_SIZE+SEASONALITY
      self.dates=trainDates[reallyStartingIndex:]
      
      #for training we also need the big batch/context
      initialSeasonalityArea=train_t[startingIndex-warmupSteps*STEP_SIZE:
                                   reallyStartingIndex, contextSeries] 
      initialSeasonality_t=initialSeasonalityArea/torch.mean(initialSeasonalityArea, dim=0)
      initialSeasonality=[]
      for ir in range(SEASONALITY):
        initialSeasonality.append(initialSeasonality_t[ir].view([NUM_OF_CONTEXT_SERIES,1]))
      self.contextInitialSeasonality=initialSeasonality.copy()
      self.contextY=train_t[reallyStartingIndex:, contextSeries].t()
      
      #this batch
      initialSeasonalityArea=train_t[startingIndex-warmupSteps*STEP_SIZE:
                                   reallyStartingIndex, batch] 
      self.y=train_t[reallyStartingIndex:, batch].t()
      #to be continued below
    else:
      warmupSteps=TESTING_WARMUP_STEPS  #Actually, we will exclude from it initialSeasonalityArea, later in the main loop
      startingIndex=warmupSteps*STEP_SIZE
      firstNotNan=0; 
      if not FINAL:
        #ser=batch[0]
        for ser in batch:
          firstNotNa=np.min(np.where(~np.isnan(test_np[:,ser])))
          firstNotNan=np.maximum(firstNotNan, firstNotNa)
      #for FINAL all testing series have a lot of non-nans at the beginning 
        
      initialSeasonalityArea=test_t[firstNotNan:firstNotNan+SEASONALITY, batch]
      assert not torch.any(torch.isnan(initialSeasonalityArea))
       
      self.y=test_t[firstNotNan+SEASONALITY:, batch].t()
      self.dates=testDates[firstNotNan+SEASONALITY:]
 
    initialSeasonality_t=initialSeasonalityArea/torch.mean(initialSeasonalityArea, dim=0)
    initialSeasonality=[]
    for ir in range(SEASONALITY):
      initialSeasonality.append(initialSeasonality_t[ir].view([batchSize,1]))  
    
    #and continue calculating levels and seasonality in the main
    self.batchSize=batchSize
    self.maseNormalizer=maseDenom[batch]
    self.initialSeasonality=initialSeasonality

    

#Slawek's S3 cel. 
class S3Cell(torch.nn.Module):
  def __init__(self, input_size, h_size, state_size, device=None):
    super(S3Cell, self).__init__()
    #if firstLayer:
    firstCellStateSize=input_size+h_size
    self.lxh=torch.nn.Linear(input_size+2*h_size, 4*firstCellStateSize)  #params of Linear are automatically added to the module params, magically :-)
    self.lxh2=torch.nn.Linear(input_size+2*h_size, 4*state_size)
    self.h_size=h_size
    self.state_size=state_size
    self.out_size=state_size-h_size
    self.device=device
    self.varImportance_t=NoneT
    self.scale = torch.nn.Parameter(torch.ones([1]))
    self.register_parameter("scale", self.scale)

  #jit does not like Optional, so we have to use bool variables and NoneT
  def forward(self, input_t: Tensor, hasDelayedState: bool, hasPrevState : bool,
              prevHState: Tensor=NoneT, delayedHstate : Tensor=NoneT,  
              prevCstate: Tensor=NoneT, delayedCstate: Tensor=NoneT,
              prevHState2: Tensor=NoneT, delayedHstate2 : Tensor=NoneT,  
              prevCstate2: Tensor=NoneT, delayedCstate2: Tensor=NoneT) \
           ->  Tuple[Tensor, Tuple[Tensor, Tensor], Tuple[Tensor, Tensor]]: #outputs: (out, (hState, cState), (hState2, cState2))
    if hasDelayedState:
      xh=torch.cat([input_t, prevHState, delayedHstate], dim=1)
    elif hasPrevState:
      xh=torch.cat([input_t, prevHState, prevHState], dim=1)
    else:
      emptyHState=torch.zeros((input_t.shape[0], 2*self.h_size), device=self.device)
      xh=torch.cat([input_t, emptyHState], dim=1)
      
    gates=self.lxh(xh)
    chunkedGates = torch.chunk(gates,4,dim=1) 

    forgetGate = (chunkedGates[0]+1).sigmoid();
    newState = chunkedGates[1].tanh();
    outGate = chunkedGates[3].sigmoid();
    
    if hasPrevState:
      if hasDelayedState:
        alpha = chunkedGates[2].sigmoid();
        weightedCState=alpha*prevCstate+(1-alpha)*delayedCstate
      else:
        weightedCState=prevCstate
        
      newState = forgetGate*weightedCState + (1-forgetGate)*newState;
        
    wholeOutput = outGate * newState
    
    output_t, hState =torch.split(wholeOutput, [wholeOutput.shape[1]-self.h_size, self.h_size], dim=1)

    self.varImportance_t=torch.exp(output_t*self.scale)
    input_t=input_t*self.varImportance_t
    prevHState=prevHState2
    prevCstate=prevCstate2
    delayedHstate=delayedHstate2
    delayedCstate=delayedCstate2
    
    if hasDelayedState:
      xh=torch.cat([input_t, prevHState, delayedHstate], dim=1)
    elif hasPrevState:
      xh=torch.cat([input_t, prevHState, prevHState], dim=1)
    else:
      emptyHState=torch.zeros((input_t.shape[0], 2*self.h_size), device=self.device)
      xh=torch.cat([input_t, emptyHState], dim=1)
      
    gates=self.lxh2(xh)
    chunkedGates = torch.chunk(gates,4,dim=1)  #==torch.split(gates, [self.state_size, self.state_size, self.state_size, self.state_size], dim=1)

    forgetGate = (chunkedGates[0]+1).sigmoid();
    newState2 = chunkedGates[1].tanh();
    outGate = chunkedGates[3].sigmoid();
    
    if hasPrevState:
      if hasDelayedState:
        alpha = chunkedGates[2].sigmoid();
        weightedCState=alpha*prevCstate+(1-alpha)*delayedCstate
      else:
        weightedCState=prevCstate
        
      newState2 = forgetGate*weightedCState + (1-forgetGate)*newState2;
        
    wholeOutput = outGate * newState2
    
    output_t, hState2 =torch.split(wholeOutput, [self.out_size, self.h_size], dim=1)
    
    return output_t, (hState, newState), (hState2, newState2)
    
    
#Slawek's S2 cell: a kind of mix of GRU and LSTM. Also splitting ouput into h and the "real output"
class S2Cell(torch.nn.Module):
  def __init__(self, input_size, h_size, state_size, device=None):
    super(S2Cell, self).__init__()
    self.lxh=torch.nn.Linear(input_size+2*h_size, 4*state_size)  #params of Linear are automatically added to the module params, magically :-)
    self.h_size=h_size
    self.state_size=state_size
    self.out_size=state_size-h_size
    self.device=device

  #jit does not like Optional, so we have to use bool variables and NoneT
  def forward(self, input_t: Tensor, hasDelayedState: bool, hasPrevState : bool,
               prevHState: Tensor=NoneT,
               delayedHstate : Tensor=NoneT,  
               prevCstate: Tensor=NoneT, 
               delayedCstate: Tensor=NoneT)\
           ->  Tuple[Tensor, Tuple[Tensor, Tensor]]: #outputs: (out, (hState, cState))
    if hasDelayedState:
      xh=torch.cat([input_t, prevHState, delayedHstate], dim=1)
    elif hasPrevState:
      xh=torch.cat([input_t, prevHState, prevHState], dim=1)
    else:
      emptyHState=torch.zeros([input_t.shape[0], 2*self.h_size], device=self.device)
      xh=torch.cat([input_t, emptyHState], dim=1)
      
    gates=self.lxh(xh)
    chunkedGates = torch.chunk(gates,4,dim=1)  

    forgetGate = (chunkedGates[0]+1).sigmoid();
    newState = chunkedGates[1].tanh();
    outGate = chunkedGates[3].sigmoid();
    
    if hasPrevState:
      if hasDelayedState:
        alpha = chunkedGates[2].sigmoid();
        weightedCState=alpha*prevCstate+(1-alpha)*delayedCstate
      else:
        weightedCState=prevCstate
        
      newState = forgetGate*weightedCState + (1-forgetGate)*newState;
        
    wholeOutput = outGate * newState
    
    output_t, hState =torch.split(wholeOutput, [self.out_size, self.h_size], dim=1)
    return output_t, (hState, newState)
   

  
class DilatedRnnStack(torch.nn.Module):
  def resetState(self):
    self.hStates=[]  #first index time, second layers
    self.cStates=[]
    if "S3Cell" in self.cellNames:
      self.hStates2=[]  #first index time, second layers
      self.cStates2=[]
      
  #dilations are like [[1,3,7]] this defines 3 layers + output adaptor layer
  #or [[1,3],[6,12]] - this defines 2 blocks of 2 layers each + output adaptor layer, with a resNet-style shortcut between output of the first block (output of the second layer)
  #and output of the second block (output of 4th layer). 
  def __init__(self, dilations, cellNames, input_size, state_size, output_size, 
               h_size=None, device=None, use_quants=False, sizeOfQuantsEmbedding=1, 
               passInputToEveryLayer=False, useSimpleAttention=False ):
    super(DilatedRnnStack, self).__init__()
    numOfBlocks=len(dilations)
    self.dilations=dilations
    self.cellNames=cellNames
    self.input_size=input_size
    self.h_size=h_size
    self.output_size=output_size
    self.device=device
    self.use_quants=use_quants
    self.passInputToEveryLayer=passInputToEveryLayer
    self.useSimpleAttention=useSimpleAttention
    
    out_sizes=[]
    for cellName in cellNames:
      if cellName!="LSTM":
        out_size=state_size-h_size
      else:
        out_size=state_size
      out_sizes.append(out_size)
    
        
    self.cells = []; 
    self.attentionLayers=[]; self.attentionLayers2=[]
    layer=0; iblock=0; 
    for iblock in range(numOfBlocks):
      for lay in range(len(dilations[iblock])):
        cellName=cellNames[layer]
        if lay==0:
          if iblock==0:
            inputSize=input_size #this may include the quant
          else: #first layer in not the first block
            inputSize=out_sizes[layer]
            if passInputToEveryLayer:
              inputSize+=input_size
            if use_quants:
              inputSize+=sizeOfQuantsEmbedding  #add quant
        else: #not the first layer in a block
          inputSize=out_sizes[layer]
          if use_quants:
            inputSize+=sizeOfQuantsEmbedding  #add quant
        
        if useSimpleAttention:
          print("att layer ",layer,"input size:",inputSize+h_size+inputSize)
          attLayer=torch.nn.Linear(inputSize+h_size+inputSize, self.dilations[iblock][lay]) #Normally inputSize+h_size==state_size, but not for the first layer of S3
          self.attentionLayers.append(attLayer)
          self.add_module("att_{}".format(layer), attLayer)
          if cellName=="S3Cell":
            attLayer=torch.nn.Linear(state_size+inputSize, self.dilations[iblock][lay])
            self.attentionLayers2.append(attLayer)
            self.add_module("att2_{}".format(layer), attLayer)

        if cellName=="S2Cell":
          if interactive:
            cell = S2Cell(inputSize, h_size, state_size, device)
          else:
            cell = torch.jit.script(S2Cell(inputSize, h_size, state_size, device))
        elif cellName=="S3Cell":
          if interactive:
            cell = S3Cell(inputSize, h_size, state_size, device)
          else:
            cell = torch.jit.script(S3Cell(inputSize, h_size, state_size, device))
        else:
          cell = torch.nn.LSTMCell(inputSize, state_size)
        #print("adding","Cell_{}".format(layer))
        self.add_module("Cell_{}".format(layer), cell)
        self.cells.append(cell)
        layer+=1
      
      if use_quants:
        self.adaptor = torch.nn.Linear(out_size+sizeOfQuantsEmbedding, output_size)
      else:
        self.adaptor = torch.nn.Linear(out_size, output_size)
      
    self.numOfBlocks=numOfBlocks  
    self.out_size=out_size
    self.resetState()
    
      
  def forward(self, input_t, quants=None):
    prevBlockOut=torch.zeros([input_t.shape[0], self.out_size], device=self.device)
    self.hStates.append([]) #append for the new t
    self.cStates.append([])
    if "S3Cell" in self.cellNames:
      self.hStates2.append([]) #append for the new t
      self.cStates2.append([])
    t=len(self.hStates)-1
    hasPrevState=t>0
        
    layer=0; layer2=0
    for iblock in range(self.numOfBlocks):
      for lay in range(len(self.dilations[iblock])):
        #print('layer=',layer)
        cellName=self.cellNames[layer]
        if lay==0:
          if iblock==0:
            input=input_t
          else: #first layer in not the first block
            input=prevBlockOut 
            if self.passInputToEveryLayer:
              input=torch.cat([input,input_t], dim=1)
            if self.use_quants:
              input=torch.cat([input,quants], dim=1)
        else: #not the first layer in a block
          input=output_t 
          if self.use_quants:
            input=torch.cat([input,quants], dim=1)
          
        dilation=self.dilations[iblock][lay]  
        ti_1=t-dilation
        hasDelayedState=ti_1>=0
        if hasDelayedState:
          if self.useSimpleAttention:
            attOut_t=self.attentionLayers[layer](torch.cat([self.cStates[t-1][layer],input],dim=1))
            attWeights_t=torch.unsqueeze(torch.softmax(attOut_t,dim=1),dim=2)
            #print('attWeights_t:', attWeights_t, attWeights_t.shape) #[batch,dilation,1]
            statesOfOneLayer=[x[layer] for x in self.hStates[-dilation-1:-1]]
            hStates_t=torch.transpose(torch.stack(statesOfOneLayer), 0, 1) #[batch,dilation,hsize]
            #print('hStates.shape:', hStates_t.shape)
            #if hStates_t.shape[0] !=attWeights_t.shape[0] or hStates_t.shape[1] !=attWeights_t.shape[1]:
            #  trouble("shapes") 
            delayedHstate=torch.sum(hStates_t*attWeights_t,dim=1)
          else:
            delayedHstate=self.hStates[ti_1][layer]
            
          if cellName=="S3Cell":
            if self.useSimpleAttention:
              #print('c.shape',self.cStates2[t-1][layer].shape)
              #print('input.shape',input.shape)
              attOut_t=self.attentionLayers2[layer2](torch.cat([self.cStates2[t-1][layer2],input],dim=1))
              attWeights_t=torch.unsqueeze(torch.softmax(attOut_t,dim=1),dim=2)
              #print('attWeights_t:', attWeights_t, attWeights_t.shape) #[batch,dilation]
              statesOfOneLayer=[x[layer2] for x in self.hStates2[-dilation-1:-1]]
              hStates_t=torch.transpose(torch.stack(statesOfOneLayer), 0, 1) #[batch,dilation,hsize]
              #print('hStates.shape:', hStates_t.shape)
              delayedHstate2=torch.sum(hStates_t*attWeights_t,dim=1)
            else:
              delayedHstate2=self.hStates2[ti_1][layer2]
          
        if cellName=="S2Cell":
          if hasDelayedState:
            output_t, (hState, newState)=self.cells[layer](input, hasDelayedState, hasPrevState,
              prevHState=self.hStates[t-1][layer], delayedHstate=delayedHstate, 
              prevCstate=self.cStates[t-1][layer], delayedCstate=self.cStates[ti_1][layer])
          elif hasPrevState:   
            output_t, (hState, newState)=self.cells[layer](input, hasDelayedState, hasPrevState,
              prevHState=self.hStates[t-1][layer],
              prevCstate=self.cStates[t-1][layer])
          else:
            output_t, (hState, newState)=self.cells[layer](input, False, False) 
        elif cellName=="S3Cell":
          if hasDelayedState:
            output_t, (hState, newState), (hState2, newState2)=self.cells[layer](input, hasDelayedState, hasPrevState,
              prevHState=self.hStates[t-1][layer],    delayedHstate=delayedHstate, 
              prevCstate=self.cStates[t-1][layer],    delayedCstate=self.cStates[ti_1][layer],
              prevHState2=self.hStates2[t-1][layer2], delayedHstate2=delayedHstate2, 
              prevCstate2=self.cStates2[t-1][layer2], delayedCstate2=self.cStates2[ti_1][layer2])
          elif hasPrevState:   
            output_t, (hState, newState), (hState2, newState2)=self.cells[layer](input, hasDelayedState, hasPrevState,
              prevHState=self.hStates[t-1][layer],prevCstate=self.cStates[t-1][layer],
              prevHState2=self.hStates2[t-1][layer2],prevCstate2=self.cStates2[t-1][layer2])
          else:
            output_t, (hState, newState), (hState2, newState2)=self.cells[layer](input, False, False) 
        else: #LSTM 
          if hasDelayedState:
            hState, newState=self.cells[layer](input, (self.hStates[ti_1][layer], self.cStates[ti_1][layer]))
          elif hasPrevState:
            hState, newState=self.cells[layer](input, (self.hStates[t-1][layer], self.cStates[t-1][layer]))
          else:
            hState, newState=self.cells[layer](input) 
          output_t=hState
            
        self.hStates[t].append(hState)
        self.cStates[t].append(newState)
        if cellName=="S3Cell":
          self.hStates2[t].append(hState2)
          self.cStates2[t].append(newState2)
        
        layer+=1
        if cellName=="S3Cell":
          layer2+=1
      prevBlockOut=output_t+prevBlockOut
      
    if self.use_quants:
      prevBlockOut=torch.cat([prevBlockOut,quants], dim=1)
    output_t = self.adaptor(prevBlockOut)
    return output_t                    

                
#pinball
#actuals_t are the original actuals
#quant is vector of the length=batchSize
#output is scalar (actually vector if the size=batchSize)
def trainingLossFunc(forec_t, actuals_t, anchorLevel, quants_t):
  if forec_t.shape[0] != actuals_t.shape[0]:
    trouble("forec_t.shape[0] != actuals_t.shape[0]")

  if forec_t.shape[1] != OUTPUT_WINDOW:
    trouble("forec_t.shape[1] != OUTPUT_WINDOW")
  
  if actuals_t.shape[1] != OUTPUT_WINDOW:
    trouble("actuals_t.shape[1] != OUTPUT_WINDOW")
   
  nans=torch.isnan(actuals_t).detach() | (actuals_t<=0).detach()
  notNans=(~nans).float()
  numOfNotNans=notNans.sum(dim=1) #vector of batchSize

  if torch.any(nans):
    actuals_t[nans]=1 #actuals have been cloned outside of this function

  #we do it here, becasue Pytorch is alergic to any opearation including nans, even if removed from the graph later
  #so we first patch nans and execute normalization and squashing and then remove results involving nans
  actualsS_t=actuals_t/anchorLevel

  diff=actualsS_t-forec_t #normalized and squashed
  rs=torch.max(diff*quants_t, diff*(quants_t-1))
  rs[nans] = 0

  if torch.any(numOfNotNans==0):
    for ib in range(len(numOfNotNans)):
      if numOfNotNans[ib]==0:
        numOfNotNans[ib]+=1
  ret=rs.sum(dim=1)/numOfNotNans #numOfNotNans is vector
  return ret
            
            
# RMSE, bias, MASE, MAPE, pinball loss,  % of exceedance. Return dimenions =[5+len(QUANTS)]
# operating on numpy arrays, not tensors
#maseNormalizer=ppBatch.maseNormalizer
def validationLossFunc(forec, actuals, maseNormalizer): 
  if np.isnan(forec.data).sum()>0:
    trouble("NaNs in forecast")
    
  if forec.shape[0] != actuals.shape[0]:
    trouble("forec.shape[0] != actuals.shape[0]")
  
  if forec.shape[1] != OUTPUT_WINDOW:
    trouble("forec.shape[1] != OUTPUT_WINDOW")
  
  if actuals.shape[1] != OUTPUT_WINDOW:  #but they may be all NANs
    trouble("actuals.shape[1] != OUTPUT_WINDOW")
    
  ret=np.zeros([5+len(QUANTS)], dtype=np.float32)+np.nan
  
  #center
  diff=forec[0]-actuals[0] #diff.shape
  rmse=np.sqrt(np.nanmean(diff*diff))
  mase=np.nanmean(abs(diff))/maseNormalizer
  mape=np.nanmean(abs(diff/actuals[0]))
  bias=np.nanmean(diff/actuals[0])
  
  ret[0]=rmse
  ret[1]=bias
  ret[2]=mase
  ret[3]=mape
  
  #exceedance and pbLoss
  diff=actuals-forec #diff.shape; QUANTS_a.shape
  rs=np.maximum(diff*QUANTS_a, diff*(QUANTS_a-1))
  pbLoss=np.nanmean(rs/actuals)
  ret[4]=pbLoss
  
  iq=0
  for iq in range(len(QUANTS)):
    quant=QUANTS[iq]
    
    if quant>=0.5:
      xceeded=diff[iq]>0
    else:
      xceeded=diff[iq]<0
        
    exceeded=np.nanmean(xceeded) 
    ret[iq+5]=exceeded
      
  return ret
            



if __name__ == '__main__' or __name__ == 'builtins':
  print(RUN)
  if len(sys.argv)==5:
    print("assuming running from within Eclipse and assuming default params")
    interactive=True
  elif len(sys.argv)==3:
    workerNumber=int(sys.argv[1])
    print('workerNumber:',workerNumber)
    gpuNumber=int(sys.argv[2])
    device='cuda:'+str(gpuNumber)
    print('using',device)
    interactive=False
  elif len(sys.argv)==2:
    workerNumber=int(sys.argv[1])
    print('workerNumber:',workerNumber)
    interactive=False
  elif len(sys.argv)==1:
    print("assuming default params")
    interactive=False
  else:
    print("you need to specify workerNumber, [gpuNumber] ")
    exit(-1)
  
    
  outputDir=OUTPUT_DIR+SHORT_RUN+"/"
  saveNetsDir=outputDir+"nets/"
  if workerNumber==1:
    dirExists = os.path.exists(outputDir)
    if not dirExists:
      os.makedirs(outputDir)
    
    dirExists = os.path.exists(saveNetsDir)
    if not dirExists:
      os.makedirs(saveNetsDir)
  
  DATA_PATH=DATA_DIR+'MHL_train_full_date.csv'
  trainDates_df=pd.read_csv(DATA_PATH, header=None)
  trainDates=pd.to_datetime(trainDates_df.iloc[:,0])
  for i in range(1,len(trainDates)):
    diff=trainDates[i]-trainDates[i-1]
    if diff!=dt.timedelta(hours=1):
      print("WTF")
      break
  
  DATA_PATH=DATA_DIR+'MHL_train_full.csv'
  train0_df=pd.read_csv(DATA_PATH, header=None)
  train_df=train0_df.copy()
  train_df.shape #(17544, 35)
  train_df.head(3) 
  assert len(trainDates)==len(train_df)
  sum(np.isnan(train_df)) #595
  
  if FINAL:
    DATA_PATH=DATA_DIR+'MHL_test_date.csv'
    testDates_df=pd.read_csv(DATA_PATH, header=None)
    testDates=pd.to_datetime(testDates_df.iloc[:,0])
    for i in range(1,len(testDates)):
      diff=testDates[i]-testDates[i-1]
      if diff!=dt.timedelta(hours=1):
        print("WTF")
        break
    
    DATA_PATH=DATA_DIR+'MHL_test.csv'
    test_df=pd.read_csv(DATA_PATH, header=None)
    test_df.shape #(8760, 35)
    #test_df.head(3) 
    assert len(testDates)==len(test_df)
    #np.where(np.isnan(test_df))
    #test_df.iloc[:,6]  #nans
    sum(np.isnan(test_df)) #595
    len(test_df)
  else:
    trainDates1=trainDates[trainDates<BEGIN_VALIDATION_TS]
    testDates1=trainDates[trainDates>=BEGIN_VALIDATION_TS]  
    #print('trainDates: min',min(trainDates1),',max', max(trainDates1))
    #print('validDates: min',min(testDates1),',max', max(testDates1))
    train1_df=train_df[trainDates<BEGIN_VALIDATION_TS]
    train1_df.shape
    test1_df=train_df[trainDates>=BEGIN_VALIDATION_TS]
    test1_df.shape
    
    train_df=train1_df
    train_df[:9]
    trainDates=trainDates1
    test_df=test1_df
    testDates=testDates1

  #add to test the warming up area
  a_df=train_df.iloc[-TESTING_WARMUP_STEPS*STEP_SIZE:,:]
  #test1_df=a_df.append(test_df,ignore_index=True)
  test1_df=pd.concat([a_df, test_df],ignore_index=True)
  test1_df.shape
  
  trainDates=trainDates.tolist()
  assert len(train_df)==len(trainDates)
  
  testDates=testDates.tolist()
  testDates1=trainDates[-TESTING_WARMUP_STEPS*STEP_SIZE:]+testDates
  len(testDates1)
  assert len(test1_df)==len(testDates1)
  testDates=testDates1
  
  print('trainDates: min',min(trainDates),',max', max(trainDates))
  print('validDates (with warmup): min',min(testDates),',max', max(testDates))
  
  test_np=test1_df.to_numpy(np.float32)
  train_np=train_df.to_numpy(np.float32)
  sum(sum(np.isnan(train_np))) #547,952
  train_np.shape[0]*train_np.shape[1] #3,067,680 -> 15%
  
  
  print("num of nans in train:")
  firstNotNans_d={}
  contextSeries=[]; emptyTrainSeries=[]
  icol=0
  for icol in range(train_np.shape[1]):
    numOfNans=sum(np.isnan(train_np[:,icol]))
    if numOfNans>0:
      if numOfNans< train_np.shape[0]:
        firstNotNan=np.min(np.where(~np.isnan(train_np[:,icol])))
        lastNan=np.max(np.where(np.isnan(train_np[:,icol])))
        print(icol,"numOfNans:",numOfNans,"firstNotNan:",firstNotNan,"lastNan:",lastNan)
      else:
        print(icol,"all NaNs")  
        emptyTrainSeries.append(icol)
    else:
      contextSeries.append(icol)  
  print("contextSeries:",contextSeries)
  
  
  train_t=torch.tensor(train_np, device=device)
  train_t.shape #torch.Size([17544, 35])
  np.nanmin(train_np) #174.0
  
  test_t=torch.tensor(test_np, device=device)
  test_t.shape #torch.Size([17544, 35])
  
  assert len(testDates)==test_t.shape[0]
  
  emptyTestSeries=[]
  test_np.shape
  print("num of nans in test:")
  icol=0
  for icol in range(test_np.shape[1]):
    numOfNans=sum(np.isnan(test_np[:,icol]))
    if numOfNans>0:
      if numOfNans< test_np.shape[0]:
        firstNan=np.min(np.where(np.isnan(test_np[:,icol])))
        lastNan=np.max(np.where(np.isnan(test_np[:,icol])))
        print(icol,"numOfNans:",numOfNans,"firstNan:",firstNan,"lastNan:",lastNan) 
      else:
        print(icol,"all NaNs in test")  
        emptyTestSeries.append(icol)
  
  maseDenom_l=[]
  istep=0
  train0_np=train0_df.to_numpy(np.float32)
  for istep in range(len(train0_np)-SEASONALITY):
    diff=train0_np[istep+SEASONALITY,]-train0_np[istep,]
    maseDenom_l.append(np.abs(diff))
  len(maseDenom_l)
  maseDenom_a=np.array(maseDenom_l)
  maseDenom_a.shape
  maseDenom=np.nanmean(maseDenom_a, axis=0)
  maseDenom.shape
  min(maseDenom)
  
  """ execute once
  if USE_ODBC:  
    MASE_INSERT_QUERY = "insert into electraMase(series, denom) values(?,?)"
  elif USE_POSTGRESS:
    MASE_INSERT_QUERY = "insert into electraMase(series, denom) values(%,%)"
    
  for iseries in range(len(maseDenom)):
    series=str(iseries)  
    val=float(maseDenom[iseries])
    if USE_ANSI_DRIVER:
      theVals=[bytearray(series, encoding='utf-8'), val] 
    else:
      theVals=[series, val] 
    cursor.execute(MASE_INSERT_QUERY,theVals)
  dbConn.commit()  
  """
  
  startingRange=range(TRAINING_WARMUP_STEPS*STEP_SIZE, 
                          len(train_t)-OUTPUT_WINDOW-NUM_OF_TRAINING_STEPS*STEP_SIZE,STEP_SIZE)
  startingIndices=list(startingRange)
  len(startingIndices)
  
  series_list=list(range(train_t.shape[1]))
  NUM_OF_CONTEXT_SERIES=len(contextSeries)
  print('NUM_OF_CONTEXT_SERIES',NUM_OF_CONTEXT_SERIES)
  numSeries=len(series_list)
  
  if USE_ODBC:        
    INSERT_QUERY = "insert into electra18(dateTimeOfPrediction, workerNo, epoch, forecOriginDate, series "
    for ih in range(OUTPUT_WINDOW):
      INSERT_QUERY+="\n, actual"+str(ih+1);
      for q in SORTED_PERCENTILES_str:
        INSERT_QUERY+= ", predQ"+q+"_"+str(ih+1)
    INSERT_QUERY+=")\n"
    
    MODEL_INSERT_QUERY = "insert into electra4Models(run, workerNo, dateTimeOfPrediction) \
      values(?,?,?)"
    
    INSERT_QUERY+="values (?,?,?,?,?"
    for ih in range(OUTPUT_WINDOW):
      INSERT_QUERY+=",?";
      for _ in SORTED_PERCENTILES_str:
        INSERT_QUERY+= ",?"
    INSERT_QUERY+=")"    
 


  now=dt.datetime.now()
  if USE_ODBC:     
    theVals=(RUN, workerNumber, now)  
    cursor.execute(MODEL_INSERT_QUERY,theVals)  
    
  perSeriesParams_d={}; perSeriesTrainers=[]
  for series in series_list:
    perSerPars=PerSeriesParams(series)
    perSeriesParams_d[series]=perSerPars
    perSerTrainer=torch.optim.Adam(perSerPars.parameters(), lr=INITIAL_LEARNING_RATE*PER_SERIES_MULTIP)
    perSeriesTrainers.append(perSerTrainer)
    
  embed=torch.nn.Linear(DATES_ENCODE_SIZE, DATES_EMBED_SIZE)
  embed=embed.to(device)
  #           dilations, cellNames, input_size, state_size, output_size, h_size=None, device=None, use_quants=False, passInputToEveryLayer
  rnn=DilatedRnnStack(DILATIONS, CELLS_NAME, INPUT_SIZE0+CONTEXT_SIZE_PER_SERIES*NUM_OF_CONTEXT_SERIES+1, #+1-> quant
    STATE_SIZE, OUTPUT_SIZE, H_SIZE, device=device, use_quants=True, passInputToEveryLayer=PASS_INPUT_TO_EVERY_LAYER,
    useSimpleAttention=USE_SIMPLE_ATTENTION)
  rnn=rnn.to(device) #we did not move every object to device
  
  contextRnn=DilatedRnnStack(DILATIONS, CELLS_NAME, INPUT_SIZE0, 
    STATE_SIZE, 2+CONTEXT_SIZE_PER_SERIES, H_SIZE, device=device, 
    passInputToEveryLayer=PASS_INPUT_TO_EVERY_LAYER,
    useSimpleAttention=USE_SIMPLE_ATTENTION)
  contextRnn=contextRnn.to(device) #we did not move every object to device
  
  allParams = list(embed.parameters()) + list(rnn.parameters()) + list(contextRnn.parameters())
  trainer=torch.optim.Adam(allParams, lr=INITIAL_LEARNING_RATE)
  if DISPLAY_SM_WEIGHTS:
    print('levSm w:', rnn.adaptor.weight[0].detach().cpu().numpy())
    print('levSm b:', rnn.adaptor.bias[0].detach().cpu().numpy())

    print('sSm w:', rnn.adaptor.weight[1].detach().cpu().numpy())
    print('sSm b:', rnn.adaptor.bias[1].detach().cpu().numpy())

  
  learningRate=INITIAL_LEARNING_RATE
  batchSize=INITIAL_BATCH_SIZE
  iEpoch=0; prevNumOfRepeats=0
  print('num of epochs:',NUM_OF_EPOCHS)
  varNames=[]
  
  contextValidBatch=Batch(contextSeries, isTraining=False) 
  
  for iEpoch in range(NUM_OF_EPOCHS):  #-<<-----------epoch------------
    nowe=dt.datetime.now()
    print(nowe.strftime("%Y-%m-%d %H:%M:%S"),  'starting epoch:',iEpoch)
    
    varImportancePath=outputDir+"e"+str(iEpoch)+ "w"+str(workerNumber)+".csv"
    varImportance_df=None
    
    if iEpoch in BATCH_SIZES:
      batchSize=BATCH_SIZES[iEpoch]
      print ("changing batch size to:",batchSize)
    
    if iEpoch in LEARNING_RATES:
      learningRate=LEARNING_RATES[iEpoch]
      for param_group in trainer.param_groups:
          param_group['lr']=learningRate     
      for series in series_list: 
        for param_group in perSeriesTrainers[series].param_groups:
          param_group['lr']=learningRate*PER_SERIES_MULTIP
      print('changing LR to:', f'{learningRate:.2}' )   
      
    epochTrainingErrors=[];
    epochValidationErrors=[];
    
    numOfEpochLoops=int(np.power(NUM_OF_UPDATES_PER_EPOCH*batchSize/numSeries,EPOCH_POW))
    if numOfEpochLoops<1:
      numOfEpochLoops=1  
    if prevNumOfRepeats!=numOfEpochLoops and numOfEpochLoops>1:
      print ("repeating epoch sub-epoch "+str(numOfEpochLoops)+" times")
    prevNumOfRepeats=numOfEpochLoops       

    numOfUpdatesSoFar=0; isubEpoch=0 
    while isubEpoch<numOfEpochLoops:
      isValidation=iEpoch>=EPOCH_TO_START_SAVING_FORECASTS and isubEpoch==numOfEpochLoops-1 #for the last subepoch first training then testing 
      
      batches=[]; batch=[]
      random.shuffle(series_list)
      for series in series_list:
        if series not in emptyTrainSeries:
          batch.append(series)
          if len(batch) >= batchSize:
            batches.append(batch);
            batch=[]
      if len(batch)>0:
        batches.append(batch)
      random.shuffle(batches)
      
      
      batch=batches[0]
      for batch in batches:
        if DEBUG or DEBUG_AUTOGRAD_ANOMALIES:
          print(batch)
        ppBatch=Batch(batch)
        #ppBatch.contextInitialSeasonality
        #ppBatch.contextY.shape  torch.Size([34, 13440])
        
        quants=np.random.beta(BETA, BETA, size=ppBatch.batchSize)
        quants_t=torch.tensor(quants, dtype=torch.float32, device=device).view([ppBatch.batchSize,1])
        
        rnn.resetState(); contextRnn.resetState()
        trainingErrors=[];  

        #start levels and extend seasonality with a static smoothiong coefs
        ii=0
        levels=[]; seasonality=ppBatch.initialSeasonality.copy()
        levSm0=torch.stack([perSeriesParams_d[x].initLevSm for x in batch])
        levSm=torch.sigmoid(levSm0).view([ppBatch.batchSize,1])
        sSm0=torch.stack([perSeriesParams_d[x].initSSm for x in batch])
        sSm=torch.sigmoid(sSm0).view([ppBatch.batchSize,1])
        contextModifier_t=torch.stack([perSeriesParams_d[x].contextModifier for x in batch])
        y_l=[]
        for ii in range(INPUT_WINDOW):
          newY=ppBatch.y[:,ii].view([ppBatch.batchSize,1])
          assert torch.isnan(newY).sum()==0
          y_l.append(newY)    

          if ii==0:
            newLevel=newY/seasonality[0]
          else:
            newLevel=levSm*newY/seasonality[ii]+(1-levSm)*levels[ii-1]
          levels.append(newLevel)
          
          newSeason=sSm*newY/levels[ii]+(1-sSm)*seasonality[ii]
          seasonality.append(newSeason)
          
        #context
        contextLevels=[]; 
        contextSeasonality=ppBatch.contextInitialSeasonality.copy()
        contextLevSm0=torch.stack([perSeriesParams_d[x].initLevSm for x in contextSeries])
        contextLevSm=torch.sigmoid(contextLevSm0).view([NUM_OF_CONTEXT_SERIES,1])
        contextSSm0=torch.stack([perSeriesParams_d[x].initSSm for x in contextSeries])
        contextSSm=torch.sigmoid(contextSSm0).view([NUM_OF_CONTEXT_SERIES,1])
        for ii in range(INPUT_WINDOW):
          newY=ppBatch.contextY[:,ii].view([NUM_OF_CONTEXT_SERIES,1])
          assert torch.isnan(newY).sum()==0
          if ii==0:
            newLevel=newY/contextSeasonality[0]
          else:
            newLevel=contextLevSm*newY/contextSeasonality[ii]+(1-contextLevSm)*contextLevels[ii-1]
          contextLevels.append(newLevel)
          
          newSeason=contextSSm*newY/contextLevels[ii]+(1-contextSSm)*contextSeasonality[ii]
          contextSeasonality.append(newSeason)
        
        remainingWarmupSteps=TRAINING_WARMUP_STEPS*STEP_SIZE-SEASONALITY-INPUT_WINDOW #we do not count here the first SEASONALITY done in Batch()
        istep=INPUT_WINDOW-1 #index of last level
        for istep in range(INPUT_WINDOW-1, 
          INPUT_WINDOW-1+remainingWarmupSteps+NUM_OF_TRAINING_STEPS*STEP_SIZE, STEP_SIZE):
          
          isTraining = istep>=INPUT_WINDOW-1+remainingWarmupSteps 
          dat=ppBatch.dates[istep]
          if istep>=INPUT_WINDOW:
            ii=istep+1-STEP_SIZE
            for ii in range(istep+1-STEP_SIZE, istep+1):
              newY=ppBatch.y[:,ii].view([ppBatch.batchSize,1])
              if torch.isnan(newY).sum()>0:
                newY=newY.clone()
                for ib in range(ppBatch.batchSize):
                  if torch.isnan(newY[ib]):
                    assert ii-SEASONALITY>=0
                    newY[ib]=ppBatch.y[ib,ii-SEASONALITY]
              assert torch.isnan(newY).sum()==0
              y_l.append(newY)    
                  
              newLevel=levSm*newY/seasonality[ii]+(1-levSm)*levels[ii-1]
              levels.append(newLevel)
              
              newSeason=sSm*newY/levels[ii]+(1-sSm)*seasonality[ii]
              seasonality.append(newSeason)
              
              #context
              newY=ppBatch.contextY[:,ii].view([NUM_OF_CONTEXT_SERIES,1])
              assert torch.isnan(newY).sum()==0
              
              newLevel=contextLevSm*newY/contextSeasonality[ii]+(1-contextLevSm)*contextLevels[ii-1]
              contextLevels.append(newLevel)
          
              newSeason=contextSSm*newY/contextLevels[ii]+(1-contextSSm)*contextSeasonality[ii]
              contextSeasonality.append(newSeason)
            
          datesMetadata=datesToMetadata(dat)
          embeddedDates0_t=embed(datesMetadata)
          
          #context
          embeddedDates_t=embeddedDates0_t.expand(NUM_OF_CONTEXT_SERIES,DATES_EMBED_SIZE)
          x0_t=ppBatch.contextY[:,istep-INPUT_WINDOW+1:istep+1] #x0_t.shape
          anchorLevel=torch.mean(x0_t, dim=1).view([NUM_OF_CONTEXT_SERIES,1])
          inputSeasonality_t=torch.cat(contextSeasonality[istep-INPUT_WINDOW+1:istep+1],dim=1)  #inputSeasonality_t.shape
          x_t=torch.log(x0_t/(inputSeasonality_t*anchorLevel))
          outputSeasonality=torch.cat(contextSeasonality[istep+1:istep+OUTPUT_WINDOW+1],dim=1) #outputSeasonality.shape
          input_t=torch.cat([x_t, torch.log10(anchorLevel), embeddedDates_t, outputSeasonality-1],  dim=1)  
          
          forec0_t=contextRnn(input_t)
          #print(forec0_t, forec0_t.shape)
          if torch.isnan(forec0_t).sum()>0:
            print(forec0_t)
            trouble("nans in forecast0")
            
          if len(forec0_t.shape)==1:
            forec0_t=torch.unsqueeze(forec0_t,dim=0)
          #forec0_t.shape
          contextLevSm=torch.sigmoid(forec0_t[:,0]+contextLevSm0).view([NUM_OF_CONTEXT_SERIES,1])
          contextSSm=torch.sigmoid(forec0_t[:,1]+contextSSm0).view([NUM_OF_CONTEXT_SERIES,1])
          context_t=torch.flatten(forec0_t[:,2:])
          context_t=context_t.expand(ppBatch.batchSize,context_t.shape[0])
          context_t=context_t*contextModifier_t
          
          #back to the batch
          embeddedDates_t=embeddedDates0_t.expand(ppBatch.batchSize,DATES_EMBED_SIZE)
          x0_t=torch.cat(y_l[istep-INPUT_WINDOW+1:istep+1], dim=1)
          anchorLevel=torch.mean(x0_t, dim=1).view([ppBatch.batchSize,1])
          inputSeasonality_t=torch.cat(seasonality[istep-INPUT_WINDOW+1:istep+1],dim=1)  #inputSeasonality_t.shape
          x_t=torch.log(x0_t/(inputSeasonality_t*anchorLevel))
          
          outputSeasonality=torch.cat(seasonality[istep+1:istep+OUTPUT_WINDOW+1],dim=1) #outputSeasonality.shape
                
          input_t=torch.cat([x_t, torch.log10(anchorLevel), embeddedDates_t, outputSeasonality-1, 
                             context_t, quants_t],  dim=1)  
          
          forec0_t=rnn(input_t, quants_t)
          #print(forec0_t, forec0_t.shape)
          if torch.isnan(forec0_t).sum()>0:
            print(forec0_t)
            trouble("nans in forecast")
            
          if len(forec0_t.shape)==1:
            forec0_t=torch.unsqueeze(forec0_t,dim=0)
          #forec0_t.shape
          levSm=torch.sigmoid(forec0_t[:,0]+levSm0).view([ppBatch.batchSize,1])
          sSm=torch.sigmoid(forec0_t[:,1]+sSm0).view([ppBatch.batchSize,1])
          
          if isTraining:  
            actuals_t=ppBatch.y[:,istep+1:istep+OUTPUT_WINDOW+1].clone()
            forec_t=torch.exp(forec0_t[:,2:])*outputSeasonality #outputSeasonality.shape
            loss_t=trainingLossFunc(forec_t, actuals_t, anchorLevel, quants_t)
            avgLoss=torch.nanmean(loss_t)
            if not torch.isnan(avgLoss):
              trainingErrors.append(torch.unsqueeze(avgLoss,dim=0))
               
        #batch level     
        if len(trainingErrors)>0:
          trainer.zero_grad()  
          for series in ppBatch.series:
            perSeriesTrainers[series].zero_grad()
            
          avgTrainLoss_t=torch.mean(torch.cat(trainingErrors))    
          assert not torch.isnan(avgTrainLoss_t)  
          avgTrainLoss_t.backward()          
          trainer.step()    
          
          for series in ppBatch.series:
            perSeriesTrainers[series].step()  #here series is integer
                
          epochTrainingErrors.append(avgTrainLoss_t.detach().cpu().numpy())
          trainingErrors=[]
        #end of batch    
        
              
        if isValidation:
          with torch.no_grad():
            if FINAL:
              if iEpoch==EPOCH_TO_START_SAVING_FORECASTS:
                np.random.seed(SEED_FOR_TESTING) #pandas reportedly use np.random
                stateOfRNG=np.random.get_state()
              #else:
              #  np.random.set_state(stateOfRNG)
                        
          
            #series=batch[0]
            for series in batch:   #we are doing one series at a time, but for several quants
              if FINAL:
                forecsSavingPath=outputDir+"e"+str(iEpoch)+"w"+str(workerNumber)+"_"+str(series)+".pickle"
                oneSeriesForecs=[]
                np.random.set_state(stateOfRNG)
                quants=MANDATORY_TESTING_QUANTS+list(np.random.uniform(size=NUM_OF_TESTING_QUANTS)) #so we generate random quants, but all workers are synchronized (generate the same quants)
                stateOfRNG=np.random.get_state() 
                np.random.seed(None) #randomize state again for all other endeavours
              else:
                quants=QUANTS
              quants_t=torch.tensor(quants, dtype=torch.float32, device=device).view([len(quants),1])
              
              ppBatch=Batch([series], False) 
              rnn.resetState(); contextRnn.resetState()
              levels=[]; 
              seasonality=[ppBatch.initialSeasonality[ix].expand([len(quants),1]) 
                            for ix in range(len(ppBatch.initialSeasonality))]
              levSm0=perSeriesParams_d[series].initLevSm.expand(len(quants),1)
              levSm=torch.sigmoid(levSm0)
              sSm0=perSeriesParams_d[series].initSSm.expand(len(quants),1)
              sSm=torch.sigmoid(sSm0)
              contextModifier_t=perSeriesParams_d[series].contextModifier.expand(len(quants),len(perSeriesParams_d[series].contextModifier))
              ii=0
              for ii in range(INPUT_WINDOW):
                newY=ppBatch.y[:,ii]
                if ii==0:
                  newLevel=newY/seasonality[0]
                else:
                  newLevel=levSm*newY/seasonality[ii]+(1-levSm)*levels[ii-1]
                levels.append(newLevel)
                
                newSeason=sSm*newY/levels[ii]+(1-sSm)*seasonality[ii]
                seasonality.append(newSeason)
                
              #context
              contextLevels=[]; 
              contextSeasonality=contextValidBatch.initialSeasonality.copy()
              contextLevSm0=torch.stack([perSeriesParams_d[x].initLevSm for x in contextSeries])
              contextLevSm=torch.sigmoid(contextLevSm0).view([NUM_OF_CONTEXT_SERIES,1])
              contextSSm0=torch.stack([perSeriesParams_d[x].initSSm for x in contextSeries])
              contextSSm=torch.sigmoid(contextSSm0).view([NUM_OF_CONTEXT_SERIES,1])          
              for ii in range(INPUT_WINDOW):
                newY=contextValidBatch.y[:,ii].view([NUM_OF_CONTEXT_SERIES,1])
                assert torch.isnan(newY).sum()==0
                if ii==0:
                  newLevel=newY/contextSeasonality[0]
                else:
                  newLevel=contextLevSm*newY/contextSeasonality[ii]+(1-contextLevSm)*contextLevels[ii-1]
                contextLevels.append(newLevel)
                
                newSeason=contextSSm*newY/contextLevels[ii]+(1-contextSSm)*contextSeasonality[ii]
                contextSeasonality.append(newSeason)
                
          
              remainingWarmupSteps=TESTING_WARMUP_STEPS*STEP_SIZE-SEASONALITY-INPUT_WINDOW #we do not count here the first SEASONALITY done in Batch()    
            
              istep=INPUT_WINDOW-1 #index of last level
              for istep in range(INPUT_WINDOW-1, ppBatch.y.shape[1]-OUTPUT_WINDOW, STEP_SIZE):
                warmupFinished = istep>=INPUT_WINDOW-1+remainingWarmupSteps  
                dat=ppBatch.dates[istep]
                if istep>=INPUT_WINDOW:
                  for ii in range(istep+1-STEP_SIZE, istep+1):
                    newY=ppBatch.y[:,ii]
                    if torch.isnan(newY).sum()>0:
                      assert ii-SEASONALITY>=0
                      newY=ppBatch.y[0,ii-SEASONALITY]
                      ppBatch.y[0,ii]=newY #patching input, not output. No gradient needed, so we can overwrite
                    assert torch.isnan(newY).sum()==0

                          
                    newLevel=levSm*newY/seasonality[ii]+(1-levSm)*levels[ii-1]
                    levels.append(newLevel)
                    
                    newSeason=sSm*newY/levels[ii]+(1-sSm)*seasonality[ii]
                    seasonality.append(newSeason)
                    
                    #context
                    newY=contextValidBatch.y[:,ii].view([NUM_OF_CONTEXT_SERIES,1])
                    assert torch.isnan(newY).sum()==0
                    
                    newLevel=contextLevSm*newY/contextSeasonality[ii]+(1-contextLevSm)*contextLevels[ii-1]
                    contextLevels.append(newLevel)
                
                    newSeason=contextSSm*newY/contextLevels[ii]+(1-contextSSm)*contextSeasonality[ii]
                    contextSeasonality.append(newSeason)
                    
                datesMetadata=datesToMetadata(dat)
                embeddedDates0_t=embed(datesMetadata)
                
                #context
                embeddedDates_t=embeddedDates0_t.expand(NUM_OF_CONTEXT_SERIES,DATES_EMBED_SIZE)
                x0_t=contextValidBatch.y[:,istep-INPUT_WINDOW+1:istep+1] #x0_t.shape
                anchorLevel=torch.mean(x0_t, dim=1).view([NUM_OF_CONTEXT_SERIES,1])
                inputSeasonality_t=torch.cat(contextSeasonality[istep-INPUT_WINDOW+1:istep+1],dim=1)  #inputSeasonality_t.shape
                x_t=torch.log(x0_t/(inputSeasonality_t*anchorLevel))
                outputSeasonality=torch.cat(contextSeasonality[istep+1:istep+OUTPUT_WINDOW+1],dim=1) #outputSeasonality.shape
                input_t=torch.cat([x_t, torch.log10(anchorLevel), embeddedDates_t, outputSeasonality-1],  dim=1)  
                
                forec0_t=contextRnn(input_t)
                if torch.isnan(forec0_t).sum()>0:
                  print(forec0_t)
                  trouble("nans in valid forecast0")
                  
                if len(forec0_t.shape)==1:
                  forec0_t=torch.unsqueeze(forec0_t,dim=0)
                #forec0_t.shape
                contextLevSm=torch.sigmoid(forec0_t[:,0]+contextLevSm0).view([NUM_OF_CONTEXT_SERIES,1])
                contextSSm=torch.sigmoid(forec0_t[:,1]+contextSSm0).view([NUM_OF_CONTEXT_SERIES,1])
                context_t=torch.flatten(forec0_t[:,2:])
                context_t=context_t*contextModifier_t  #context_t.shape
                context_t=context_t.expand(len(quants),context_t.shape[1])  #context_t.shape
                
                #back to the batch
                embeddedDates_t=embeddedDates0_t.expand([len(quants),-1])    #embeddedDates_t.shape
                x0_t=ppBatch.y[:,istep-INPUT_WINDOW+1:istep+1] #x0_t.shape
                anchorLevel=torch.mean(x0_t, dim=1).expand([len(quants),-1])
                inputSeasonality_t=torch.cat(seasonality[istep-INPUT_WINDOW+1:istep+1],dim=1)  #inputSeasonality_t.shape
                x_t=torch.log(x0_t/(inputSeasonality_t*anchorLevel)) #x_t.shape
                outputSeasonality=torch.cat(seasonality[istep+1:istep+OUTPUT_WINDOW+1],dim=1)
                input_t=torch.cat([x_t, torch.log10(anchorLevel), embeddedDates_t, 
                                   outputSeasonality-1, context_t],  dim=1)
                #input_t.shape
                input_t=torch.cat([input_t, quants_t], dim=1)  #here we expand
                
                if len(varNames)==0:
                  for i in range(x_t.shape[1]):  
                    varNames.append("x"+str(i))
                  varNames.append("anch")
                  for i in range(embeddedDates_t.shape[1]):  
                    varNames.append("dat"+str(i))
                  for i in range(outputSeasonality.shape[1]):  
                    varNames.append("seas"+str(i))
                  for i in range(context_t.shape[1]):  
                    varNames.append("ctx"+str(i))
                      
                forec0_t=rnn(input_t, quants_t)
                #print(forec0_t, forec0_t.shape)
                if torch.isnan(forec0_t).sum()>0:
                  print(forec0_t)
                  trouble("nans in test forecast")
                  
                if len(forec0_t.shape)==1:
                  forec0_t=torch.unsqueeze(forec0_t,dim=0)
                  
                levSm=torch.sigmoid(forec0_t[:,0].view([len(quants),1])+levSm0)
                sSm=torch.sigmoid(forec0_t[:,1].view([len(quants),1])+sSm0)
                  
                if warmupFinished:
                  actuals=ppBatch.y[:,istep+1:istep+OUTPUT_WINDOW+1].expand(len(quants),-1).detach().cpu().numpy()
                  forec_t=torch.exp(forec0_t[:,2:])*anchorLevel*outputSeasonality  #forec_t.shape
                  forec=np.maximum(smallNegative,forec_t.detach().cpu().numpy())
                  if not FINAL:
                    loss=validationLossFunc(forec, actuals, ppBatch.maseNormalizer)#rmse, bias, mase 
                    epochValidationErrors.append([loss])
            
                  if saveVarImportance:
                    keys_l=[[series,str(dat)]]
                    key_df=pd.DataFrame(keys_l, columns=["series","date"])
                    varImportance=rnn.cells[0].varImportance_t.detach().cpu().numpy()
                    v_df = pd.DataFrame(varImportance, columns=varNames)
                    var_df=pd.concat([key_df,v_df],axis=1)
                    if varImportance_df is None:
                      varImportance_df=var_df
                    else:
                      varImportance_df=varImportance_df.append(var_df)
                    
                  if FINAL:
                    #horizon=0
                    for horizon in range(OUTPUT_WINDOW):
                      qForecs1=forec[:len(MANDATORY_TESTING_QUANTS),horizon].astype(float)
                      qForecs2=sorted(forec[len(MANDATORY_TESTING_QUANTS):,horizon].astype(float))
                      qForec=np.concatenate((qForecs1, qForecs2), axis=0)
                      
                      if workerNumber==1: # or workerNumber==4:
                        actu=float(actuals[0,horizon])
                        actuas=[actu]*len(quants)
                        
                        quants1=MANDATORY_TESTING_QUANTS
                        quants2=sorted(quants[len(MANDATORY_TESTING_QUANTS):])
                        quants12=quants1+quants2
  
                        save_df=pd.DataFrame(zip(quants12, actuas, list(qForec)), columns=["quants","actuals","forec"+str(workerNumber)])
                      else:
                        save_df=pd.DataFrame(list(qForec), columns=["forec"+str(workerNumber)])  
  
                      #save_df.dtypes
                      save_df=save_df.astype(np.float32)
                      oneSeriesForecs.append(save_df)
                  else:
                    da=dat                      
                    theVals=[now, workerNumber, iEpoch, da,  
                              series]
                      
                    horizon=0
                    for horizon in range(OUTPUT_WINDOW):
                      actu=float(actuals[0,horizon])
                      if np.isnan(actu):
                        actu=None
                      theVals.extend([actu])
                        
                      floats=forec[:,horizon].astype(float)
                      theVals.extend(sorted(floats))
                          
                    cursor.execute(INSERT_QUERY,theVals)
                  #end of saving
                #end of warmup finished
              #through steps of the batch
          
              if FINAL:
                oneSeriesForec = pd.concat(oneSeriesForecs, axis=0)
                with open(forecsSavingPath, 'wb') as handle:
                  pickle.dump(oneSeriesForec, handle, protocol=pickle.HIGHEST_PROTOCOL)
            #through batches/series
        numOfUpdatesSoFar+=1          
      #through batches
      isubEpoch+=1
    #through sub-epoch
    if USE_ODBC and iEpoch>=2:
      if saveVarImportance:
        varImportance_df.to_csv(varImportancePath, index=False)
      dbConn.commit()      
      

    if len(epochTrainingErrors)>0:
      avgTrainLoss=np.mean(epochTrainingErrors)
      print('epoch:',iEpoch,
            ' avgTrainLoss:',f'{avgTrainLoss:.3}') 
      
    if len(epochValidationErrors)>0:
      validErrors0=np.concatenate(epochValidationErrors, axis=0) 
      validErrors=np.nanmean(validErrors0,axis=0)
      print('valid, RMSE:', f'{validErrors[0]:.3}',
            ' MAPE:', f'{validErrors[3]*100:.3}', 
            ' %bias:', f'{validErrors[1]*100:.3}',  
            ' MASE:', f'{validErrors[2]:.3}',
            ' pbLoss:', f'{validErrors[4]:.3}', 
            end=', % exceeded:')
      for iq in range(NUM_OF_QUANTS):
        print(' ',QUANTS[iq],':', f'{validErrors[5+iq]*100:.3}', end=',')
        
      minLSm=torch.tensor(100., device=device); maxLSm=torch.tensor(-100., device=device); sumLSm=torch.tensor(0., device=device)
      minSSm=torch.tensor(100., device=device); maxSSm=torch.tensor(-100., device=device); sumSSm=torch.tensor(0., device=device)
      for series in perSeriesParams_d.keys():
        psp=perSeriesParams_d[series]
        sumLSm+=psp.initLevSm
        sumSSm+=psp.initSSm
        if psp.initLevSm<minLSm:
          minLSm=psp.initLevSm
        if psp.initSSm<minSSm:
          minSSm=psp.initSSm
        if psp.initLevSm>maxLSm:
          maxLSm=psp.initLevSm
        if psp.initSSm>maxSSm:
          maxSSm=psp.initSSm
      print("\nLSm logit avg:",f'{(sumLSm/len(perSeriesParams_d)).cpu().item():.3}', 
            " min:",f'{minLSm.detach().cpu().item():.3}', 
            " max:",f'{maxLSm.detach().cpu().item():.3}')
      print("SSm logit avg:",f'{(sumSSm/len(perSeriesParams_d)).cpu().item():.3}', 
            " min:", f'{minSSm.detach().cpu().item():.3}', 
            " max:",f'{maxSSm.detach().cpu().item():.3}')
      print()
  
      if SAVE_NETS and iEpoch>=FIRST_EPOCH_TO_SAVE_NETS: 
        perSeriesParamsExport_d={}
        for series in perSeriesParams_d.keys(): 
          psp=perSeriesParams_d[series]
          perSeriesParamsExport_d[series]=\
            {"initLSm": psp.initLevSm.detach().cpu().item(),
             "initSSm": psp.initSSm.detach().cpu().item(),
             "contextModifier": psp.contextModifier.detach().cpu().numpy()}
        savePath=saveNetsDir+"perSeriesParams"+"_e"+str(iEpoch)+ "_w"+str(workerNumber)+".pickle"   
        with open(savePath, 'wb') as handle:
          pickle.dump(perSeriesParamsExport_d, handle, protocol=pickle.HIGHEST_PROTOCOL)
          
        #
        savePath=saveNetsDir+"e"+str(iEpoch)+ "_w"+str(workerNumber)+"E.pt"
        torch.save(embed, savePath)
             
        #saving cells
        icell=0; cell = rnn.cells[0]
        for cell in rnn.cells:
          savePath=saveNetsDir+"e"+str(iEpoch)+ "_w"+str(workerNumber)+"_c"+str(icell)+".pt"
          torch.jit.save(cell, savePath)
          icell+=1
        savePath=saveNetsDir+"e"+str(iEpoch)+ "_w"+str(workerNumber)+"A.pt"
        torch.save(rnn.adaptor, savePath)
        
        icell=0; cell = contextRnn.cells[0]
        for cell in contextRnn.cells:
          savePath=saveNetsDir+"e"+str(iEpoch)+ "_w"+str(workerNumber)+"_x"+str(icell)+".pt"
          torch.jit.save(cell, savePath)
          icell+=1
        savePath=saveNetsDir+"e"+str(iEpoch)+ "_w"+str(workerNumber)+"Ax.pt"
        torch.save(rnn.adaptor, savePath)              
      
      
    if DISPLAY_SM_WEIGHTS:
      print('levSm w:', rnn.adaptor.weight[0].detach().cpu().numpy())
      print('levSm b:', rnn.adaptor.bias[0].detach().cpu().numpy())

      print('sSm w:', rnn.adaptor.weight[1].detach().cpu().numpy())
      print('sSm b:', rnn.adaptor.bias[1].detach().cpu().numpy())
  
  
  #for series in perSeriesParams_d.keys():
  #  psp=perSeriesParams_d[series]
  #  print(ipsp, "initLSm:", psp.initLevSm.detach().item(), 
  #        "initSSm:", psp.initSSm.detach().item(),
  #        "contextModifier:",psp.contextModifier.detach().cpu().numpy())
        
  print("done.")
  #dbConn.rollback() 

            
          
"""
 CREATE TABLE electraMase(
  series varchar(50) NOT NULL,
  denom real NOT NULL,
 CONSTRAINT electraMase_PK PRIMARY KEY
(
  series asc
))


 CREATE TABLE electra4Models(
  run varchar(300) NOT NULL,
  workerNo tinyint NOT NULL,
  dateTimeOfPrediction datetime NOT NULL,
 CONSTRAINT electra4Models_PK PRIMARY KEY
(
  run ASC,
  workerNo asc
))
  
  query="CREATE TABLE electra18(\
    dateTimeOfPrediction datetime NOT NULL,\
    workerNo tinyint NOT NULL,\
    epoch tinyint NOT NULL,\
    forecOriginDate datetime NOT NULL,\
    series varchar(50) NOT NULL,"
  for ih in range(OUTPUT_WINDOW):
    query+="\nactual"+str(ih+1)+" real,"
    for q in SORTED_PERCENTILES_str:
      query+=" predQ"+q+"_"+str(ih+1)+" real,"
  query+="\nCONSTRAINT electra18_PK PRIMARY KEY (\
    dateTimeOfPrediction ASC,\
    workerNo ASC,\
    epoch ASC, \
    forecOriginDate ASC, \
    series ASC))"
  print(query)
  

#SQL Server or mySQL  
#validation
query="with avgValues as \n(select run, epoch, forecOriginDate, series, \
  count(distinct d.workerNo) workers "
for ih in range(1,OUTPUT_WINDOW+1):
  query+="\n, avg(actual"+str(ih)+") actual"+str(ih) 
  for q in SORTED_PERCENTILES_str:
    query+=", avg(predQ"+q+"_"+str(ih)+") predQ"+q+"_"+str(ih)
query+="\n from electra18 d with (nolock), electra4Models m with (nolock) \
  \n where d.dateTimeOfPrediction =m.dateTimeOfPrediction  and d.workerNo=m.workerNo \
  \n group by run, epoch, forecOriginDate, series)"
query+="\n, perForecMetrics as ( select run, epoch, forecOriginDate, a.series, workers "

for ii in range(len(QUANTS)):
  iq=SORTED_PERCENTILES[ii]
  q=SORTED_PERCENTILES_str[ii]
  for ih in range(1,OUTPUT_WINDOW+1):
    if ih==1:
      query+="\n,("
    else:
      query+=" + "
    if iq>=50:
      query+="case when actual"+str(ih)+">predQ"+q+"_"+str(ih)+" then 100. else (actual"+str(ih)+"-actual"+str(ih)+") end"
    else:
      query+="case when actual"+str(ih)+"<predQ"+q+"_"+str(ih)+" then 100. else (actual"+str(ih)+"-actual"+str(ih)+") end"
  query+=")/"+str(OUTPUT_WINDOW)+" as exceed"+q
  
for ii in range(len(QUANTS)):
  iq=SORTED_PERCENTILES[ii]
  q=SORTED_PERCENTILES_str[ii]
  for ih in range(1,OUTPUT_WINDOW+1):
    if ih==1:
      query+="\n,("
    else:
      query+=" + "
    query+="case when actual"+str(ih)+">predQ"+q+"_"+str(ih)+" then (actual"+str(ih)+"-predQ"+q+"_"+str(ih)+")*"+q+ \
         " else (actual"+str(ih)+"-predQ"+q+"_"+str(ih)+")*"+str(iq-100) +" end"
  query+=")/"
  for ih in range(1,OUTPUT_WINDOW+1):
    if ih==1:
      query+="("
    else:
      query+=" + "
    query+="actual"+str(ih)
  query+=") as pbLoss"+q
  
for ii in range(len(QUANTS)):
  iq=SORTED_PERCENTILES[ii]
  q=SORTED_PERCENTILES_str[ii]
  for ih in range(1,OUTPUT_WINDOW+1):
    if ih==1:
      query+="\n,("
    else:
      query+=" + "
    query+="case when actual"+str(ih)+">predQ"+q+"_"+str(ih)+" then (predQ"+q+"_"+str(ih)+"-actual"+str(ih)+")*"+q+ \
         " else (actual"+str(ih)+"-predQ"+q+"_"+str(ih)+")*"+str(iq-100)+" end"
  query+=")/"
  for ih in range(1,OUTPUT_WINDOW+1):
    if ih==1:
      query+="("
    else:
      query+=" + "
    query+="actual"+str(ih)
  query+=") as pbBias"+q
  
for ih in range(1,OUTPUT_WINDOW+1):
  if ih==1:
    query+=",("
  else:
    query+=" + "
  query+="abs(predQ50_"+str(ih)+"-actual"+str(ih)+")" 
query+=")/(m.denom*"+str(OUTPUT_WINDOW)+") as MASE \n"

for ih in range(1,OUTPUT_WINDOW+1):
  if ih==1:
    query+=",("
  else:
    query+=" + "
  query+="(predQ50_"+str(ih)+"-actual"+str(ih)+")" 
query+=")/(m.denom*"+str(OUTPUT_WINDOW)+") as mBias \n"    
  
for ih in range(1,OUTPUT_WINDOW+1):
  if ih==1:
    query+=",("
  else:
    query+=" + "
  query+="abs(predQ50_"+str(ih)+"-actual"+str(ih)+")/actual"+str(ih) 
query+=")/"+str(OUTPUT_WINDOW)+" as MAPE \n"  

for ih in range(1,OUTPUT_WINDOW+1):
  if ih==1:
    query+=",("
  else:
    query+=" + "
  query+="(predQ50_"+str(ih)+"-actual"+str(ih)+")/actual"+str(ih) 
query+=")/"+str(OUTPUT_WINDOW)+" as MPE \n"  

for ih in range(1,OUTPUT_WINDOW+1):
  if ih==1:
    query+=",("
  else:
    query+=" + "
  query+="(predQ50_"+str(ih)+"-actual"+str(ih)+")*(predQ50_"+str(ih)+"-actual"+str(ih)+")" 
query+=")/"+str(OUTPUT_WINDOW)+" as MSE \n"  

query+="from avgValues a, electraMase m where m.series=a.series),\n"
#aggregate over forecasts
query+="perSeries as (select run, series, epoch,  \
  avg(MASE) MASE, avg(mBias) mBias, \
  avg(MAPE) MAPE, avg(MPE) pcBias, \
  sqrt(avg(MSE)) RMSE  \n"
for q in SORTED_PERCENTILES_str:
  query+= ", avg(exceed"+q+") exceed"+q
  query+= ", avg(pbLoss"+q+") pbLoss"+q
  query+= ", avg(pbBias"+q+") pbBias"+q
query+="\n,count(distinct forecOriginDate) numForecasts, avg(workers) workers \n\
 from perForecMetrics \n\
 group by run, series, epoch) \n"
query+="select run, epoch, min(workers) workers, count(distinct series) numSeries,\
 round(avg(MASE),3) MASE, round(avg(mBias)*100,1) mBias, \
 round(avg(MAPE)*100,3) MAPE, round(avg(pcBias)*100,3) pcBias, \
 round(avg(RMSE),3) RMSE \n "

query+=",round(avg("
for q in SORTED_PERCENTILES_str:
  if q!=SORTED_PERCENTILES_str[0]:
    query+="+"
  query+="pbLoss"+q
query+="),2) sumPbLoss\n"
   
for q in SORTED_PERCENTILES_str:
  query+= ", round(avg(exceed"+q+"),1) exceed"+q
  query+= ", round(avg(pbLoss"+q+"),3) pbLoss"+q
  query+= ", round(avg(pbBias"+q+"),3) pbBias"+q
query+=", avg(numForecasts) numForecasts, max(workers) workers \n\
 from perSeries \n\
 group by run, epoch \n\
 order by run, epoch\n" 
  
print(query)
  
"""