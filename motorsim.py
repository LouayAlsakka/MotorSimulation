import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import ltspice
import matplotlib.pyplot as pl
import numpy as np
import os


def sigavrg(sig,time):
    sa=0
    for t in range (len(time)-1):
        sa=sa+sig[t]*(time[t+1]-time[t])
    return(sa/time[len(time)-1])

projfn="motor1"
paramfn="param"+projfn+".txt"
param=open(paramfn,"w")
R="200m"
L="5m"
I="4.25"
R1=R
R2=R
R3=R
L1=L
L2=L
L3=L
I1=I
I2=I
I3=I
param.write(".param R1 "+R1+ "\n",)
param.write(".param R2 "+R2+ "\n",)
param.write(".param R3 "+R3+ "\n",)

param.write(".param L1 "+L1+ "\n",)
param.write(".param L2 "+L2+ "\n",)
param.write(".param L3 "+L3+ "\n",)

param.write(".param I1 "+I1+ "\n",)
param.write(".param I2 "+I2+ "\n",)
param.write(".param I3 "+I3+ "\n",)

teta1=0
teta2=teta1+120
teta3=teta1+240

param.write(".param teta1 "+str(teta1) +"\n")
param.write(".param teta2 "+str(teta2) +"\n")
param.write(".param teta3 "+str(teta3) +"\n")

Ifreq=80
bmffreq=80
param.write(".param Ifreq1 "+str(Ifreq) +"\n")
param.write(".param Ifreq2 "+str(Ifreq) +"\n")
param.write(".param Ifreq3 "+str(Ifreq) +"\n")

param.write(".param bmffreq1 "+str(bmffreq) +"\n")
param.write(".param bmffreq2 "+str(bmffreq) +"\n")
param.write(".param bmffreq3 "+str(bmffreq) +"\n")

bmff=10
bmfr=0.6
param.write(".param bmff1 "+str(bmff) +"\n")
param.write(".param bmff2 "+str(bmff) +"\n")
param.write(".param bmff3 "+str(bmff) +"\n")

param.write(".param bmfr1 "+str(bmfr) +"\n")
param.write(".param bmfr2 "+str(bmfr) +"\n")
param.write(".param bmfr3 "+str(bmfr) +"\n")

param.close()
cmd="/Applications/LTspice.app/Contents/MacOS/LTspice -b "+projfn+".net"
os.system(cmd)

rawfname="./" + projfn+".raw"
l = ltspice.Ltspice(rawfname)
l.parse()
time = l.getTime()
ln=l.getVariableNames()
sig = {}
for var in ln:
     s=var.replace('(','').replace(')','')
     sig[s]=l.getData(var)


WM=sig["Vn007"]*sig["IR1"]
WS=sig["Vn002"]*sig["IR1"]
WR=abs((sig["Vn002"]-sig["Vn004"])* sig["IR1"])
print(sigavrg(WM,time)*100)
print(sigavrg(WS,time)*100)
print(sigavrg(WR,time)*100)

eff=sigavrg(WM,time)*100/sigavrg(WS,time)
eff
sigavrg(WR,time)*100/sigavrg(WM,time)

WW=abs((sig["Vn007"]-sig["Vp001"])* sig["IR1"])
print(sigavrg(WW,time)*100)

