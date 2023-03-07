'''
/*
Copyright <2023> <Louay Alsakka>

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


'''

#!/usr/bin/python                                                               
import sys,os
import subprocess
import os
import sys
import argparse
import numpy as np
import ltspice
import matplotlib.pyplot as pl
import numpy as np
from scipy.fftpack import fftfreq, irfft, rfft

parser = argparse.ArgumentParser();

parser.add_argument('--pn'  , default="motor4",type=str,help='Project name, default motor4')
parser.add_argument('--st'  , default=0,type=int,help='Starting theta')
parser.add_argument('--et'  , default=10,type=int,help='Ending theta')
parser.add_argument('--ts'  , default=5,type=int,help= 'theta steps')
parser.add_argument('--uc'  , default=10,type=int,help='number of switch period before updating the angle (convergence time)')
parser.add_argument('--swfreq'  , default=10000,type=int,help='Switch frequency in HZ')
parser.add_argument('--vbus'  , default=180,type=int,help='Switch frequency in HZ')
parser.add_argument('--rpm'  , default=1500,type=int,help='RPM mechanical rotation with X4 electrical')
parser.add_argument('--swonv'  , default=210,type=int,help='Swwitch  ON voltage')
parser.add_argument('--swoffv'  , default=0,type=int,help='Swwitch  OFF voltage')
parser.add_argument('--swot'  , default=500,type=int,help='Switch off time in nano seconds when all switches are off xx%%')
parser.add_argument('--scale'  , default=50,type=int,help='Percent of ON time default 50%')
parser.add_argument('--simdur'  , default=1000,type=int,help='Duration of simulation in ms')
parser.add_argument('--emfK'  , default=8,type=float,help='b-emf voltage scaler per rpm in HZ, 2 means 100HZ will cause 50V EMF, while 4 means 25V and so on')
parser.add_argument('--RemfK'  , default=10,type=float,help='Reverseb-emf voltage scaler per emf V default is 10, mean 10V of EMF will cause 1 V of Remf')
parser.add_argument('--bmft1'  , default=0,type=int,help='Shift by degree back emf to find best angle (simulating motor closing in when consistent error in angle show up)')
parser.add_argument('--R'  , default=200,type=float,help='Motor per branch resistence per mohm defualt 200m')
parser.add_argument('--L'  , default=5,type=float,help='Motor per branch inductence per mH defualt 5mH')
parser.add_argument('--harmonic'  , default=1,type=int,help='Calculate harmonic losses')
parser.add_argument('--fn'  , default="S",help='output file name')

args = parser.parse_args(sys.argv[1:]);

def genpid(stability=10,decayp=100,startp=20,endp=10,sample=100):
    S=[]
    scale=10./(2.5*stability)
    for i in range(sample):
        w1=(np.exp(scale*(-1.)*i))
        w2=w1*np.cos(2*np.pi*(i%decayp)*1.0/decayp)
        w=endp+(startp-endp)*w2

        S.append(w)
    return S

def sigexpan(S,time,st=0,et=0,stopfreq=10000,inc=0.00000001):
    if(et==0):
        et=time[len(time)-1]
    si=[]
    t=st
    tidx=0
    while(time[tidx]<st):
        tidx=tidx+1
    if(tidx):
        tidx=tidx-1
    while(t<et):
        si.append(S[tidx])
        #increase by 10 ns
        t=t+inc
        if(time[tidx]<t):
            tidx=tidx+1
    return(si)

def sigavrg(sig,time,st=0,et=0,mode='a'):
    sa=0
    if(et==0):
        et=time[len(time)-1]
    avrgt=0
    for t in range (len(time)-1):
        if(time[t]>et):
            break
        if(time[t]<st):
            continue
        if(((mode=='p')and(sig[t]>0)) or ((mode=='n') and (sig[t]<0)) or (mode=='a') ):
            sa=sa+sig[t]*(time[t+1]-time[t])
            avrgt=avrgt+(time[t+1]-time[t])
        else:
            avrgt=avrgt+(time[t+1]-time[t])
    return(sa*1.0/avrgt)

def sigavrgexp(sig,ts=0.00000001,st=0,et=0,mode='a'):
    sa=0
    if(et==0):
        et=(len(sig)-1)*ts
    avrgt=0
    sti=int(st/ts)
    eti=int(et/ts)
    if(eti>len(sig)-1):
        eti=(len(sig)-1)
    #print(sti,eti)
    for t in range (sti,eti-1):
        if(((mode=='p')and(sig[t]>0)) or ((mode=='n') and (sig[t]<0)) or (mode=='a') ):
            sa=sa+sig[t]*ts
            avrgt=avrgt+ts
        else:
            avrgt=avrgt+ts
    return(sa*1.0/avrgt)

def gettimeidx(time,t):
    for i in range(len(time)):
        if(time[i]>t):
            return(i-1)
    return((len(time)-1))
           
def calphase(s1,s2,time,dur,st=0.1):
    et=st+dur
    sti=gettimeidx(time,st)    
    eti=gettimeidx(time,et)
    #print(sti,eti)
    maxis1=np.argmax(s1[sti:eti])
    maxis2=np.argmax(s2[sti:eti])
    ts1=time[maxis1]
    ts2=time[maxis2]
    phase=(ts1-ts2)*360.0/dur
    if(phase<0):
        phase =phase+360
    if(phase>180):
        phase =phase-360
        
    return phase
#expanded signal calculate the phase
def calphaseexp(s1,s2,dur,ts=0.00000001,st=0.1):
    et=st+dur
    sti=int(st/ts)
    eti=sti+int(dur/ts)

    maxis1=np.argmax(s1[sti:eti])
    maxis2=np.argmax(s2[sti:eti])
    ts1=maxis1*ts
    ts2=maxis2*ts
    phase=(ts1-ts2)*360.0/dur
    if(phase<0):
        phase =phase+360
    if(phase>180):
        phase =phase-360
    return phase
    
def runproj(projfn,dur):
    global IV1
    global IV1F
    global V1
    global op1F
    global op1e
    global op1ea
    cmd="/Applications/LTspice.app/Contents/MacOS/LTspice -b "+projfn+".net"
    os.system(cmd)

    rawfname="./" + projfn+".raw"
    l = ltspice.Ltspice(rawfname)
    l.parse()
    time = l.getTime()
    global lttime
    global signals
    ln=l.getVariableNames()
    sig = {}
    global signames
    signames=[]
    for var in ln:
        s=var.replace('(','').replace(')','')
        signames.append(s)
        sig[s]=l.getData(var)
    lttime=time
    signals=sig
        
    V1=sig['Vvemf1']-sig['Vmg']
    V2=sig['Vvemf2']-sig['Vmg']
    V3=sig['Vvemf3']-sig['Vmg']
    IR1=sig['IR1']
    IR2=sig['IR2']
    IR3=sig['IR3']
    I1V=sig['IV1']
    I2V=sig['IV2']
    I3V=sig['IV3']
    VSW1=np.subtract(sig['Vn001'],sig['Vv1'])
    VSW2=np.subtract(sig['Vn001'],sig['Vv2'])
    VSW3=np.subtract(sig['Vn001'],sig['Vv3'])
    VSW4=sig['Vv1']
    VSW5=sig['Vv2']
    VSW6=sig['Vv3']
    VGSW1=np.subtract(sig['Vvsw1'],sig['Vv1'])
    VGSW2=np.subtract(sig['Vvsw2'],sig['Vv2'])
    VGSW3=np.subtract(sig['Vvsw3'],sig['Vv3'])
    VGSW4=sig['Vvsw4']
    VGSW5=sig['Vvsw5']
    VGSW6=sig['Vvsw6']
    
    if (projfn=="motor4"): 
    	ISW1=sig['IdM2']
    	ISW2=sig['IdM4']
    	ISW3=sig['IdM6']
    	ISW4=sig['IdM3']
    	ISW5=sig['IdM5']
    	ISW6=sig['IdM7']

    	IGSW1=sig['IgM2']
    	IGSW2=sig['IgM4']
    	IGSW3=sig['IgM6']
    	IGSW4=sig['IgM3']
    	IGSW5=sig['IgM5']
    	IGSW6=sig['IgM7']
    else:
        ISW1=sig['Ixu6:1']
        ISW2=sig['Ixu4:1']
        ISW3=sig['Ixu2:1']
        ISW4=sig['Ixu5:1']
        ISW5=sig['Ixu3:1']
        ISW6=sig['Ixu1:1']
        
        IGSW1=sig['Ixu6:2']
        IGSW2=sig['Ixu4:2']
        IGSW3=sig['Ixu2:2']
        IGSW4=sig['Ixu5:2']
        IGSW5=sig['Ixu3:2']
        IGSW6=sig['Ixu1:2']
    	
    SWL=[]
    LOSSES=[]
    SW1L=np.add(np.multiply(VSW1,ISW1),np.multiply(VGSW1,IGSW1))
    SW2L=np.add(np.multiply(VSW2,ISW2),np.multiply(VGSW2,IGSW2))
    SW3L=np.add(np.multiply(VSW3,ISW3),np.multiply(VGSW3,IGSW3))
    SW4L=np.add(np.multiply(VSW4,ISW4),np.multiply(VGSW4,IGSW4))
    SW5L=np.add(np.multiply(VSW5,ISW5),np.multiply(VGSW5,IGSW5))
    SW6L=np.add(np.multiply(VSW6,ISW6),np.multiply(VGSW6,IGSW6))
    SW1La=sigavrg(SW1L,time,st=0.1)
    SW2La=sigavrg(SW2L,time,st=0.1)
    SW3La=sigavrg(SW3L,time,st=0.1)
    SW4La=sigavrg(SW4L,time,st=0.1)
    SW5La=sigavrg(SW5L,time,st=0.1)
    SW6La=sigavrg(SW6L,time,st=0.1)
    
    SWL.append(SW1La)
    SWL.append(SW2La)
    SWL.append(SW3La)
    SWL.append(SW4La)
    SWL.append(SW5La)
    SWL.append(SW6La)
    SWL.append(SW1La+SW2La+SW3La+SW4La+SW5La+SW6La)
    global I1Ve
    IR1s=np.multiply(IR1,IR1)
    IR2s=np.multiply(IR2,IR2)
    IR3s=np.multiply(IR3,IR3)
    WR1=np.multiply(IR1s,args.R)
    WR2=np.multiply(IR2s,args.R)
    WR3=np.multiply(IR3s,args.R)
    p=[]
    pp=[]
    global V1e
    global V2e
    global V3e
    V1e=sigexpan(V1,time,st=0.1,et=0)
    V2e=sigexpan(V2,time,st=0.1,et=0)
    V3e=sigexpan(V3,time,st=0.1,et=0)
    I1Ve=sigexpan(I1V,time,st=0.1,et=0)
    I2Ve=sigexpan(I2V,time,st=0.1,et=0)
    I3Ve=sigexpan(I3V,time,st=0.1,et=0)                
    bmffreq=args.rpm*4/60.
    freq=bmffreq*2e-7
    fac=0.05
    global I1VF
    global I2VF
    global I3VF
    
    W = fftfreq(len(I1Ve), d=0.1);f_signal1 = rfft(I1Ve);cut_f_signal1 = f_signal1.copy();cut_f_signal1[(np.abs(W)>(freq+freq*fac))] = 0;cut_f_signal1[(np.abs(W)<(freq-freq*fac))] = 0;I1VF = irfft(cut_f_signal1);
    W = fftfreq(len(I2Ve), d=0.1);f_signal2 = rfft(I2Ve);cut_f_signal2 = f_signal2.copy();cut_f_signal2[(np.abs(W)>(freq+freq*fac))] = 0;cut_f_signal2[(np.abs(W)<(freq-freq*fac))] = 0;I2VF = irfft(cut_f_signal2);
    W = fftfreq(len(I3Ve), d=0.1);f_signal3 = rfft(I3Ve);cut_f_signal3 = f_signal3.copy();cut_f_signal3[(np.abs(W)>(freq+freq*fac))] = 0;cut_f_signal3[(np.abs(W)<(freq-freq*fac))] = 0;I3VF = irfft(cut_f_signal3);

    p.append(calphaseexp(V1e,I1VF,dur,st=0.05))
    p.append(calphaseexp(V2e,I2VF,dur,st=0.05))
    p.append(calphaseexp(V3e,I3VF,dur,st=0.05))
    pp.append(calphaseexp(I2VF,I1VF,dur,st=0.05))
    pp.append(calphaseexp(I3VF,I2VF,dur,st=0.05))
    pp.append(calphaseexp(I1VF,I3VF,dur,st=0.05))
    
    ip=np.multiply(sig['Vvbus'],sig['IVbus'])
    ipsw1=np.multiply(sig['Vvsw1'],sig['IVsw1'])
    ipsw2=np.multiply(sig['Vvsw2'],sig['IVsw2'])
    ipsw3=np.multiply(sig['Vvsw3'],sig['IVsw3'])
    ipsw4=np.multiply(sig['Vvsw4'],sig['IVsw4'])
    ipsw5=np.multiply(sig['Vvsw5'],sig['IVsw5'])
    ipsw6=np.multiply(sig['Vvsw6'],sig['IVsw6'])
    tip=np.add(ip,ipsw1)
    tip=np.add(tip,ipsw2)
    tip=np.add(tip,ipsw3)
    tip=np.add(tip,ipsw4)
    tip=np.add(tip,ipsw5)
    tip=np.add(tip,ipsw6)
    ipa=sigavrg(tip,time,st=0.096,et=0.199,mode='a')
    ipna=sigavrg(tip,time,st=0.096,et=0.199,mode='n')
    ippa=sigavrg(tip,time,st=0.096,et=0.199,mode='p')
    tipe=sigexpan(tip,time,st=0.1,et=0)
    #iprms=np.sqrt(np.mean(tipe**2))
    iprms=np.sqrt(np.mean(np.power(tipe,2)))
    ipc=[]
    ipc.append(ipa)
    ipc.append(ipna)
    ipc.append(ippa)
    ipc.append(iprms)

    LOSSES.append((SW1La+SW2La+SW3La+SW4La+SW5La+SW6La)/ipa)

    
    op=[]    

    opn=[]
    opp=[]
    RL=[]

    op1=np.multiply(V1,sig['IV1'])
    op2=np.multiply(V2,sig['IV2'])
    op3=np.multiply(V3,sig['IV3'])


    IS1=np.multiply(sig['IV1'],sig['IV1'])
    IS2=np.multiply(sig['IV2'],sig['IV2'])
    IS3=np.multiply(sig['IV3'],sig['IV3'])
    RL1=args.R/1000.0*sigavrg(IS1,time,st=0.1)
    RL2=args.R/1000.0*sigavrg(IS2,time,st=0.1)
    RL3=args.R/1000.0*sigavrg(IS3,time,st=0.1)
    RL.append(RL1)
    RL.append(RL2)
    RL.append(RL3)
    
    LOSSES.append((RL1+RL2+RL3)/ipa)
    op1pa=sigavrg(op1,time,st=0.096,et=0.199,mode='p')
    op1na=sigavrg(op1,time,st=0.096,et=0.199,mode='n')
    op1a=sigavrg(op1,time,st=0.096,et=0.199,mode='a')
    
    op2pa=sigavrg(op2,time,st=0.096,et=0.199,mode='p')
    op2na=sigavrg(op2,time,st=0.096,et=0.199,mode='n')
    op2a=sigavrg(op2,time,st=0.096,et=0.199,mode='a')
    
    op3pa=sigavrg(op3,time,st=0.096,et=0.199,mode='p')
    op3na=sigavrg(op3,time,st=0.096,et=0.199,mode='n')
    op3a=sigavrg(op3,time,st=0.096,et=0.199,mode='a')
    #freq=26.6e-6
    if(args.harmonic):
        op1e=np.multiply(I1Ve,V1e)
        op2e=np.multiply(I2Ve,V2e)
        op3e=np.multiply(I3Ve,V3e)
        #op1ea=sigavrgexp(op1e,st=0)
        op1F=np.multiply(I1VF,V1e)
        op2F=np.multiply(I2VF,V2e)
        op3F=np.multiply(I3VF,V3e)
        sti=int(0.07*len(op1F))
        eti=int(0.93*len(op1F)) 
        op1fa=np.mean(op1F[sti:eti])        
        op2fa=np.mean(op2F[sti:eti])
        op3fa=np.mean(op3F[sti:eti])
        
        op1frms=np.sqrt(np.mean(op1F**2))
        op2frms=np.sqrt(np.mean(op2F**2))
        op3frms=np.sqrt(np.mean(op3F**2))
        
        
        '''
        #this to make sure we blance the filter signal same way the original signal
        op1fa=op1fa*abs(op1pa/op1a)
        op2fa=op2fa*abs(op2pa/op2a)
        op3fa=op3fa*abs(op3pa/op3a)
        '''
    '''
    # this to make sure we tax the generation since it will have unefficency going to mechanical then back to electrical 0.9*0.9 ~0.8 which mean 20% of generations will be true losses forever    
    op1a=op1a+op1na*.2
    op2a=op2a+op2na*.2
    op3a=op3a+op3na*.2
    if(args.harmonic):
        op1fa=op1fa+op1na*.2
        op2fa=op1fa+op2na*.2
        op3fa=op1fa+op3na*.2
    '''
    oprms=[]
    op1rms=np.sqrt(np.mean(op1e**2))
    op2rms=np.sqrt(np.mean(op2e**2))
    op3rms=np.sqrt(np.mean(op3e**2))
    oprms.append(op1rms)
    oprms.append(op2rms)
    oprms.append(op3rms)
    
    op.append(op1a)
    op.append(op1na)
    op.append(op1pa)
    
    op.append(op2a)
    op.append(op2na)
    op.append(op2pa)
    
    op.append(op3a)            
    op.append(op3na)            
    op.append(op3pa)
    if(args.harmonic):
        op.append(op1fa)            
        op.append(op2fa)            
        op.append(op3fa)
        oprms.append(op1frms)
        oprms.append(op2frms)
        oprms.append(op3frms)

    
    eff=(op1a+op2a+op3a)*100./ipa
    if(op1a+op2a+op3a < 0):
        eff=0
    eff2=0
    effa=[]
    effa.append(eff)
    if(args.harmonic):
        eff2=(op1fa+op2fa+op3fa)*100./ipa
        if(eff==0):
            eff2=0
        effa.append(eff2)
    #print("Phase=%d degree",p)
    return(p,pp,ipc,op,effa,SWL,RL,oprms,LOSSES)

def writeparam(paramfn="parammotor3.txt",emfK=2,R=0.2,L=0.005):
    #    Ifreq=80, bmffreq=80):

    bmffreq=args.rpm*4/60
    bmff=bmffreq/emfK
    bmfr=bmff/args.RemfK
    #bmfr=bmff*np.sin(2*np.pi*eteta/360)
    param=open(paramfn,"w")
    R1=R
    R2=R
    R3=R
    L1=L
    L2=L
    L3=L
    param.write(".param bmft1 "+str(args.bmft1)+ "\n",)
    param.write(".param bmft2 "+str(240+args.bmft1)+ "\n",)
    param.write(".param bmft3 "+str(120+args.bmft1)+ "\n",)
    param.write(".param R1 "+str(R1)+ "\n",)
    param.write(".param R2 "+str(R2)+ "\n",)
    param.write(".param R3 "+str(R3)+ "\n",)

    param.write(".param L1 "+str(L1)+ "\n",)
    param.write(".param L2 "+str(L2)+ "\n",)
    param.write(".param L3 "+str(L3)+ "\n",)


    teta1=0
    teta2=teta1+120
    teta3=teta1+240

    param.write(".param teta1 "+str(teta1) +"\n")
    param.write(".param teta2 "+str(teta2) +"\n")
    param.write(".param teta3 "+str(teta3) +"\n")

    
    param.write(".param bmffreq1 "+str(bmffreq) +"\n")
    param.write(".param bmffreq2 "+str(bmffreq) +"\n")
    param.write(".param bmffreq3 "+str(bmffreq) +"\n")
    

    param.write(".param bmff1 "+str(bmff) +"\n")
    param.write(".param bmff2 "+str(bmff) +"\n")
    param.write(".param bmff3 "+str(bmff) +"\n")

    param.write(".param bmfr1 "+str(bmfr) +"\n")
    param.write(".param bmfr2 "+str(bmfr) +"\n")
    param.write(".param bmfr3 "+str(bmfr) +"\n")
    simdur=args.simdur/1000.
    param.write(".param simend "+str(simdur)+"\n")
    vbus=args.vbus
    param.write(".param vbus "+str(vbus)+"\n")
    param.close()


scale=args.scale/100.
cmd="python3 "
for ar in sys.argv:
    cmd+=ar
    cmd+=" "
cmdfn=sys.argv[0].split('.')[0]+"cmd.txt"    
cmdf=open(cmdfn,"w")
print(cmd)
cmdf.write(cmd)
cmdf.write("\n")
cmdf.close()
t=0

phase2state=[[1,2],[2,3],[3,4],[4,5],[5,6],[6,1]]
def initState():
    state=[]
    #state 0,7 are off, 1,2,3,4,5,6 are the active states. state  8 is used to switch off between changes
    state.append([0,0,0,1,1,1])
    
    state.append([1,0,0,0,1,1])
    state.append([1,1,0,0,0,1])
    state.append([0,1,0,1,0,1])
    state.append([0,1,1,1,0,0])
    state.append([0,0,1,1,1,0])
    state.append([1,0,1,0,1,0])
    
    state.append([1,1,1,0,0,0])
    state.append([0,0,0,0,0,0])
    return(state)

def addsig(time,sig,value,dur,t):
    for i in range(6):
        time[i].append(t)
        sig[i].append(value[i])
        time[i].append(t+dur)
        sig[i].append(value[i])
    return(t+dur+50)
def writeStofiles(S,time,fname):
    for sw in range(6):
        fn=fname+str(sw+1)+".txt"
        #print(fn)
        f=open(fn,"w")
        for i in range(len(time[sw])):
            if(S[sw][i]):
                f.write(str(time[sw][i])+"ns"+"  "+str(args.swonv)+"v\n")
            else:
                f.write(str(time[sw][i])+"ns"+"  "+str(args.swoffv)+"v\n")
        f.close()
def initST():
    time=[]
    S=[]    
    for i in range(6):
        time.append([])
        S.append([])
    return(S,time)
def filterS(S,time,dur):
    for sw in range(6):
        for i in range(1,len(time)-2):
            if((S[sw][i]==0)and(S[sw][i+1]==0)and (S[sw][i-1]==1) and (S[sw][i+2]==1)and ((time[i+2]-time[i-1]) < due)):
                S[sw][i]=1
                S[sw][i+1]=1
        
def calS(S,time,swot,theta,ocd,scd,lastt,uc):
    t=0
    angle=0
    angleupdate=0
    laststate=8
    while(t<lastt):
        angleupdate=angleupdate+1
        if(angleupdate >= uc):
            angle=((t%ocd)*360/ocd+theta)%360
            #anga.append(angle)
            angle=angle%360
            angleupdate=0    
        phase=int(angle/60)
        #phasea.append(phase)
        angoff=angle%60
        #angoffa.append(angoff)
        #lweight=np.cos(2*np.pi*angoff/360)
        #lweight=(60-angoff)/60.
        #lweight=np.sin(60-angoff)/(np.sin(60-angoff)+np.sin(angoff))
        lweight=np.sin(2*np.pi*(60-angoff)/360.)
        #lweighta.append(lweight)
        #hweight=np.cos(2*np.pi*(60-angoff)/360)
        #hweight=angoff/60.0
        #hweight=np.sin(angoff)/(np.sin(60-angoff)+np.sin(angoff))
        hweight=np.sin(2*np.pi*angoff/360.)
        #hweighta.append(hweight)
        tlow=int(scd*scale*lweight)
        if(tlow<swot*2):
            tlow=0        
        thigh=int(scd*scale*hweight)
        if(thigh<swot*2):
            thigh=0        
        toff=int(scd-(tlow+thigh))
        if(toff<swot*2):
            toff=0
        lstate=phase2state[phase][0]
        hstate=phase2state[phase][1]

        if(laststate==0):
            t=addsig(time,S,state[0],toff,t)
            if(thigh>0):
                t=addsig(time,S,state[8],swot,t)
                t=addsig(time,S,state[hstate],thigh,t)
            if(tlow>0):
                t=addsig(time,S,state[8],swot,t)
                t=addsig(time,S,state[lstate],tlow,t)
                laststate=lstate
        elif(laststate==lstate):
            t=addsig(time,S,state[lstate],tlow,t)
            if(thigh>0):
                t=addsig(time,S,state[8],swot,t)
                t=addsig(time,S,state[hstate],thigh,t)
            if(toff):
                t=addsig(time,S,state[8],swot,t)
                t=addsig(time,S,state[0],toff,t)          
                laststate=0
        else:
            if(tlow>0):
                t=addsig(time,S,state[8],swot,t)
                t=addsig(time,S,state[lstate],tlow,t)
            if(thigh>0):
                t=addsig(time,S,state[8],swot,t)
                t=addsig(time,S,state[hstate],thigh,t)
            if(toff):
                t=addsig(time,S,state[8],swot,t)
                t=addsig(time,S,state[0],toff,t)          
                laststate=0                
projname=args.pn
bmffreq=args.rpm*4/60.
#print("\nbmffreq="+str(bmffreq))
scd=int(1000000000/args.swfreq)
ocd=int(1000000000/bmffreq)
lastt=args.simdur*1000000
paramfn="param"+projname+".txt"
print("PARAM FN=",paramfn)
writeparam(paramfn=paramfn,emfK=args.emfK,R=args.R/1000.0,L=args.L/1000.)
swot=args.swot

ofn=sys.argv[0].split('.')[0]+".log"
log=open(ofn,"a")
state=initState()
res=[[],[],[],[],[],[]]
swot=args.swot
lastt=args.simdur*1000000
for theta1 in range(args.st,args.et,args.ts):
#for step in range(int(255/5)-1,int(255/5)):
    theta=theta1%360
    S,time=initST()
    calS(S,time,swot,theta,ocd,scd,lastt,args.uc)
    #filterS(S,time,swot*2)
    writeStofiles(S,time,args.fn)
    p,pp,ip,op,effa,SWL,RL,oprms,losses=runproj(projname,1.0/bmffreq)
    res[0].append(p)
    res[1].append(pp)
    res[2].append(ip)
    res[3].append(op)
    res[4].append(effa[0])
    res[5].append(effa[1])
    logentry=""
    logentry=logentry+"pn="+args.pn+","
    logentry=logentry+"simdur="+str(args.simdur)+","
    logentry=logentry+"rpm="+str(args.rpm)+","
    logentry=logentry+"swfreq="+str(args.swfreq)+","
    logentry=logentry+"R="+str(args.R)+","
    logentry=logentry+"L="+str(args.L)+","
    logentry=logentry+"scale="+str(args.scale)+","
    logentry=logentry+"uc="+str(args.uc)+","
    logentry=logentry+"emfK="+str(args.emfK)+","
    logentry=logentry+"RemfK="+str(args.RemfK)+","
    logentry=logentry+"emft1="+str(args.bmft1)+","    
    logentry=logentry+"swot="+str(args.swot)+","
    logentry=logentry+"theta="+str(theta)+","
    logentry=logentry+"phase1="+str(p[0])+","
    logentry=logentry+"phase2="+str(p[1])+","
    logentry=logentry+"phase3="+str(p[2])+","
    logentry=logentry+"phase12="+str(pp[0])+","
    logentry=logentry+"phase23="+str(pp[1])+","
    logentry=logentry+"phase31="+str(pp[2])+","

    logentry=logentry+"IPT="+str(ip[0])+","
    logentry=logentry+"IPN="+str(ip[1])+","
    logentry=logentry+"IPP="+str(ip[2])+","
    logentry=logentry+"IPRMS="+str(ip[3])+","

    logentry=logentry+"SWL1="+str(SWL[0])+","
    logentry=logentry+"SWL2="+str(SWL[1])+","
    logentry=logentry+"SWL3="+str(SWL[2])+","
    logentry=logentry+"SWL4="+str(SWL[3])+","
    logentry=logentry+"SWL5="+str(SWL[4])+","
    logentry=logentry+"SWL6="+str(SWL[5])+","
    logentry=logentry+"SWLT="+str(SWL[6])+","

    
    logentry=logentry+"RL1="+str(RL[0])+","
    logentry=logentry+"RL2="+str(RL[1])+","
    logentry=logentry+"RL3="+str(RL[2])+","
    
    logentry=logentry+"OP1T="+str(op[0])+","
    logentry=logentry+"OP1N="+str(op[1])+","
    logentry=logentry+"OP1P="+str(op[2])+","
    logentry=logentry+"OP1RMS="+str(oprms[0])+","
    logentry=logentry+"OP2T="+str(op[3])+","
    logentry=logentry+"OP2N="+str(op[4])+","
    logentry=logentry+"OP2P="+str(op[5])+","
    logentry=logentry+"OP2RMS="+str(oprms[1])+","
    logentry=logentry+"OP3T="+str(op[6])+","
    logentry=logentry+"OP3N="+str(op[7])+","
    logentry=logentry+"OP3P="+str(op[8])+","
    logentry=logentry+"OP2RMS="+str(oprms[2])+","

    if(args.harmonic):
        logentry=logentry+"OP1F="+str(op[9])+","
        logentry=logentry+"OP2F="+str(op[10])+","
        logentry=logentry+"OP3F="+str(op[11])+","
        logentry=logentry+"OP1FRMS="+str(oprms[3])+","
        logentry=logentry+"OP2FRMS="+str(oprms[4])+","
        logentry=logentry+"OP3FRMS="+str(oprms[5])+","
        
    logentry=logentry+"GENR="+str(np.abs((op[1]+op[4]+op[7])/(op[2]+op[5]+op[8])))+","
    logentry=logentry+"EFF="+str(np.abs(effa[0]))+","
    if(args.harmonic):
        logentry=logentry+"EFF2="+str(np.abs(effa[1]))+","
    logentry=logentry+"SWLOSSES="+str(np.abs(losses[0]))+","
    logentry=logentry+"RLOSSES="+str(np.abs(losses[1]))+","
    logentry=logentry+"\n"    
    log.write(logentry)
    log.flush()
    print(logentry)


