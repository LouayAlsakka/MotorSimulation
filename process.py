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
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

def getrecs(db,cond):
    res=[]
    count=0
    for recidx in range(len(db)):
        #print("\nrecidx=:",recidx,len(cond))
        rec=db[recidx]
        match = 1
        for i in range(len(cond)):
            if(cond[i].get("val")!=rec.get(cond[i].get("name"))):
                #print(i)
                match=0
                break
        if(match):
            res.append(rec)
    return res
def getsdbd(sdb):
    sdba={}
    for i in range(len(param)):
        sdba.update({param[i]:[]})
    sdba.update({"EFF3":[]})
    sdba.update({"EFF4":[]})
    sdba.update({"TLP":[]})
    sdba.update({"HLOSSES":[]})
    sdba.update({"GENLOSSES":[]})
    sdba.update({"GENRP":[]})        
    for i in range(len(sdb)):
        rec=sdb[i]
        for j in range(len(param)):
            sdba[param[j]].append(rec.get(param[j]))
        # please note last OP2RMS need to be OP3RMS but there was typo in the script for now this should be very close
        eff3=rec.get("EFF2")*(rec.get("OP1FRMS")+rec.get("OP2FRMS")+rec.get("OP3FRMS"))/(rec.get("OP1RMS")+rec.get("OP2RMS")+rec.get("OP2RMS"))
        sdba["EFF3"].append(eff3)
        if((eff3>0)and (eff3<rec.get("EFF"))):
            sdba["HLOSSES"].append((((rec.get("EFF")-eff3))))
        else:
            sdba["HLOSSES"].append(0)
        #1/3 of generation losses are real losses
        genl=rec.get("GENR")/4*(rec.get("GENR")+1)
        genrp=rec.get("GENR")/(rec.get("GENR")+1)
        sdba["GENLOSSES"].append(100*genl)
        sdba["GENRP"].append(100*genrp)
        eff4=0
        if(eff3>0):
            eff4=eff3*(1-genl)
            if(eff4<0):
                eff4=0
        sdba["EFF4"].append(eff4)
        sdba["TLP"].append(100-eff4)
        
    return sdba

def getBVda(sdb):
    sdba={}
    for i in range(len(BVparam)):
        sdba.update({BVparam[i]:[]})
    for i in range(len(sdb)):
        rec=sdb[i]
        for j in range(len(BVparam)):
            sdba[BVparam[j]].append(rec.get(BVparam[j]))
        
    return sdba

#get dic base on condition
def getsdb(db,cond):
    return(getrecs(db,cond))

def getdc(db,cond):
    sdb=getsdb(db,cond)
    return(getsdbd(sdb))

def argminlast(a):
    if(len(a)==1):
        return(0)
    return(len(a)-np.argmin(a[len(a):0:-1])-1)
def getbrec(d):
    res={}
    bidx=argminlast(d.get("GENR"))
    for key in d:
        res.update({key:d.get(key)[bidx]})
    return(res)


def genBVdb(fn):
    f=open(fn)
    lines=f.readlines()
    db=[]
    for line in lines:
        #print(line)
        rec={}
        for i in range(len(BVparam)):
            #print(i)
            #Process data here, we can use abs, change value for certain keys (like multiply by 100 and so on)
            if(dtype.get(BVparam[i])=="float"):
                rec.update({BVparam[i]:abs(float(line.split(BVparam[i]+"=")[1].split(",")[0]))})
            elif(dtype.get(BVparam[i])=="int"):
                if(BVparam[i]=="theta"):
                    if(int(line.split(BVparam[i]+"=")[1].split(",")[0])>=180):
                        t=int(line.split(BVparam[i]+"=")[1].split(",")[0])-360
                        rec.update({BVparam[i]:t})
                    else:                        
                        rec.update({BVparam[i]:int(line.split(BVparam[i]+"=")[1].split(",")[0])})
                else:
                    rec.update({BVparam[i]:abs(int(line.split(BVparam[i]+"=")[1].split(",")[0]))})
            elif(dtype.get(BVparam[i])=="str"):
                try:
                    rec.update({BVparam[i]:str(line.split(BVparam[i]+"=")[1].split(",")[0])})
                except:
                    print(BVparam[i] ,"not found on: ")
                    print(line)
        print(rec)
        db.append(rec)
    f.close()
    return db

def gendb(fn):
    f=open(fn)
    lines=f.readlines()
    db=[]
    for line in lines:
        #print(line)
        rec={}
        for i in range(len(param)):
            #print(i)
            #Process data here, we can use abs, change value for certain keys (like multiply by 100 and so on)
            if(dtype.get(param[i])=="float"):
                if((param[i]=="SWLOSSES") or (param[i]=="RLOSSES")):
                    rec.update({param[i]:100*abs(float(line.split(param[i]+"=")[1].split(",")[0]))})
                else:
                    rec.update({param[i]:abs(float(line.split(param[i]+"=")[1].split(",")[0]))})
            elif(dtype.get(param[i])=="int"):
                if(param[i]=="theta"):
                    if(int(line.split(param[i]+"=")[1].split(",")[0])>=180):
                        t=int(line.split(param[i]+"=")[1].split(",")[0])-360
                        rec.update({param[i]:t})
                    else:                        
                        rec.update({param[i]:int(line.split(param[i]+"=")[1].split(",")[0])})
                else:
                    rec.update({param[i]:abs(int(line.split(param[i]+"=")[1].split(",")[0]))})
            elif(dtype.get(param[i])=="str"):
                try:
                    rec.update({param[i]:str(line.split(param[i]+"=")[1].split(",")[0])})
                except:
                    print(param[i] ,"not found on: ")
                    print(line)
        #print(rec)
        db.append(rec)
    f.close()
    return db
def checkdb(db,econd=[]):
    bestr=[]
    rpmset={1000,2000,3000,4000,5000,6000}
    #rpmset={6000}
    ucset={1,10,15,20}
    scaleset={10,20,30,40,50,60,70,80,90,100}
    for rpm in rpmset:
        for scale in scaleset:
            for uc in ucset:
                cond=[]
                for c in econd:
                    cond.append(c)                
                cond.append({"name":"rpm","val":rpm})                
                cond.append({"name":"scale","val":scale})
                cond.append({"name":"uc","val":uc})
                d=getdc(db,cond)
                #72 but since now we now where is the good angle we might not rin the whole 360 degree
                if(len(d.get("theta"))< 1):
                    print("RPM:",rpm,"UC:",uc,"Scale:",scale,"LEN!!=",len(d.get("theta")))
                else:
                    bd=getbrec(d)
                    ft=np.min(d.get("theta"))
                    lt=np.max(d.get("theta"))
                    if((bd.get("theta)")==lt) or (bd.get("theta)")==ft)):
                        print("RPM:",rpm,"UC:",uc,"Scale:",scale,"BestTheta=",bd.get("theta"),ft,lt)
                    bestr.append(bd)
    return(bestr)


def ordera(a):
    if(len(a)==0):
        return(a)
    res=[]
    for i in range(len(a)):
        res.append([])
    while(len(a[0])):
        ei=np.argmin(a[0])
        for i in range(len(a)):
            res[i].append(a[i][ei])
        for i in range(len(a)):
            a[i].pop(ei)
    return(res)
#def geteff[]:


def convertcond(cond):
    ocond=[]
    cl=cond.split(",")
    if(cl[0]):
        for i in range(len(cl)):
            key=cl[i].split("=")[0]
            val=cl[i].split("=")[1]
            if(dtype.get(key)=="float"):
                val=float(val)
            elif(dtype.get(key)=="int"):
                val=int(val)
            ocond.append({"name":key,"val":val})
    return(ocond)
            
def getlist(bra,cond,keys):
    keya=[]
    vala=[]
    res=[]
    for i in range(len(keys)):
        res.append([])
    cl=cond.split(",")
    if(cl[0]):
        for i in range(len(cl)):
            key=cl[i].split("=")[0]
            val=cl[i].split("=")[1]
            if(dtype.get(key)=="float"):
                val=float(val)
            elif(dtype.get(key)=="int"):
                val=int(val)
            keya.append(key)
            vala.append(val)
    for i in range(len(bra)):
        match=1
        for j in range(len(keya)):
            if(bra[i].get(keya[j])!= vala[j]):
                match=0
        if(match):
            for kidx in range(len(keys)):
                res[kidx].append(bra[i].get(keys[kidx]))

    res=ordera(res)
    return(res)
#res=getlist(bra,"uc=20,scale=100",["rpm","EFF","EFF2","EFF3","GENR","SWLOSSES","RLOSSES"])
def plotlist(bra,cond,keys):
    keyl=[]
    for i in range(len(keys)):
        if(type(keys[i])==list):
            for j in range(len(keys[i])):
                keyl.append(keys[i][j])
        else:
            keyl.append(keys[i])
    #print(keyl)
    res=getlist(bra,cond,keyl)
    fig,axs=pl.subplots(len(keys)-1,figsize=(12, 10))
    fig.canvas.set_window_title(cond)
    fig.suptitle(cond)
    residx=1
    for i in range(1,len(keys)):        
        axs[i-1].set_xlabel(keys[0])
        axs[i-1].set_ylabel(keys[i])
        if(type(keys[i])==list):
            for subi in range(len(keys[i])):
                axs[i-1].plot(res[0],res[residx])
                #print(residx)
                residx=residx+1
        else:
            axs[i-1].plot(res[0],res[residx])
            #print(i,residx,"\n")
            residx=residx+1 
    pl.show()
def genlog():
    cmd="./gendata.cmd"
    os.system(cmd)

def plot3dres(bra,rpml=[1000,2000,3000,4000,5000,6000],scalel=[10,20,30,40,50,60,70,80,90,100],cond="uc=20",zp="EFF4"):

    X, Y = np.meshgrid(scalel,rpml)
    Z=np.empty((len(rpml),len(scalel)))
    x=-1
    for rpm in rpml:
        x=x+1
        y=-1
        for scale in scalel:
            y=y+1
            cond2=cond+",rpm="+str(rpm)+",scale="+str(scale)
            print(cond2)
            l=getlist(bra,cond2,[zp])[0][0]            
            Z[x][y]=l


    fig = pl.figure()
    title=zp+" for: "+cond
    fig.canvas.set_window_title(title)
    fig.suptitle(title)

    ax = fig.gca(projection='3d')
    



    # Plot the surface.
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
    ax.set_ylabel("rpm")
    ax.set_xlabel("Torque")
    ax.set_zlabel(zp)
    '''
    # Customize the z axis.
    ax.set_zlim(-1.01, 1.01)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    '''
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    pl.show()
    return Z

def plot3dresd(bra,rpml=[1000,2000,3000,4000,5000,6000],scalel=[10,20,30,40,50,60,70,80,90,100],cond=["uc=20","uc=20"],zp=["EFF","EFF2"]):

    X, Y = np.meshgrid(scalel,rpml)
    Z=np.empty((len(rpml),len(scalel)))
    Z2=np.empty((len(rpml),len(scalel)))
    x=-1
    for rpm in rpml:
        x=x+1
        y=-1
        for scale in scalel:
            y=y+1
            if(len(cond[0])):
                cond1=cond[0]+",rpm="+str(rpm)+",scale="+str(scale)
            else:
                cond1=cond[0]+"rpm="+str(rpm)+",scale="+str(scale)
            if(len(cond[1])):
                cond2=cond[1]+",rpm="+str(rpm)+",scale="+str(scale)
            else:
                cond2=cond[1]+"rpm="+str(rpm)+",scale="+str(scale)
            print(cond2)
            l1=getlist(bra[0],cond1,[zp[0]])[0][0]
            l2=getlist(bra[1],cond2,[zp[1]])[0][0]            
            Z[x][y]=l1
            Z2[x][y]=l2

    fig = pl.figure()
    print(cond[0])
    title=zp[0]+" for: "+cond[0] +" Vs " +zp[1]+ " On:" +cond[1]
    fig.canvas.set_window_title(title)
    fig.suptitle(title)

    ax = fig.gca(projection='3d')
    



    # Plot the surface.
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
    surf2 = ax.plot_surface(X, Y, Z2, cmap=cm.Spectral,
                       linewidth=0, antialiased=False)
    #fig.scene.renderer.use_depth_peeling = 1
    ax.set_ylabel("rpm")
    ax.set_xlabel("Torque")
    ax.set_zlabel(zp)
    '''
    # Customize the z axis.
    ax.set_zlim(-1.01, 1.01)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    '''
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    fig.colorbar(surf2, shrink=0.5, aspect=5)

    pl.show()
    return Z

def plot3dresn(bra,cond,zp,rpml=[1000,2000,3000,4000,5000,6000],scalel=[10,20,30,40,50,60,70,80,90,100]):
    X, Y = np.meshgrid(scalel,rpml)
    n=len(bra)
    if(n<=3):
        nx=1
        ny=3
    elif(n<=4):
        nx=2
        ny=2
    elif(n<=6):
        nx=2
        ny=3
    elif(n<=9):
        nx=3
        ny=3
    else:
        print("Not supported!")
        return
    Z=[]
    for i in range(n):
        Z.append(np.empty((len(rpml),len(scalel))))
    x=-1
    for rpm in rpml:
        x=x+1
        y=-1
        for scale in scalel:
            y=y+1
            for i in range(n):
                if(len(cond[i])>0):
                    ncond=cond[i]+","
                else:
                    ncond=""
                print(i,"::",cond[i])
                ncond=ncond+"rpm="+str(rpm)+",scale="+str(scale)
                print(ncond)
                print(zp[i])
                Z[i][x][y]=getlist(bra[i],ncond,[zp[i]])[0][0]


    fig = pl.figure()
    #title=zp+" for: "+cond[0]
    title=zp[0]+" for: "
    fig.canvas.set_window_title(title)
    fig.suptitle(title)
    ax=[]
    #ax = fig.gca(projection='3d')
    for i in range(n):
        print(nx,ny,i)
        ax.append(fig.add_subplot(nx,ny,i+1,projection='3d'))
        # Plot the surface.
        surf = ax[i].plot_surface(X, Y, Z[i], cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
        ax[i].set_ylabel("rpm")
        ax[i].set_xlabel("Torque")
        ax[i].set_zlabel(zp[i])
        ax[i].title.set_text(cond[i])
        # Add a color bar which maps values to colors.
        fig.colorbar(surf, shrink=0.5, aspect=5)
    pl.show()
    return Z


def printbest():    
    for rpm in [3000,4000,5000,6000]:
        for scale in [10,20,30,40,50,60,70,80,90,100]:
            for uc in [1,10,15,20]:
                cond="uc="+str(uc)+",rpm="+str(rpm)+",scale="+str(scale)
                theta=getlist(bra,cond,["theta"])[0][0]
                if(theta<0):
                    theta=360+theta
                precmd="python3 gen6s_3.py --swfreq 10000 --swot 1000 --swoffv 0  --emfK 2.5 --RemfK 100 --R 200 --ts 5 --harmonic 1 --simdur 400	"

                cmd=precmd+"--uc "+str(uc)+" --rpm "+str(rpm)+" --scale "+str(scale) +" --st "+ str(theta)+ " --et "+ str(theta+5)
                print(cmd)


def printbest5(fn=""):
    if(len(fn)):
        f=open(fn,"w")
    for rpm in [1000,2000,3000,4000,5000,6000]:
        for scale in [10,20,30,40,50,60,70,80,90,100]:
            for uc in [1,10,15,20]:
                cond="uc="+str(uc)+",rpm="+str(rpm)+",scale="+str(scale)
                #print(cond)
                theta=getlist(bra5,cond,["theta"])[0][0]
                if(theta<0):
                    theta=360+theta
                  
                precmd=  "python3 gen6s_4.py --swonv 560 --swfreq 10000 --swot 400 --swoffv 0  --emfK 1 --RemfK 100  --R 200  --ts 5  --harmonic 1 --pn motor5 --vbus 550 --simdur 800 "

                cmd=precmd+"--uc "+str(uc)+" --rpm "+str(rpm)+" --scale "+str(scale) +" --st "+ str(theta)+ " --et "+ str(theta+5)
                print(cmd)
                if(len(fn)):
                    cmd=cmd+"\n"
                    f.write(cmd)
    if(len(fn)):
        f.close()
                    
param=["pn","simdur","rpm","swfreq","R","L","scale","uc","emfK","RemfK","swot","theta","phase1","phase2","phase3","phase12","phase23","phase31","IPT","IPN","IPP","IPRMS","SWL1","SWL2","SWL3","SWL4","SWL5","SWL6","SWLT","RL1","RL2","RL3","OP1T","OP1N","OP1P","OP1RMS","OP2T","OP2N","OP2P","OP2RMS","OP3T","OP3N","OP3P","OP2RMS","OP1F","OP2F","OP3F","OP1FRMS","OP2FRMS","OP3FRMS","GENR","EFF","EFF2","SWLOSSES","RLOSSES"]
    #dtype={"pn":"str","simdur":"int","rpm":"int","swfreq":"int"}
dtype={"pn":"str","simdur":"int","rpm":"int","swfreq":"int","R":"float","L":"float","scale":"int","uc":"int","emfK":"float","RemfK":"float","swot":"int","theta":"int","phase1":"float","phase2":"float","phase3":"float","phase12":"float","phase23":"float","phase31":"float","IPT":"float","IPN":"float","IPP":"float","IPRMS":"float","SWL1":"float","SWL2":"float","SWL3":"float","SWL4":"float","SWL5":"float","SWL6":"float","SWLT":"float","RL1":"float","RL2":"float","RL3":"float","OP1T":"float","OP1N":"float","OP1P":"float","OP1RMS":"float","OP2T":"float","OP2N":"float","OP2P":"float","OP2RMS":"float","OP3T":"float","OP3N":"float","OP3P":"float","OP2RMS":"float","OP1F":"float","OP2F":"float","OP3F":"float","OP1FRMS":"float","OP2FRMS":"float","OP3FRMS":"float","GENR":"float","EFF":"float","EFF2":"float","SWLOSSES":"float","RLOSSES":"float","IP1":"float","IP2":"float","IP3":"float","IPT":"float","OP1":"float","OP2":"float","OP3":"float","OPT":"float","RL1":"float","RL2":"float","RL3":"float","RLT":"float","PRL":"float","I":"float","EFF":"float","emfK":"float","RemfK":"float","theta":"int"}
fn="data.txt"



BVparam=["pn","simdur","R","L","emfK","RemfK","theta","rpm","scale","IP1","IP2","IP3","IPT","OP1","OP2","OP3","OPT","RL1","RL2","RL3","RLT","PRL","I","EFF"]
#BVdtype={"pn":"str","simdur":"int"]
#BVdtype={"pn":"str","simdur":"int","rpm":"int","R":"float","L":"float","scale":"int","IP1":"float","IP2":"float","IP3":"float","IPT":"float","OP1":"float","OP2":"float","OP3":"float","OPT":"float","RL1":"float","RL2":"float","RL3":"float","RLT":"float","PRL":"float","I":"float","EFF":"float","emfK":"float","RemfK":"float","theta":"int"}


#for traditional
fn="data.txt"
fn5="data5.txt"
BVfn="BVdata.txt"
BVfn5="BVdata5.txt"
#only enable this when we want to gen the log again
#genlog()
db=gendb(fn)
db5=gendb(fn5)

bra=checkdb(db)
bra5=checkdb(db5,econd=convertcond("simdur=800"))
#bra5=checkdb(db5,econd=[])
BVdb=genBVdb(BVfn)
BVdb5=genBVdb(BVfn5)
#BVda=getBVda(BVdb)

'''
plotlist(bra,"uc=20,scale=100",["rpm",["EFF","EFF2","EFF3"],"GENR","SWLOSSES","RLOSSES"])             
scale,eff=getlist(bra,"uc=10,rpm=1000",["scale","EFF"])
scale,eff=getlist(BVdb,"rpm=1000",["scale","EFF"])
plot3dresn([bra,bra,bra,bra,bra,BVdb],["uc=20","uc=20","uc=20","uc=20","uc=20",""],["RLOSSES","SWLOSSES","HLOSSES","GENLOSSES","TLP","PRL"])
plot3dresn([bra,bra,bra,bra],["uc=20","uc=20","uc=20","uc=20"],["EFF","EFF2","EFF3","EFF4"])
plot3dresd([bra,BVdb],cond=["uc=20","pn=BV"],zp=["EFF4","EFF"])
plot3dresd([bra,BVdb],cond=["uc=10","pn=BV"],rpml=[1000,2000,3000,4000,5000,6000],zp=["OP1T","OP1"])


plot3dres(BVdb,cond="pn=BV",zp="EFF")




plotlist(bra5,"uc=20,scale=100",["rpm",["EFF","EFF2","EFF3"],"GENR","SWLOSSES","RLOSSES"])             
scale,eff=getlist(bra5,"uc=10,rpm=1000",["scale","EFF"])
scale,eff=getlist(BVdb5,"rpm=1000",["scale","EFF"])
plot3dresn([bra5,bra5,bra5,bra5,bra5,BVdb5],["uc=20","uc=20","uc=20","uc=20","uc=20",""],["RLOSSES","SWLOSSES","HLOSSES","GENLOSSES","TLP","PRL"])
plot3dresn([bra5,bra5,bra5,bra5],["uc=20","uc=20","uc=20","uc=20"],["EFF","EFF2","EFF3","EFF4"])
plot3dresd([bra5,BVdb5],cond=["uc=20","pn=BV"],zp=["EFF4","EFF"])
plot3dresd([bra5,BVdb5],cond=["uc=10","pn=BV"],rpml=[1000,2000,3000,4000,5000,6000],zp=["OP1T","OP1"])
plot3dres(BVdb5,cond="pn=BV",zp="EFF")

'''                
                
