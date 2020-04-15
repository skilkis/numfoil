# -*- coding: utf-8 -*-
"""
Created on Fri Jan 25 11:13:22 2019

@author: Mike
"""
import math
import numpy as np
import matplotlib.pyplot as plt
def Yc(x, Mc, Pc): #camber line
    if x >= 0 and x < Pc:
        return (Mc/(Pc**2)*(2.0*Pc*x - x**2))
    elif x >= Pc and x <= 1:
        return (Mc/((1- Pc)**2)*(1 - 2.0*Pc + 2*Pc*x - x**2))
    
def dYc(x, Mc, Pc): #gradient
    if x >= 0 and x < Pc:
        return (Mc/(Pc**2)*(2.0*Pc - 2*x))
    elif x >= Pc and x <= 1:
        return (Mc/((1- Pc)**2)*(2.0*Pc - 2*x))
    
def Yt(T, x): #thickness
    a0 = 0.2969
    a1 = -.126
    a2 = -.3516
    a3 =  .2843
    a4 = -.1036 #this is for closed trailing edge, -0.1015 for open trailing edge, -.1036 for closed
    
    return T/(0.2)*(a0*x**(0.5) + a1*x + a2*x**2 + a3*x**3 + a4*x**4)

def Theta(dyc): #angle of camberline
    return math.atan(dyc)

def Upx(xc, yt, theta): #upper x of airfoil
    return xc - yt*math.sin(theta)

def Lox(xc, yt, theta): #lower x of airfoil
    return xc + yt*math.sin(theta)

def Upy(yc, yt, theta): #upper y of airfoil
    return yc + yt*math.cos(theta)

def Loy(yc, yt, theta): #lower y of airfoil
    return yc - yt*math.cos(theta)

def Spacing(beta): #convert the spacing from pi based to 1, giving a good spacing for points along the curve
    space = []
    for i in xrange(0, len(beta), 1):
        space.append((1.0 - math.cos(beta[i]))/2.0)
        #space.append((.5 - .5*math.cos(beta[i])))
    return space

def Surf(X1, X2, Y1, Y2):
    xv = X2 - X1
    yv = Y2 - Y1
    return math.sqrt(xv**2 + yv**2)

def Orient(X1, X2, Y1, Y2):
    xv = X2 - X1
    yv = Y2 - Y1
    Hv = math.sqrt(xv**2 + yv**2)
    return math.acos(xv/Hv)

def Cramer4(A, RHS, N):
    CC = np.zeros(shape = (N, N))
    den = np.linalg.det(A)*1.0
    #print "this is A: ", den
    X = np.zeros(N)
    for k in xrange(0, N, 1):
        for i in xrange(0, N, 1):
            for j in xrange(0, N, 1):
                CC[i, j] = A[i, j]
                
        for m in xrange(0, N, 1):
            CC[m, k] = RHS[m]
        X[k] = np.linalg.det(CC)/den
    return X
    

def Cramer5(Array, RHS):
    return np.linalg.solve(Array, RHS)

def Lift(point, MAX_CAM, POS_CAM, THICKNESS, ANGLE_OF_ATTACK,V_inf):
    Xpos = []
    Ypos = []
    Xpos.append(1.0)
    Ypos.append(0.0)

    #point = 100 #number of points
    #MAX_CAM = 0 #max camber 
    #POS_CAM = 0 #location of max camber
    #THICKNESS = 12 #Max thickness t/c
    #ANGLE_OF_ATTACK = 2 #angle of attack
    
    
    Mc = MAX_CAM/100.0
    Pc = POS_CAM/10.0
    T = THICKNESS/100.0
    
    Mair = point/2 #split into 2, for top and bottom
    Mp = point + 1
    Mm = point
    RHS = np.zeros(Mp)
    SIN = np.zeros(Mp)
    COS = np.zeros(Mp)
    
    
    
    
    Uni = np.linspace(0, math.pi, Mair+1)
    Xc = Spacing(Uni)
    Y_Cam = []
    Y_Grad = []
    Y_Thick = []
    theta = []
    X_upper = []
    X_lower = []
    Y_upper = []
    Y_lower = []
    
    
    alpha = ANGLE_OF_ATTACK*math.pi/180.0
    
    for i in xrange(0, Mair+1, 1):
        Y_Cam.append(Yc(Xc[i], Mc, Pc))
        Y_Grad.append(dYc(Xc[i], Mc, Pc))
        Y_Thick.append(Yt(T, Xc[i]))
        theta.append(Theta(Y_Grad[i]))  
        X_upper.append(Upx(Xc[i], Y_Thick[i], theta[i]))
        X_lower.append(Lox(Xc[i], Y_Thick[i], theta[i]))
        Y_upper.append(Upy(Y_Cam[i], Y_Thick[i], theta[i]))
        Y_lower.append(Loy(Y_Cam[i], Y_Thick[i], theta[i]))
        
    
    plt.plot(Xc, Y_upper)
    plt.plot(Xc, Y_lower)
    plt.axis([0, 1 , -.25, .25])
    plt.show()

    V_X = V_inf*math.cos(alpha)
    V_Y = V_inf*math.sin(alpha)
    
    Paneltop = []
    Panelbot = []
    S_top = []
    S_bot = []
    O_top = []
    O_bot = []
    X_mid_top = []
    X_mid_bot = []
    Y_mid_top = []
    Y_mid_bot = []
    
    for i in xrange(0, Mair, 1):
        S_top.append(Surf(X_upper[i], X_upper[i+1], Y_upper[i], Y_upper[i+1]))
        S_bot.append(Surf(X_lower[i], X_lower[i+1], Y_lower[i], Y_lower[i+1]))
        O_top.append(Orient(X_upper[i], X_upper[i+1], Y_upper[i], Y_upper[i+1]))
        O_bot.append(Orient(X_lower[i], X_lower[i+1], Y_lower[i], Y_lower[i+1]))
        
        X_mid_top.append(0.5*(X_upper[i] + X_upper[i + 1]))
        X_mid_bot.append(0.5*(X_lower[i] + X_lower[i + 1]))
        
        Y_mid_top.append(0.5*(Y_upper[i] + Y_upper[i + 1]))
        Y_mid_bot.append(0.5*(Y_lower[i] + Y_lower[i + 1]))
    
    Or_sin_top = []
    Or_sin_bot = []
    
    CN1 = np.zeros(shape = ((Mm), (Mm)))
    CN2 = np.zeros(shape = ((Mm), (Mm)))
    CT1 = np.zeros(shape = ((Mm), (Mm)))
    CT2 = np.zeros(shape = ((Mm), (Mm)))
    
    for i in xrange(0, Mair, 1):
        Or_sin_top.append(math.sin(O_top[i] - alpha))
        Or_sin_bot.append(math.sin(O_bot[i] - alpha))
        
    
    
    Opos = []
    SinPos = []
    Xm = []
    Ym = []
    St = []
    #print "Length 1: ", len(O_bot)
    #print Mm/2 - 1
    for i in xrange((Mair-1) , 0 , -1):
        #print i
        
        Xpos.append(X_lower[i])
        Ypos.append(Y_lower[i])
        SinPos.append(Or_sin_bot[i])
    
        
        
    for i in xrange((Mair-1) , -1 , -1):
        Xm.append(X_mid_bot[i])
        Ym.append(Y_mid_bot[i])
        St.append(S_bot[i])
        Opos.append(O_bot[i])
        
    for i in xrange(0, Mair, 1):
        Opos.append(O_top[i])
        Xpos.append(X_upper[i])
        Ypos.append(Y_upper[i])
        SinPos.append(Or_sin_top[i])
        Xm.append(X_mid_top[i])
        Ym.append(Y_mid_top[i])
        St.append(S_top[i])
    
    Xpos.append(1.0)
    Ypos.append(0.0)
    Opos.append(0)
    Boold = False
    for i in xrange(0, Mp-1, 1):
        
        #print Xpos[i+1]- Xpos[i]
        #print i+1, i
        #print " "
        Temp = np.arctan2((Ypos[i+1] - Ypos[i]), (Xpos[i+1]*1.0 - Xpos[i]*1.0))
        if Temp == 100000:
            Boold = True
            
        if Boold == True:
            Opos[i] = np.arctan2((Ypos[i+2] - Ypos[i+1]), (Xpos[i+2]*1.0 - Xpos[i+1]*1.0))
        else:
            Opos[i] = np.arctan2((Ypos[i+1] - Ypos[i]), (Xpos[i+1]*1.0 - Xpos[i]*1.0))
        Opos[i] = np.arctan2((Ypos[i+1] - Ypos[i]), (Xpos[i+1]*1.0 - Xpos[i]*1.0))
    
    for i in xrange(0, Mp-1, 1):
        RHS[i] = math.sin(Opos[i] - alpha)
        SIN[i] = math.sin(Opos[i])
        COS[i] = math.cos(Opos[i])
        
        
    for i in xrange(0, Mp-1, 1):
        for j in xrange(0, Mp-1, 1):
            if i == j:
                CN1[i, j] = -1.0
                CN2[i, j] = 1.0
                CT1[i, j] = 0.5*math.pi
                CT2[i, j] = 0.5*math.pi
            else:
                A = -(Xm[i] - Xpos[j])*COS[j] - (Ym[i] - Ypos[j])*SIN[j]
                B = (Xm[i] - Xpos[j])**2 + (Ym[i] - Ypos[j])**2
                C = math.sin(Opos[i] - Opos[j]*1.0)
                D = math.cos(Opos[i] - Opos[j]*1.0)
                E = (Xm[i]-Xpos[j])*SIN[j] - (Ym[i]-Ypos[j])*COS[j]
                F = np.log(1.0 + St[j]*(St[j] + 2.0*A)/B)
                G = np.arctan2(E*St[j], (B + A*St[j]))
                P = (Xm[i] - Xpos[j])*math.sin(Opos[i] - 2.0*Opos[j]) + (Ym[i] - Ypos[j])*math.cos(Opos[i] - 2.0*Opos[j])
                Q = (Xm[i] - Xpos[j])*math.cos(Opos[i] - 2.0*Opos[j]) - (Ym[i] - Ypos[j])*math.sin(Opos[i] - 2.0*Opos[j])
                
                
                CN2[i, j] = (D + 0.5*Q*F/St[j] - (A*C + D*E)*G/St[j])
                CN1[i, j] = (0.5*D*F + C*G - CN2[i, j])
                CT2[i, j] = C + 0.5*P*F/St[j] + (A*D - C*E)*G/St[j]
                CT1[i, j] = 0.5*C*F - D*G - CT2[i, j]
    
    
    An = np.zeros(shape = ((Mp), (Mp)))
    At = np.zeros(shape = ((Mp), (Mp)))
    MP = point
    for i in xrange(0, (Mm), 1):
        An[i, 0] = CN1[i, 0]
        An[i, (MP)] = CN2[i, Mm -1]
        At[i, 0] = CT1[i, 0]
        At[i, (MP)] = CT2[i, Mm -1]     
        for j in xrange(1,  (Mm), 1):
            An[i, j] = CN1[i, j] + CN2[i, j - 1]
            At[i, j] = CT1[i, j] + CT2[i, j - 1]
            An[MP, j] = 0
    An[MP, MP] = 1.0
    An[MP, 0] = 1.0
    RHS[MP] = 0.0
    
    for i in xrange(1,  Mm, 1):
        An[MP, i] = 0.0
    
    

    
    
    #print np.linalg.det(An)
    
    CP1 = np.zeros(Mp-1)
    #print Mp-1
    Gamma = Cramer5(An, RHS)
    #print Gamma
    #print "gap"
    Vel = np.zeros(Mp-1)
    #print "#    Xm        Ym       Opos     Surf      Gamma     Vel      CP"
    Angle = np.zeros(Mp -1)
    SumAG = np.zeros(Mp)
    for i in xrange(0,  Mp-1, 1):
        Vel[i] = math.cos(Opos[i]  - alpha)
        for j in xrange(0, Mp, 1):
            Vel[i] = Vel[i] + At[i, j]*Gamma[j]
        CP1[i] = 1.0 - Vel[i]**2
        #print i,"  ", np.round(Xm[i], 3),"  ", np.round(Ym[i], 3),"  ", np.round(Opos[i], 3),"  ", np.round(St[i], 3),"  ", np.round(Gamma[i], 3),"  ", np.round(Vel[i], 3),"  ", np.round(CP1[i], 3)
    
    j = Mp-1
    i = Mm-1
    #print j,"  ", "    ","  ",  "         ","  ",  "    ","  ", "    ","  ", np.round(Gamma[j], 3),"  ",  "    ","  ",  "    "
    

    sum41 = np.zeros(Mp)
    for i in xrange(0, Mp, 1):
        for j in xrange(0, Mp, 1):
            sum41[i] =  An[i, j]*Gamma[j] + sum41[i]
    Cl = 0
    Cd = 0
    Xs = np.zeros(Mm/2)
    Lift = np.zeros(Mp-2)
    Ys = np.zeros(Mm/2)
    Drag = np.zeros(Mp-2)
    CPyu = np.zeros(Mp-2)
    CPyd = np.zeros(Mp-2)
    Yu = np.zeros(Mp-2)
    Yd = np.zeros(Mp-2)
    for i in xrange(0, Mm/2, 1):
        Xs[i] = abs(Xc[i + 1] - Xc[i])
        Ys[i] = abs(Y_Cam[i + 1] - Y_Cam[i])
    for i in xrange(0, Mm/2, 1):
        Lift[i] = (CP1[i] - CP1[Mm-1-i])*Xs[i]
        Drag[i] = CP1[i]*Ys[i] + CP1[Mm-1-i]*Ys[i]
        Cl = Lift[i] + Cl
        Cd = Drag[i] + Cd
    for i in xrange(0, Mm/2, 1):
        Y = np.abs(Ym[i])
        if Y < np.max(Ym):
            CPyu[i] = CP1[i]*np.abs(Ym[i])
            
        elif Y >= np.max(Ym):
            CPyd[i] = CP1[i]*np.abs(Ym[i])
    for i in xrange(Mm/2, Mm-1, 1):
        Y = np.abs(Ym[i])
        if Y < np.max(Ym):
            CPyu[i] = CP1[i]*np.abs(Ym[i])
            
        elif Y >= np.max(Ym):
            CPyd[i] = CP1[i]*np.abs(Ym[i])   
    
    Cd = np.sum(CPyu) - np.sum(CPyd)
    size(CP1)

    return CP1, Xm, Cl
a,b,c= Lift(100,0,0,12,0,12.26)
print(np.size(a))
Samples = 1
Points = 100
alpha1 = np.zeros((Samples)) 

CP = np.zeros((Samples,Points))   #[[0 for x in range(Points)] for y in range(Samples)] 
Xm = np.zeros((Samples,Points))
Cl =  np.zeros((Samples))
for i in xrange(0,Samples,1):
    alpha1[i] = i-(Samples-1)/2
    CP[i,:], Xm[i,:], Cl[i] = Lift(100,6,6,12,-alpha1[i],12.26)

#plt.plot(Xm, -CP, color = 'b', label = "NACA 0012")
#plt.legend(loc="lower right")
#plt.plot([0,1], [0,0])
#plt.axis([0, 1 , np.min(-CP), np.max(-CP)])
#plt.show()



#plt.plot(alpha1, -Cl, color = 'm', label = "NACA 6612")
#plt.legend(loc="lower right")
#plt.plot([0,1], [0,0])
#plt.axis([np.min(alpha1), np.max(alpha1) , np.min(-Cl), np.max(-Cl)])
#plt.show()