#!/usr/bin/env python
# coding: utf-8

# In[34]:

import os 

import numpy as np
from matplotlib import pyplot, cm
#get_ipython().run_line_magic('matplotlib', 'inline')

from tqdm import tqdm
import time

import sys
np.set_printoptions(threshold=sys.maxsize)

# Set the font family and size to use for Matplotlib figures.
pyplot.rcParams['font.family'] = 'serif'
pyplot.rcParams['font.size'] = 16


# In[35]:

os.chdir("C:/Users/Indrajeet Saji/git/numerical-mooc/my_work/developing_one_stop_solun")

g0=4            #Bacterial growth Parameter
gamma=16        #Bacterial growth Parameter
k_value=42.5    #Bacterial growth Parameter

Dc=1e-3     #Diffusion Coefficient of Nutrient 
Db=.21      #Diffusion Coefficient of Bacterial Colony dispersion

t_ref=20    #Reference time (time when death of bacterial cells initiate)
Xe_value=.3       #Cut-off population density

X0=1e-3     #Reference population density for demarcating colony boundary

C0=15       #Initial nutrient concentration (15g/L)
C_thresh_value=1.5

d=20 #separation between inoculum strictly in mm only; check for 5mm, 10mm, 20mm
rho=2.5
mu=40

dx=dy = 0.25

print("Current time is: ", end="")
print(time.asctime( time.localtime(time.time())))

gridx,gridy=159,159    # should be odd numbers, keep both numbers equal e.g. 369,369 as per paper
dx=0.25*(369/gridx)
dy=0.25*(369/gridy)


# In[36]:


X1=np.zeros((gridx,gridy),dtype=np.float64)   # 2-D array 
X2=np.zeros((gridx,gridy),dtype=np.float64)
Xe=np.zeros((gridx,gridy),dtype=np.float64)
C=np.zeros((gridx,gridy),dtype=np.float64)
m=np.zeros((gridx,gridy))
C_thresh=np.zeros((gridx,gridy))
k=np.zeros((gridx,gridy))

#(taking x,y for coordinate positions) :
x=np.linspace(-int(gridx/2),int(gridx/2),num=gridx,dtype=int)  #grid positions --> e.g. for 369 grids, numbering would be -184,-183,..,0,..,183,184
y=np.linspace(-int(gridy/2),int(gridy/2),num=gridy,dtype=int)  
x0=np.where(x==0)[0][0]   # x-coordinate of position of Origin 
y0=np.where(y==0)[0][0]
xl=np.linspace(-184*.25,184*.25,num=len(x))      # x-coordinate of each grid point (gives distance from origin in x direction)       
yl=np.linspace(-184*.25,184*.25,num=len(y))

############# Defining Initial Conditions #############
# for X1 and X2
for i in range(len(xl)):
    for j in range(len(yl)):
        if  ((xl[i]-d/2)**2 + yl[j]**2 <= rho**2):
            X1[i,j] = X0
        
        if  (xl[i]+d/2)**2 + yl[j]**2 <= rho**2:
            X2[i,j] = X0


                                                                       #np.savetxt('output.txt',X1,fmt='%.4e')

#Initializing parameters in array for all positional coordinates on 2D grid:
for i in range(len(x)):
    for j in range(len(y)):
        C[i,j]=C0
        C_thresh[i,j]=C_thresh_value
        k[i,j]=k_value
        Xe[i,j]=Xe_value


print("Initializing Initial Conditions has been completed")
#######################################################        
print("Current time is: ", end="")
print(time.asctime( time.localtime(time.time())))


# In[37]:


def H_t(var,value):        # heavy side step function for time & Xe, not for matrix output
    if var-value <0:
        return 0
    if var-value==0:
        return 0.5
    else: return 1

    
def H_pos(mat1,mat2):      # heavy side step function for position, for matrix output of heavy side function values at each grid point
    
    mat3=np.zeros((max(mat1.shape),min(mat1.shape)))
    
    for i in range(max(mat1.shape)):
        for j in range(min(mat1.shape)):
            if mat1[i,j]-mat2[i,j]<0:
                mat3[i,j]=0
            if mat1[i,j]-mat2[i,j]==0:
                mat3[i,j]=0.5
            else: mat3[i,j]=1
    return mat3

# def H_xe(Xe,mat2):      # heavy side step function for position, for matrix output of Mortality term which only activates at Origin
   
    # mat3=np.zeros((mat2.shape[0],mat2.shape[1]))
    
    # if Xe-mat2[0,0] < 0:
            # mat3[int(mat3.shape[0]/2),int(mat3.shape[0]/2)]=0
    # if Xe-mat2[0,0] == 0:
             # mat3[int(mat3.shape[0]/2),int(mat3.shape[0]/2)]=0.5
    # if Xe-mat2[0,0] > 0:
            # mat3[int(mat3.shape[0]/2),int(mat3.shape[0]/2)]=1
    # return mat3


def growth(nt,time_length):  # to be given number of time steps
    
    # nt : number of time-steps -- to be made 5001
    # time_length in hours: duration of time upto which you wish to see the simulation --- to be made 48 hours

    dt=time_length/nt       # time-step size
    print("The growth for time by time step %f for %f hour time length has been completed for:-" %(dt,time_length))
    
    sigma=Db*dt*((1/dx**2)+(1/dy**2))   #Stability Condition
    print("Value of sigma is %f" %sigma)
    if sigma <0.5:
        print("Sigma is less than 0.5, so, solution would be STABLE")
    else: print("Sigma is not less than 0.5, so, solution would be UNSTABLE")
    
    for n in range(nt):
        
        g_c=g0*np.divide(C,np.add(k,C))
        
        m[:,:]=mu*np.multiply(X1[:,:],X2[:,:])*H_pos(Xe,X1+X2)*H_t((n+1)*dt,t_ref)
        X1[1:-1,1:-1]=X1[1:-1,1:-1]+dt*((np.multiply(g_c[1:-1,1:-1],X1[1:-1,1:-1]))+Db*np.multiply(H_pos(C[1:-1,1:-1],C_thresh[1:-1,1:-1]),(((X1[2:,1:-1]-2*X1[1:-1,1:-1]+X1[:-2,1:-1])/dx**2) + ((X1[1:-1,2:]-2*X1[1:-1,1:-1]+X1[1:-1,:-2])/dy**2))) - m[1:-1,1:-1])
        X2[1:-1,1:-1]=X2[1:-1,1:-1]+dt*((np.multiply(g_c[1:-1,1:-1],X2[1:-1,1:-1]))+Db*np.multiply(H_pos(C[1:-1,1:-1],C_thresh[1:-1,1:-1]),(((X2[2:,1:-1]-2*X2[1:-1,1:-1]+X2[:-2,1:-1])/dx**2) + ((X2[1:-1,2:]-2*X2[1:-1,1:-1]+X2[1:-1,:-2])/dy**2))) - m[1:-1,1:-1])
        
        C[1:-1,1:-1]= C[1:-1,1:-1] +dt*(-gamma*(np.multiply(g_c[1:-1,1:-1],np.add(X1[1:-1,1:-1],X2[1:-1,1:-1]))) + Dc*(((C[2:,1:-1]-2*C[1:-1,1:-1]+C[:-2,1:-1])/dx**2) + ((C[1:-1,2:]-2*C[1:-1,1:-1]+C[1:-1,:-2])/dy**2)))

        ############# Start: Boundary Conditions ######################
        
        #Neumann Condition: Normal component of Bacterial density on boundary is zero
        
        X1[:][-1] = X1[:][-2]
        X1[:][0]  = X1[:][1] 
        X1[-1][:] = X1[-2][:]
        X1[0][:]  = X1[1][:] 

        X2[:][-1] = X2[:][-2]
        X2[:][0]  = X2[:][1] 
        X2[-1][:] = X2[-2][:]
        X2[0][:]  = X2[1][:] 

        
        
        #Dirichlet condition: Fixed and Unchanging Concentration of growth media on boundary
        C[0][:]=C[-1][:]=C[:][0]=C[:][-1]=C0
        
        ############# End: Boundary Conditions ######################

        ############ Start: Saving history of Bacterial Density and Growth Media Concentration #########
        
        X1_hist[n+1][:][:]=X1[:][:]
        X2_hist[n+1][:][:]=X2[:][:]
        C_hist[n+1][:][:]=C[:][:]
        
        ############ End: Saving history of Bacterial Density and Growth Media Concentration #########
        
            
        if n%100==0:
            print("%d steps, Wall clock time: "  %n, end="")
            print(time.asctime( time.localtime(time.time())))
        
        
nt=5

	#In case you want to move with the sigma value and cross the value of 0.5 for unstable solution 
		#sigma=1   # Enter the value of stability parameter ( <0.5 for stable, >=0.5 for unstable)   
		#dt=sigma * ( 1/((1/dx**2)+(1/dy**2)) ) * (1/Db)                                           
        #time_length= nt*dt 

time_length=24         #(48/5001)*nt # 48 hr simulation was carried out in paper

X1_hist=np.zeros((nt+1,gridx,gridy))
X2_hist=np.zeros((nt+1,gridx,gridy))
C_hist=np.zeros((nt+1,gridx,gridy))

X1_hist[0][:][:]=X1[:][:]
X2_hist[0][:][:]=X2[:][:]
C_hist[0][:][:]=C[:][:]

growth(nt,time_length)

print("All the step calculations for growth have been completed for total steps %d" %nt)  
print("Wall clock time: " ,end="")
print(time.asctime( time.localtime(time.time())))


# In[40]:

# Creating a sub-folder to save current output there

new_folder='Db'+str(Db)+' d'+str(d)+'mm'+' time_length'+str(int(time_length))
os.mkdir(new_folder)

os.chdir(new_folder)

# Starting the figure printing process
print("Plotting figures for X1 & X2 growth")
for i in np.linspace(0,nt,num=5,dtype=int):
    fig=pyplot.figure(figsize=(9.2,9.2))

    pyplot.xlabel('x')
    pyplot.ylabel('y')
    levels = np.linspace(0, np.max(X2_hist[i]+X1_hist[i]), num=30) # the lowest point is being set zero because there is a negative bacterial density coming in picture otherwise
    contf = pyplot.contourf(x, y, X2_hist[i]+X1_hist[i], levels=levels)
    fig.suptitle('X1+X2 after %d time steps(%.2fhr) in grid %dx%d of \ngrid length dx=%.3f, dy=%.3f \nfor all step total time length %d hour' %(i,i*time_length/nt,gridx,gridy,dx,dy,time_length) )
    cbar = pyplot.colorbar(contf)
    cbar.set_label('X1+X2 after %d time steps' %i)
    pyplot.axis('scaled', adjustable='box')
    
 
    fig.savefig('X1 n X2 %d.jpg' %i)
    
    pyplot.close(fig)
    
    print(i)
    ("")
print("Current time is: ", end="")
print(time.asctime( time.localtime(time.time())))



#%%

txt_file=open("Input_Parameter.txt","w")


txt_file.write('g0             	' +str(g0            ) +'\n'	)
txt_file.write('gamma          	' +str(gamma         ) +'\n'  )
txt_file.write('k_value        	' +str(k_value       ) +'\n'  )
txt_file.write('Db             	' +str(Db            ) +'\n'  )
txt_file.write('Dc             	' +str(Dc            ) +'\n'  )
txt_file.write('\n'    )
txt_file.write('t_ref          	' +str(t_ref         ) +'\n'  )
txt_file.write('Xe_value       	' +str(Xe_value      ) +'\n'  )
txt_file.write('X0             	' +str(X0            ) +'\n'  )
txt_file.write('C0             	' +str(C0            ) +'\n'  )
txt_file.write('C_thresh_value 	' +str(C_thresh_value) +'\n'  )
txt_file.write('\n'    )
txt_file.write('d              	' +str(d             ) +'\n'  )
txt_file.write('rho            	' +str(rho           ) +'\n'  )
txt_file.write('mu             	' +str(mu            ) +'\n'  )
txt_file.write('dx             	' +str(dx            ) +'\n'  )
txt_file.write('dy             	' +str(dy            ) +'\n'  )
txt_file.write('\n'    )
txt_file.write('gridx          	' +str(gridx         ) +'\n'  )
txt_file.write('gridy          	' +str(gridy         ) +'\n'  )
txt_file.write('dx             	' +str(dx            ) +'\n'  )
txt_file.write('dy             	' +str(dy            ) +'\n'  )
txt_file.write('\n'    )
txt_file.write('nt             	' +str(nt            ) +'\n'  )
txt_file.write('time_length    	' +str(time_length   ) +'\n'  )

txt_file.close()

os.chdir('..')

#%% 
import winsound
duration = 1000  # milliseconds
freq = 400  # Hz
winsound.Beep(freq, duration)

#%%
print("Current Working Directory is:")
print(os.getcwd())
print("CODE OVER... BYE BYE")

#%%
