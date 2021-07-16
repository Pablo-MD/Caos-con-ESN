import numpy as np
import matplotlib.pyplot as plt


class ESN:
  
    def __init__(self, a=0.01, EscaladoRho=0.1, Nx=1000, Nu=1, Ny=1, T=2000, Test=1000, Inicial=100, x=0, X=0, Wout=0):
        self.Nu=Nu
        self.Ny=Ny
        self.T=T           #Numero de datos de datos a entrenar
        self.Test=Test        #Numero de datos de predicción
        self.Inicial=Inicial      #Numero de datos de entrenamiento inicial
        
        self.a=a
        self.EscaladoRho=EscaladoRho
        self.Nx=Nx
        
        np.random.seed(58)
        self.W=np.random.normal(0.0,1,size=(self.Nx,self.Nx))
        self.Win=np.random.normal(0.0,1,size=(self.Nx,1+Nu))
        
        self.x = np.zeros((self.Nx,1))
        self.X = np.zeros((1+self.Nu+self.Nx,self.T-100))
        self.Wout = Wout 
        
    def Reservoir(self,datos):
        rhoW = max(abs(eig) for eig in np.linalg.eig(self.W)[0])         #Cálculo del radio espectral 
        self.W*=self.EscaladoRho/rhoW

        for n in range(self.T):
            u = datos[n]
            self.x = (1-self.a)*self.x+ self.a*np.tanh( np.dot( self.Win, np.vstack((1,u)) ) + np.dot( self.W, self.x ) )
            if n >= self.Inicial:
                self.X[:,n-self.Inicial] = np.vstack((1,u,self.x))[:,0]
        
        #plt.title(r'$Activación \ del \ reservoirio \ con \ \rho = {0}$'.format(self.EscaladoRho))
        #plt.plot(self.X[0:7,0:250].T)
        #plt.savefig("Activation{0}.png".format(self.EscaladoRho))
        #plt.clf()
    
    def Plasticidad(self, LearningRate, datos):
       rhoW = max(abs(eig) for eig in np.linalg.eig(self.W)[0])         #Cálculo del radio espectral 
       self.W*=self.EscaladoRho/rhoW
       for n in range(self.T):
            u = datos[n]
            self.x = np.tanh( np.dot( self.Win, np.vstack((1,u)) ) + np.dot( self.W, self.x ) )
            y=np.zeros((self.Nx,1))
            for j in range (self.Nx):
                for i in range (self.Nx):
                    y[j]+=self.W[i,j]*self.x[i]
            for i in range (self.Nx):
                for j in range (self.Nx):   
                    self.W[i,j]=self.W[i,j]-LearningRate*(self.x[i]*y[j]-self.W[i,j]*y[j]*y[j])
                                                          
    def RidgeRegression(self,datos):
        target = datos[None,self.Inicial+1:self.T+1] 
        Beta=5e-5*np.identity(self.Nx+self.Nu+1)
        self.Wout = np.dot(np.dot(target,np.transpose(self.X)),np.linalg.inv(np.dot(self.X,np.transpose(self.X))+Beta))
        
    def TestDynamic(self,datos):
        Y = np.zeros((self.Ny,self.Test))
        u = datos[self.T]
        for t in range(self.Test):
            self.x = (1-self.a)*self.x + self.a*np.tanh( np.dot( self.Win, np.vstack((1,u)) ) + np.dot( self.W, self.x ) )
            y = np.dot( self.Wout, np.vstack((1,u,self.x)) )
            Y[:,t] = y
            u = y
        
        fig=plt.subplot()
        fig.plot( np.transpose(Y), 'b', linewidth=0.5, label= "Predicción del modelo" )
        fig.plot(datos[self.T+1:self.T+self.Test+1],'r', linewidth=0.5, label = "Serie temporal")
        fig.legend(loc='upper left', frameon=False)
        plt.savefig("MackeyGlass.png")
        
        return Y
            
    def MSE(self, datos, Y, Lerror=500):
        target=datos[self.T+1:self.T+1+Lerror]
        Y=Y[0,0:Lerror]
        return np.sqrt(sum((target-Y)*(target-Y)))/np.sqrt(Lerror)
    
"""
#datos=np.zeros(3000)
datos=np.loadtxt('MackeyGlass_t17.txt')
rango=[0.1,0.6,1,1.3,1.7,2,3]
for k in rango:
    Activacion=ESN(a=1, EscaladoRho=k)
    Activacion.Reservoir(datos)
"""



datos=np.loadtxt('MackeyGlass_t17.txt')
MackeyGlass=ESN(a=0.75, EscaladoRho=0.5)
MackeyGlass.Reservoir(datos)

MackeyGlass.RidgeRegression(datos)
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 16}

plt.ylim(-0.6,0.5)
plt.ylabel(r"$\frac{P(n)}{\theta}$", **font )
plt.xlabel("Tiempo (n)")
plt.title("Predicción: Mackey-Glass")
Y=MackeyGlass.TestDynamic(datos)
print(MackeyGlass.MSE(datos,Y))




"""
Auxdatos=np.loadtxt('Rossler.txt')
maxdatos=max(Auxdatos)
datos=np.array([i/maxdatos for i in Auxdatos])
datos=np.loadtxt('Rossler.txt')

Rossler=ESN(a=0.1,EscaladoRho=1,Nx=1000)
Rossler.Reservoir(datos)
Rossler.RidgeRegression(datos)
plt.ylim(-1.1,1.25)
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 10}
plt.ylabel(r"$x(n)$", **font )
plt.xlabel("Tiempo (n)")
plt.title("Predicción: Atractor de Rössler")
Y=Rossler.TestDynamic(datos)
print(Rossler.MSE(datos, Y))
"""



"""
datos=np.loadtxt('Lorenz.txt')
Lorenz=ESN(a=0.1,EscaladoRho=0.30001,Nx=1000)
Lorenz.Reservoir(datos)
Lorenz.RidgeRegression(datos)
plt.ylim(-20,20)
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 10}
plt.ylabel(r"$x(n)$", **font )
plt.xlabel("Tiempo (n)")
plt.title("Predicción: Atractor de Lorenz [x]")
Y=Lorenz.TestDynamic(datos)
print(Lorenz.MSE(datos, Y))
"""
