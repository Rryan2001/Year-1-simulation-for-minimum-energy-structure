import numpy as np
from matplotlib import pylab as plt
import copy

class polymer( ):
    def __init__( self ):
        self.myBeads = [] #A list of objects of type beads
    def LJFunction( self, r, sigma, eps):
        '''LJ potential
        Input parameters
        r: cut off distance
        sigma: distance between two beads
        eps:epsilon

        Output parameters
        LJ potential
        '''
        return 4 * eps * ( ( sigma / r )**12 - ( sigma / r )**6 )
    def harmonicFunction( self, r,sigma_h = 1,k_spring = 40, r0 = 1):
        '''Harmonic bond potential
        Input LJparameters
        r : distance between two bonded beads
        r0: equilibrium(rest) distance

        Output parameters
        Harmonic bond potential
        '''
        return 0.5*k_spring*((r-r0)**2)
    def linked_cell_blocks (self, row = 10, line = 10):
        '''divide entire system into small cells.
        Row and lines are defautly set as 10, 100 cells in total.
        Cell size should be slightly larger than cut off radius

        Output parameters
        array of blocks with (room) numbers
        '''
        xx = self.boxSize[0]/row
        yy = self.boxSize[1]/line
        '''for 2d cell lists, maximum is 8 neighbours'''
        blocks = np.zeros((row*line,8))
        '''define the relationship between neighbour cells
        by indexing the cells line by line
        // --+----+----+----+
        //  |  0  |  1  |  2  |
        // --+----+----+----+
        //  |  3  |  4  |  5  |
        // --+----+----+----+
        //  |  6  |  7  |  8  |
        // --+----+----+----+
        '''
        for i in range(line):
            for j in range(row):
                if i == 0:
                    '''left border'''
                    if j == 0 :
                        '''top left'''
                        blocks[i + j*line][0] = i + 1
                        blocks[i + j*line][1] = i + line
                        blocks[i + j*line][2] = i + line + 1
                    elif j == row - 1:
                        '''bottom left'''
                        blocks[i + j*line][0] = (j-1)*line
                        blocks[i + j*line][1] = j*10 +1
                        blocks[i + j*line][2] = (j-1)*line + 1
                    else:
                        '''other cells on left border will have 5 neighbours'''
                        blocks[i + j*line][0] = (j-1)*line
                        blocks[i + j*line][1] = (j-1)*line + 1
                        blocks[i + j*line][2] = j*10 +1
                        blocks[i + j*line][3] = (j+1)*line
                        blocks[i + j*line][4] = (j+1)*line + 1
                elif i == line - 1:
                    '''right border'''
                    if j == 0 :
                        '''top right'''
                        blocks[i + j*line][0] = i - 1
                        blocks[i + j*line][1] = i + line
                        blocks[i + j*line][2] = i + line - 1
                    elif j == row - 1:
                        '''bottom right'''
                        blocks[i + j*line][0] = i + (j-1)*line
                        blocks[i + j*line][1] = i + j*line -1
                        blocks[i + j*line][2] = i + (j-1)*line - 1
                    else:
                        '''other cells on right border will have 5 neighbours'''
                        blocks[i + j*line][0] = i + (j-1)*line
                        blocks[i + j*line][1] = i + (j-1)*line - 1
                        blocks[i + j*line][2] = i + j*10 - 1
                        blocks[i + j*line][3] = i + (j+1)*line
                        blocks[i + j*line][4] = i + (j+1)*line - 1
                elif j == 0:
                    '''top border (excluding top left and right)'''
                    blocks[i + j*line][0] = i + j*line + 1
                    blocks[i + j*line][1] = i + j*line - 1
                    blocks[i + j*line][2] = i + (j+1)*line - 1
                    blocks[i + j*line][3] = i + (j+1)*line
                    blocks[i + j*line][4] = i + (j+1)*line + 1
                elif j == row - 1:
                    '''bottom border (excluding bottom left and right)'''
                    blocks[i + j*line][0] = i + j*line + 1
                    blocks[i + j*line][1] = i + j*line - 1
                    blocks[i + j*line][2] = i + (j-1)*line - 1
                    blocks[i + j*line][3] = i + (j-1)*line
                    blocks[i + j*line][4] = i + (j-1)*line + 1
                else:
                    '''the rest have 8 neighbours'''
                    blocks[i + j*line][0] = i + (j-1)*line - 1
                    blocks[i + j*line][1] = i + (j-1)*line
                    blocks[i + j*line][2] = i + (j-1)*line + 1
                    blocks[i + j*line][3] = i + j*line - 1
                    blocks[i + j*line][4] = i + j*line + 1
                    blocks[i + j*line][5] = i + (j+1)*line - 1
                    blocks[i + j*line][6] = i + (j+1)*line
                    blocks[i + j*line][7] = i + (j+1)*line + 1
        return blocks

class Bead(polymer):
    def __init__( self,myIndex, pos):
        self.myIndex = myIndex
        self.pos = pos
    def virtualMove( self, myIndex, dxMax = 0.2, xx=4, yy=4 ):
        '''generate random displacement
        Input parameters
        myIndex: index of selected particle
        dxMax: maximum displacement
        xx: dimension of the small block along x
        yy: dimension of the small block along y

        Output parameters
        update configuration(if movement is accepted)
        '''
        dis = np.random.uniform(-1,1) * dxMax
        p_xy = np.random.randint(2)
        ii = self.pos[myIndex][2]//xx
        jj = self.pos[myIndex][3]//yy
        dx = 0.0
        dy = 0.0
        temp_bNeigh = copy.deepcopy(self.bNeigh)
        temp_pos = copy.deepcopy(self.pos)
        temp_cellNumber = copy.deepcopy(self.cellNumber)
        temp_neigh_list = self.Effective_pair(temp_pos, temp_cellNumber, temp_bNeigh,myIndex )
        if p_xy == 0:
            #move x
            dx = dis
            if temp_pos[myIndex][2]+dx > 0 and temp_pos[myIndex][2]+dx < self.boxSize[0]:
                temp_pos[myIndex][2] += dx
                if temp_pos[myIndex][2]//xx != ii:
                    #change room(cell)
                    temp_cellNumber[myIndex] = int(temp_pos[myIndex][2]//xx + temp_pos[myIndex][3]//yy*10)
                    #remove it from the original room and add it to the new room
                    temp_bNeigh[int(ii + jj*10)].remove(myIndex)
                    temp_bNeigh[int(temp_pos[myIndex][2]//xx + temp_pos[myIndex][3]//yy*10)].append(myIndex)
                    temp_neigh_list = self.Effective_pair(temp_pos, temp_cellNumber, temp_bNeigh,myIndex )
            dE = self.deltaEnergy( myIndex, dx, dy, temp_neigh_list )
            if dE < 0:
                accept = True
            else:
                X = np.random.rand()
                if X < np.exp( -dE ):
                    accept = True
                else:
                    accept = False
        elif p_xy == 1:
            #move y
            dy = dis
            if temp_pos[myIndex][3]+dy > 0 and temp_pos[myIndex][3]+dy < self.boxSize[1]:
                temp_pos[myIndex][3] += dy
                if temp_pos[myIndex][3]//yy != jj:
                    #change room(cell)
                    temp_cellNumber[myIndex] = int(temp_pos[myIndex][2]//xx + temp_pos[myIndex][3]//yy*10)
                    #remove it from the old room and add it to the new room
                    temp_bNeigh[int(ii + jj*10)].remove(myIndex)
                    temp_bNeigh[int(temp_pos[myIndex][2]//xx + temp_pos[myIndex][3]//yy*10)].append(myIndex)
                    temp_neigh_list = self.Effective_pair(temp_pos, temp_cellNumber, temp_bNeigh,myIndex )
            dE = self.deltaEnergy( myIndex, dx, dy, temp_neigh_list )
            if dE < 0:
                accept = True
            else:
                X = np.random.rand()
                if X < np.exp( -dE ):
                    accept = True
                else:
                    accept = False
        if accept:
            self.bNeigh = temp_bNeigh
            self.pos = temp_pos
            self.cellNumber = temp_cellNumber
            self.neigh_list = temp_neigh_list
        return

    def deltaEnergy( self, myIndex, dx, dy, neigh_list):
        '''calculate the energy difference at two positions
        Input parameters
        myIndex: index of selected particle
        movement along x and y (dx and dy)
        neighbour list of selective particle

        Output parameters
        delta energy
        '''
        Elj = 0.0
        Ek = 0.0
        # sigma_A = 1
        # eps_AA = eps_BB = 5 kT
        # eps_AB = 1 kT
        for d in neigh_list:
            d_new = ( ( d[1]+dx )**2 + ( d[2]+dy )**2 )**0.5
            d_old = ( ( d[1] )**2 + ( d[2] )**2 )**0.5
            if self.pos[myIndex][1] == 1:
                if int(d[0]) == 1:  #AA
                    Elj += self.LJFunction( d_new,self.LJparameters['sigmaA'],self.LJparameters['AA'] ) - self.LJFunction( d_old,self.LJparameters['sigmaA'],self.LJparameters['AA']  )
                elif int(d[0]) == 2: #AB
                    Elj += self.LJFunction( d_new,self.LJparameters['sigmaAB'],self.LJparameters['AB']  ) - self.LJFunction( d_old,self.LJparameters['sigmaAB'],self.LJparameters['AB']  )
            elif self.pos[myIndex][1] == 2:
                if int(d[0]) == 1:  #BA
                    Elj += self.LJFunction( d_new,self.LJparameters['sigmaAB'],self.LJparameters['AB'] ) - self.LJFunction( d_old,self.LJparameters['sigmaAB'],self.LJparameters['AB']  )
                elif int(d[0]) == 2: #BB
                    Elj += self.LJFunction( d_new,self.LJparameters['sigmaB'],self.LJparameters['BB']  ) - self.LJFunction( d_old,self.LJparameters['sigmaB'],self.LJparameters['BB']  )
        for d in self.bonded[myIndex]:
            d_old = ((self.pos[myIndex][2]-self.pos[d][2])**2+(self.pos[myIndex][3]-self.pos[d][3])**2)**0.5
            d_new = ((self.pos[myIndex][2]+dx-self.pos[d][2])**2+(self.pos[myIndex][3]+dy-self.pos[d][3])**2)**0.5
            Ek += self.harmonicFunction(d_new) - self.harmonicFunction(d_old)
        E = Ek + Elj
        return  E
#
    def myNeighbours( self, allPos,row = 10, line = 10 ):
        '''generate cell list

        '''
        xx = self.boxSize[0]/row
        yy = self.boxSize[1]/line
        '''d are rooms'''
        d = [[] for x in range(0, row*line)]
        '''prepare to allocate atoms into cells'''
        atomlist_cell_number = np.zeros(len(allPos))
        for i in range(len(allPos)):
            #know ith room has what atoms
            d[int(allPos[i][2]//xx + allPos[i][3]//yy*10)].append(i)
            #by assigning the value of room to every atom, we know which atom is in which room
            atomlist_cell_number[i] = int(allPos[i][2]//xx + allPos[i][3]//yy*10)
        return atomlist_cell_number,d

    def Effective_pair( self, allPos, atomlist_cell_number, d,myIndex ):
        '''this function finds out all valid pairs for selected atom
        '''
        #check for lj pairs
        neigh_list = []
        cell_number = int(atomlist_cell_number[myIndex])
        '''firstly check atoms in the same cell'''
        for atom in d[cell_number]:
            distance = ( ( allPos[myIndex][2]-allPos[atom][2] )**2 + ( allPos[myIndex][3]-allPos[atom][3] )**2 )**0.5
            '''effective LJ pair, energy counted'''
            if distance < 2.5 and distance > 0:
                #record distance and type of bead (in x and y)
                   neigh_list.append([allPos[atom][1],allPos[myIndex][2]-allPos[atom][2],allPos[myIndex][3]-allPos[atom][3]])
        for i in range( len(self.blocks[cell_number] ) ):
            '''check if invalid neighbour cell'''
            if self.blocks[cell_number][i] == 0 and self.blocks[cell_number][i+1] == 0:
                break
            else:
                for cell_no in self.blocks[cell_number]:
                    for atom in d[int(cell_no)]:
                        distance = ( ( allPos[myIndex][2]-allPos[atom][2] )**2 + ( allPos[myIndex][3]-allPos[atom][3] )**2 )**0.5
                        if distance < 2.5 and distance > 0:
                            #record distance and type of bead (in x and y)
                            neigh_list.append([allPos[atom][1],allPos[myIndex][2]-allPos[atom][2],allPos[myIndex][3]-allPos[atom][3]])
        return (neigh_list)

class simulation( Bead ):
    def __init__( self, nPolymers, typeCoPolymers, LJparameters, boxSize, nSample, nSteps ):
        self.nPolymers = nPolymers
        self.typeCopolymers = typeCopolymers
        self.LJparameters = LJparameters
        self.boxSize = boxSize
        self.nSteps = nSteps
        self.nSample = nSample
        self.LJparameters = LJparameters
        self.blocks = self.linked_cell_blocks()
        self.pos,self.bonded = self.generateRandomConf()
        self.cellNumber,self.bNeigh = self.myNeighbours(self.pos)

    def generateRandomConf( self ):
        '''generate initial random configuration for the system'''
        #prepare an empty list for positions
        pos = []
        #maximum have two bonds
        bond = []
        #prepare an empty list for bonds
        for ii in range(self.nPolymers):
            for i in self.typeCopolymers.keys():
                for j in range( self.typeCopolymers[ i ]['number'] ):
                    La = self.typeCopolymers[ i ]['La']
                    Lb = self.typeCopolymers[ i ]['Lb']
                    '''randomly generate x,y coordinates for atom, type A and B'''
                    if i == 'type1':
                        for k in range( La ):
                            x = self.boxSize[0]*np.random.rand()
                            y = self.boxSize[1]*np.random.rand()
                            pos.append([1,1,x,y])
                        for l in range( Lb ):
                            x = self.boxSize[0]*np.random.rand()
                            y = self.boxSize[1]*np.random.rand()
                            pos.append([1,2,x,y])
                    elif i == 'type2':
                        for k in range( La ):
                            x = self.boxSize[0]*np.random.rand()
                            y = self.boxSize[1]*np.random.rand()
                            pos.append([2,1,x,y])
                        for l in range( Lb ):
                            x = self.boxSize[0]*np.random.rand()
                            y = self.boxSize[1]*np.random.rand()
                            pos.append([2,2,x,y])
                    if i == 'type3':
                        for k in range( La ):
                            x = self.boxSize[0]*np.random.rand()
                            y = self.boxSize[1]*np.random.rand()
                            pos.append([3,1,x,y])
                        for l in range( Lb ):
                            x = self.boxSize[0]*np.random.rand()
                            y = self.boxSize[1]*np.random.rand()
                            pos.append([3,2,x,y])
                    '''linking bonds between atoms in one chain, 1-2-3-4-5...'''
                    for m in range( j*(La+Lb),(j+1)*(La+Lb) ):
                        if m - 1 >= j*(La+Lb) and m + 1 <= (j+1)*(La+Lb)-1:
                            '''beads which have two neighbours to form bonds'''
                            bond.append([m-1,m+1])
                        elif m + 1 <= (j+1)*(La+Lb)-1:
                            '''first bead in a chain'''
                            bond.append([m+1])
                        else:
                            '''last bead in a chain'''
                            bond.append([m-1])
        return np.array(pos),np.array(bond)


    def calculateTotEnergy( self ):
        '''calculate the total energy of system
        
        output parameters
        total energy of system
        '''
        Elj = 0.0
        Ek = 0.0
        for myIndex in range(len(self.pos)):
            neigh_list = []
            cell_number = int(self.cellNumber[myIndex])
            d = self.bNeigh
            '''firstly check atoms in the same cell'''
            for atom in d[cell_number]:
                distance = ( ( self.pos[myIndex][2]-self.pos[atom][2] )**2 + ( self.pos[myIndex][3]-self.pos[atom][3] )**2 )**0.5
                '''effective LJ pair, energy counted'''
                if distance < 2.5 and distance > 0:
                    #record distance and type of bead (in x and y)
                       neigh_list.append([self.pos[atom][1],self.pos[myIndex][2]-self.pos[atom][2],self.pos[myIndex][3]-self.pos[atom][3]])
            for i in range( len(self.blocks[cell_number] ) ):
                '''check if invalid neighbour cell'''
                if self.blocks[cell_number][i] == 0 and self.blocks[cell_number][i+1] == 0:
                    break
                else:
                    for cell_no in self.blocks[cell_number]:
                        for atom in d[int(cell_no)]:
                            distance = ( ( self.pos[myIndex][2]-self.pos[atom][2] )**2 + ( self.pos[myIndex][3]-self.pos[atom][3] )**2 )**0.5
                            if distance < 2.5 and distance > 0:
                                #record distance and type of bead (in x and y)
                                neigh_list.append([self.pos[atom][1],self.pos[myIndex][2]-self.pos[atom][2],self.pos[myIndex][3]-self.pos[atom][3]])
            for d in neigh_list:
                distance = ( ( d[1] )**2 + ( d[2] )**2 )**0.5
                if self.pos[myIndex][1] == 1:
                    if int(d[0]) == 1:  #AA
                        Elj += self.LJFunction( distance,self.LJparameters['sigmaA'],self.LJparameters['AA'] )
                    elif int(d[0]) == 2: #AB
                        Elj += self.LJFunction( distance,self.LJparameters['sigmaAB'],self.LJparameters['AB']  )
                elif self.pos[myIndex][1] == 2:
                    if int(d[0]) == 1:  #BA
                        Elj += self.LJFunction( distance,self.LJparameters['sigmaAB'],self.LJparameters['AB'] )
                    elif int(d[0]) == 2: #BB
                        Elj += self.LJFunction( distance,self.LJparameters['sigmaB'],self.LJparameters['BB']  )
            for d in self.bonded[myIndex]:
                x_2 = ((self.pos[myIndex][2]-self.pos[d][2])**2+(self.pos[myIndex][3]-self.pos[d][3])**2)**0.5
                Ek += self.harmonicFunction(x_2)
        return (Elj + Ek)

    def calculateDeltaEnergy( self ):
        '''calculate energy difference'''
        MyIndex = np.random.choice(len(self.pos))
        DeltaEnergy = self.virtualMove(MyIndex)
        return DeltaEnergy

    def makeMCstep( self ):
        '''make monte-carlo step'''
        DeltaEnergy = self.calculateDeltaEnergy()
        return
    #def plotConfiguration( self ):

    def plotMe( self ):
        list1 = []
        list2 = []
        list3 = []
        list4 = []
        list5 = []
        list6 = []
        for atom in self.pos:
            if atom[0] == 1:
                if atom[1] == 1:
                    list1.append(atom[2:])
                else:
                    list2.append(atom[2:])
            if atom[0] == 2:
                if atom[1] == 1:
                    list3.append(atom[2:])
                else:
                    list4.append(atom[2:])
            else:
                if atom[1] == 1:
                    list5.append(atom[2:])
                else:
                    list6.append(atom[2:])
        array1 = np.array(list1)
        array2 = np.array(list2)
        #array3 = np.array(list3)
        #array4 = np.array(list4)
        #array5 = np.array(list5)
        #array6 = np.array(list6)
        plt.scatter(array1[:,0],array1[:,1],c = 'r',label = 'type1 - A')
        plt.scatter(array2[:,0],array2[:,1],c = 'b',label = 'type1 - B')
        #plt.scatter(array3[:,0],array3[:,1],c = 'g',label = 'type2 - A')
        #plt.scatter(array4[:,0],array4[:,1],c = 'y',label = 'type2 - B')
        #plt.scatter(array5[:,0],array5[:,1],c = 'k',label = 'type3 - A')
        #plt.scatter(array6[:,0],array6[:,1],c = 'orange',label = 'type3 - B')
        plt.xlim(0, 40)
        plt.ylim(0, 40)
        plt.legend(loc = 'best')
        plt.show()

"""
typeCopolymers = {
                    "type1" : { "number":10, "La" : 10, "Lb": 20 },
                    "type2" : { "number":20, "La" : 5, "Lb": 20 },
                    "type3" : { "number":20, "La" : 20, "Lb": 5 }
                 }
"""
typeCopolymers = {
                    "type1" : { "number":20, "La" : 10, "Lb": 10 },
                 }
LJparameters = { "AA" : 20,"BB": 1, "AB": 1,"sigmaA": 1,"sigmaB": 1,"sigmaAB": 1 }
#aa = sigma_A / sigma_B
#bb = La / Lb
boxSize = np.array([40,40])
#number of Monte Carlo (MC)
nSteps = 300000
nSample = 1000
nSample1 = 30000
nSample2 = 10000
nPolymers = 1
simu = simulation( nPolymers, typeCopolymers, LJparameters, boxSize, nSample, nSteps )
print(simu.bonded)
for i in range( nSteps ):
    simu.makeMCstep()
    if ( i % nSample1 ) == 0:
        with open('MonteCarlo_Energy.txt', 'a') as f:
            dE = simu.calculateTotEnergy()/100
            f.write("%f\n" % dE)
        with open('MonteCarlo_pos.txt', 'a') as f:
            for item in simu.pos:
                f.write("%s\n" % str(item))
    if ( i % nSample2 ) == 0:
        simu.plotMe()
