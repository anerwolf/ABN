import matplotlib.pyplot as plt
import graphUtils
import scipy.stats as st
import numpy as np

listOfAllEdges = graphUtils.generateEdgesListFromFile("HuRI.tsv")
G, modelNeighborsLists, modelNumOfNeighbors = graphUtils.createGAndDics(listOfAllEdges)

CC = graphUtils.calcClusteCoeff(modelNumOfNeighbors, modelNeighborsLists)
print('Clustering Coefficient: ' + str(CC))
numOf4Cliques = graphUtils.find_cliques_size_k(G, 4)
print('Number of size-4 cliques: ' + str(numOf4Cliques))
numOfMotifs = graphUtils.find_special_motif(G,modelNumOfNeighbors, modelNeighborsLists)
print('Number of motifs: ' + str(numOfMotifs))

x, y, x_fit, y_fit, c1, c2 = graphUtils.calcDistParams(modelNumOfNeighbors)
print('Params: c - ' + str(c1) + ', a - ' + str(c2)) #P(k) = a * k ** (-c)
plt.plot(x, y, '*', color='blue')
plt.plot(x_fit, y_fit, '-', color='red')
plt.title('Network Degree Distribution: $P(k) = Ak^{-c}$, A = ' + str(round(c2, ndigits=2)) + ', c = ' + str(round(c1, ndigits=2)))
plt.xlabel('$k$')
plt.ylabel('$P(k)$')
plt.yscale("log")
plt.xscale("log")
plt.savefig('degDist.png')
plt.close()

#Generating an ensemble
numberOfRandomG = 20
numOfIterations = 100
CCGen = []
num4CliquesGen = []
numMotifGen = []
paramsGen = []
for i in range(numberOfRandomG):
    print('Generating random graph: ' + str(i))
    listOfAllEdgesAfterSwitches = graphUtils.generateRandomList(listOfAllEdges, numOfIterations, modelNeighborsLists)
    GAfterSwitches, modelNeighborsListsAfterSwitches, modelNumOfNeighborsAfterSwitches = graphUtils.createGAndDics(listOfAllEdgesAfterSwitches)
    currCC = graphUtils.calcClusteCoeff(modelNumOfNeighborsAfterSwitches, modelNeighborsListsAfterSwitches)
    CCGen.append(currCC)
    print('Clustering Coefficient: ' + str(currCC))
    currNum4Cliques = graphUtils.find_cliques_size_k(GAfterSwitches, 4)
    num4CliquesGen.append(currNum4Cliques)
    print('Number of size-4 cliques: ' + str(currNum4Cliques))
    currNumMotif = graphUtils.find_special_motif(GAfterSwitches,modelNumOfNeighborsAfterSwitches, modelNeighborsListsAfterSwitches)
    numMotifGen.append(currNumMotif)
    print('Number of motifs: ' + str(currNumMotif))

#Calculate confidence interval
CCConf = st.t.interval(alpha=0.95, df=len(CCGen)-1, loc=np.mean(CCGen), scale=st.sem(CCGen))
num4CliquesConf = st.t.interval(alpha=0.95, df=len(num4CliquesGen)-1, loc=np.mean(num4CliquesGen), scale=st.sem(num4CliquesGen))
numMotifConf = st.t.interval(alpha=0.95, df=len(numMotifGen)-1, loc=np.mean(numMotifGen), scale=st.sem(numMotifGen))

plt.plot(CCGen,'*', color='blue')
plt.plot([CCConf[0]] * len(CCGen), '-', color='red')
plt.plot([CCConf[1]] * len(CCGen), '-', color='red')
plt.title('Generated Cluster Coefficient')
plt.xlabel('# Network')
plt.ylabel('Cluster Coefficient')
plt.savefig('CC.png')
plt.close()

plt.plot(num4CliquesGen,'*', color='blue')
plt.plot([num4CliquesConf[0]] * len(num4CliquesGen), '-', color='red')
plt.plot([num4CliquesConf[1]] * len(num4CliquesGen), '-', color='red')
plt.title('Generated Number of Size-4 Cliques')
plt.xlabel('# Network')
plt.ylabel('# Size-4 Cliques')
plt.savefig('cliques.png')
plt.close()

plt.plot(numMotifGen,'*', color='blue')
plt.plot([numMotifConf[0]] * len(numMotifGen), '-', color='red')
plt.plot([numMotifConf[1]] * len(numMotifGen), '-', color='red')
plt.title('Generated Number of Motif')
plt.xlabel('# Network')
plt.ylabel('# Motif')
plt.savefig('motifs.png')
plt.close()