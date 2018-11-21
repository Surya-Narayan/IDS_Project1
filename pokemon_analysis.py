import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import math
from random import sample
from scipy.stats import norm

df = pd.read_csv('Pokemon_uncleaned.csv')

# CLEANING THE INDEX  
df.columns = df.columns.str.upper().str.replace('_','')
df = df.drop(['#'],axis = 1 )
for i in range(df.shape[0]):
    name = df.loc[i,'NAME'].split()
    if 'Mega' in name[0]:
        df.loc[i,'NAME'] = 'Mega ' + name[0].split('Mega')[0]
#print(df.shape)
#print(df.columns)
count = 0;

#REPLACING TYPE 2 WITH TYPE 1
for i in range(df.shape[0]):
    if type(df['TYPE 2'][i]) != str :
        count += 1
        df.loc[i,'TYPE 2'] = df.loc[i,'TYPE 1']
#print(df.head(10))
print(df.isnull().sum())
#CLEANING SP. ATK AND SP. DEF BY REPLACING WITH THE MEAN
spatk_mean = df['SP. ATK'].mean(skipna = True)
spdef_mean = df['SP. DEF'].mean(skipna = True)
print(spatk_mean)
print(spdef_mean)
df['SP. ATK'].fillna(round(spatk_mean),inplace = True)
df['SP. DEF'].fillna(round(spdef_mean),inplace = True)
df['LEGENDARY'].fillna('FALSE',inplace = True)
print(df.isnull().sum())
#df.to_csv('cleaned_dataset.csv')


#BOX PLOT FOR EACH COLUMN
df2=df.drop(['GENERATION','LEGENDARY','TOTAL'],axis=1)
sns.boxplot(data=df2)
plt.ylim(0,300)  #change the scale of y axix
plt.show()

#HISTOGRAM
bins=range(0,200,20)#they act as containers
plt.hist(df["SPEED"],bins,histtype="bar",rwidth=1.2,color='#0ff0ff') #hist() is used to plot a histogram
#print(df["SPEED"])
plt.hist(df["DEFENSE"],bins,histtype="bar",rwidth=1.2,color='#0ff0ff') #hist() is used to plot a histogram
plt.xlabel('Defense')#set the xlabel name
plt.ylabel('Count') #set the ylabel name
plt.plot()
plt.axvline(df['ATTACK'].mean(),linestyle='dashed',color='red') #draw a vertical line showing the average Attack value
plt.show()

#PIE CHART
labels = 'Water', 'Normal', 'Grass', 'Bug', 'Psychic', 'Fire', 'Electric', 'Rock', 'Other'
sizes = [112, 98, 70, 69, 57, 52, 44, 44, 175]
colors = ['Y', 'B', 'M', 'C', 'R', 'G', 'silver', 'brown', 'M']
explode = (0, 0, 0.1, 0, 0, 0, 0, 0, 0)  # only "explode" the 3rd slice 
plt.pie(sizes, explode = explode,labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=90)
plt.axis('equal')
plt.title("Percentage of Different Types of Pokemon")
plt.plot()
fig=plt.gcf()
fig.set_size_inches(7,7)
plt.show()

#LINE GRAPH
a=df.groupby(['GENERATION','TYPE 1']).count().reset_index()
a=a[['GENERATION','TYPE 1','TOTAL']]
a=a.pivot('GENERATION','TYPE 1','TOTAL')
a[['Water','Fire','Grass','Dragon','Normal','Rock','Flying','Electric']].plot(color=['b','r','g','#FFA500','brown','#6666ff','#001012','y'],marker='o')
fig=plt.gcf()
fig.set_size_inches(12,6)
plt.show()

#BAR GRAPH
x_bar = df["TYPE 1"].unique()
y_bar = []
xy  =dict()
for i in x_bar[:8]:
    y_bar.append((df['TYPE 1']==i).sum())
xy = dict(sorted(zip(x_bar,y_bar) , key = lambda x : x[1]))
bg = pd.DataFrame.from_dict(xy, orient = 'index')
bg.plot(kind = 'bar')
plt.ylabel("No of Pokemons")
plt.legend("P")
plt.show()
#SCATTERPLOT
fire=df[(df['TYPE 1']=='Fire') | ((df['TYPE 2'])=="Fire")] #fire contains all fire pokemons
#water=df[(df['TYPE 1']=='Water') | ((df['TYPE 2'])=="Water")]  #all water pokemins
plt.scatter(fire.DEFENSE.head(50),fire.ATTACK.head(50),color='R',label='Fire',marker="*",s=50) #scatter plot
#plt.scatter(water.ATTACK.head(50),water.DEFENSE.head(50),color='B',label="Water",s=25)
plt.ylabel("ATTACK")
plt.xlabel("DEFENCE")
plt.legend()
plt.plot()
fig=plt.gcf()  #get the current figure using .gcf()
fig.set_size_inches(12,6) #set the size for the figure
plt.show()

#HEAT MAP FOR CORRELATION
fire=df[(df['TYPE 1']=='Fire') | ((df['TYPE 2'])=="Fire")]
plt.figure(figsize=(16,10)) #manage the size of the plot
sns.heatmap(fire.corr(),annot=True)#df.corr() makes a correlation matrix and sns.heatmap is used to show the correlations heatmap
plt.show()
pop = list(df.HP)
pop_mean = np.mean(pop)
sd = np.std(pop)
sam=[]
for i in range(1,500):
    sam.append(np.mean(sample(pop,50)))
    sam_mean = np.mean(sam)
#print(pop)
print("\n\n\n\nHYPOTHESIS TESTING")
print("Sample Mean( present healing power ) : ",sam_mean)
print("Sample Standard Deviation ",sd/(math.sqrt(500)))

print("Expected Healing power after an upgrade : ",sam_mean)
z = (71-sam_mean)*math.sqrt(500)/sd
print("z value = ",z)
p = norm.pdf(z)
print(" p value = ",p)
print("As p-value is greater than alpha, the hypothesis is NOT REJECTED");




