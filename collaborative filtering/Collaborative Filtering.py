from ast import literal_eval

import pandas as pd # to read data from csv file
import numpy as np
import matplotlib as  plt  #  for creating static, animated, and interactive visualizations in Python
import matplotlib.pyplot as  pplt

import seaborn as sns      #Seaborn is a Python data visualization library based on matplotlib.
# It provides a high-level interface for drawing attractive and informative statistical graphics
from datetime import datetime

from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
#from surprise import SVD, Dataset, Reader
#from surprise.model_selection import train_test_split, cross_validate

#from ast import literal_eval

# load the movie metadata

df_movies = pd.read_csv( 'movies.csv'   )

print (df_movies.shape) # return the size of matrix
print (df_movies.columns) # return the name of the columns
df_movies.head(1)

# load the movie ratings
path = 'E:\\computers\\Level 3\\AI\\project\\data set of the article link\\ml-latest-small\\'
df_ratings=pd.read_csv( 'ratings.csv')

print(df_ratings.shape)
print(df_ratings.columns)
print(df_ratings.head(3))

rankings_count = df_ratings.rating.value_counts().sort_values() #return Series a datatype which its indexes can be float
# print( rankings_count[.5])
sns.barplot ( x= rankings_count.index.sort_values() , y = rankings_count , color = 'b' )
#sns.set_theme(style="whitegrid")
plt.pyplot.show() # shows us the board

final_dataset = df_ratings.pivot( index='movieId' , columns='userId' , values='rating' )
#to make a new dataframe where each column would represent each unique userId and each row represents each unique movieId.
#print( final_dataset.head() )
final_dataset.fillna(0,inplace=True) #final_dataset.fillna(0,inplace=True)
print( final_dataset.head() )
#(  df_ratings.groupby('movieId')['rating'].agg('count') )
'''/*
Let’s visualize how these filters look like
*/'''
no_user_voted = df_ratings.groupby('movieId')['rating'].agg('count')
no_movies_voted = df_ratings.groupby('userId')['rating'].agg('count')
#print( df_ratings.groupby('userId')['rating'].agg('count') )
f,ax = pplt.subplots(1,1,figsize=(16,4)) #show empty 2d
#df_ratings['rating'].plot(kind='hist') #A histogram shows the number of occurrences of different values in a dataset.
pplt.scatter( no_user_voted.index , no_user_voted , color = 'mediumseagreen' )
#A scatter plot is a diagram where each value in the data set is represented by a dot.
#Let’s visualize the number of users who voted with our threshold of 10.
pplt.axhline(y=10,color='r') #is used to add a horizontal line across the axis.
pplt.xlabel('MovieId')
pplt.ylabel('No. of users voted')
pplt.show()

                                # rows we want                          , column( : that means all column with us )
final_dataset = final_dataset.loc[no_user_voted[no_user_voted > 10].index,:] # remove all rows(films) that no_user_voted > 10
#Let’s visualize the number of votes by each user with our threshold of 50.
# lock :  DataFrame.loc attribute to access a particular cell in the given Dataframe using the index and column labels.
f,ax = pplt.subplots(1,1,figsize=(16,4))
pplt.scatter( no_movies_voted.index , no_movies_voted , color = 'mediumseagreen' )
pplt.axhline(y=50,color='r')
pplt.xlabel('UserId')
pplt.ylabel('No. of votes by user')
pplt.show()

#Making the necessary modifications as per the threshold set.
final_dataset=final_dataset.loc[:,no_movies_voted[no_movies_voted > 50].index]
print(final_dataset)
'''Sparse Data: is a data set where most of the item values are zero.

Dense Array: is the opposite of a sparse array: most of the values are not zero
CSR - Compressed Sparse Row. For fast row slicing, faster matrix vector products
return the indexes of the non zero value
'''
csr_data = csr_matrix(final_dataset.values)
print ( csr_data )
final_dataset.reset_index(inplace=True) # add column with the original indexes 0 to n not from 1,
#inplace: Boolean value, make changes in the original data frame itself if True.
print(final_dataset)

knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)
# metric :
knn.fit(csr_data)
l = ()
def get_movie_recommendation(movie_name):
#movie_name = 'Toy Story'
    n_movies_to_reccomend = 10
    movie_list = df_movies[ df_movies['title'].str.contains(movie_name) ]#return series of all movies have the same name
    print(movie_list)
    if ( len(movie_list) ) :
            movie_idx = movie_list.iloc[0]['movieId'] # return the
            print(movie_idx)
            movie_index = final_dataset[ final_dataset['movieId'] == movie_idx ].index[0]
            distances, indices = knn.kneighbors(csr_data[movie_idx], n_neighbors=n_movies_to_reccomend + 1)
            rec_movie_indices = sorted(list(zip(indices.squeeze().tolist(), distances.squeeze().tolist())),
                                       key=lambda x: x[1])[:0:-1]
            recommend_frame = []
            for val in rec_movie_indices:
                movie_idx = final_dataset.iloc[val[0]]['movieId']
                idx = df_movies[df_movies['movieId'] == movie_idx].index
                recommend_frame.append({'Title': df_movies.iloc[idx]['title'].values[0], 'Distance': val[1]})
            global df
            df = pd.DataFrame(recommend_frame, index=range(1, n_movies_to_reccomend + 1))
            #print(str(type(l))+str(len(l))+str(l.shape))
            return df
    else:
        return "No movies found. Please check your input"

print(get_movie_recommendation('Toy Story'))
print(len(l))
#l = np.array(df)
from tkinter import *
top = Tk()
top.title("collaborative")
top.minsize(500,500)
filmName = Label(text="film name")
filmName.pack()
filmNameEntry = Entry()
filmNameEntry.pack()
def fun() :
    get_movie_recommendation(filmNameEntry.get())
    print(len(df))
    l = np.array(df)
    for i in range(len(l)) :
        filmName = Label(text=str(i+1)+" : film is "+l[i][0])
        filmName.pack()
but = Button(text="predict" , command=fun)
but.pack()


top.mainloop()



