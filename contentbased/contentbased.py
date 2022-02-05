import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

###### helper functions. Use them when needed #######
def get_title_from_index(index):
	return df[df.index == index]["title"].values[0]

def get_index_from_title(title):
	return df[df.title == title]["index"].values[0]
##################################################
 
##Step 1: Read CSV File using pands

df=pd.read_csv("movie_dataset.csv")
#print(df.head)
#print(df.columns)
###################################################


v=df['vote_count']
R=df['vote_average']
C=df['vote_average'].mean()
m=df['vote_count'].quantile(0.70)
#now we have a new attribute (weighted_average) so our coulmns increment 1
df['weighted_average']=((R*v)+ (C*m))/(v+m)
#print(df.head())

#now we start to sort the movies based on weighted_average
movie_sorted_ranking=df.sort_values('weighted_average',ascending=False)
#print(movie_sorted_ranking[['original_title', 'vote_count', 'vote_average', 'weighted_average', 'popularity']].head(7))

#now we make agraph to clealer information
 
 
# best movie by average votes
# weight_average=movie_sorted_ranking.sort_values('weighted_average',ascending=False)
# plt.figure(figsize=(12,6))
# axis1=sns.barplot(x=weight_average['weighted_average'].head(10), y=weight_average['original_title'].head(10), data=weight_average)
# plt.xlim(4, 10)
# plt.title('Best Movies by average votes', weight='bold')
# plt.xlabel('Weighted Average Score', weight='bold')
# plt.ylabel('Movie Title', weight='bold')
# plt.savefig('best_movies.png')
#----------------------------------------------------------------------------------------------------------


# print( movie_sorted_ranking['popularity'].head())

# #most popular by votes
# popularity=movie_sorted_ranking.sort_values('popularity',ascending=False)
# plt.figure(figsize=(12,6))
# ax=sns.barplot(x=popularity['popularity'].head(10), y=popularity['original_title'].head(10), data=popularity)

# plt.title('Most Popular by Votes', weight='bold')
# plt.xlabel('Score of Popularity', weight='bold')
# plt.ylabel('Movie Title', weight='bold')
# plt.savefig('best_popular_movies.png')

#############################################
#Recommendation based on scaled weighted average and popularity score(Priority is given 50% to both)
#now we take the popularity and weghit_averge max them in data frame and make scaling for them because of differents values


scaling=MinMaxScaler()
movie_scaled_df=scaling.fit_transform(df[['weighted_average','popularity']])

movie_normalized_df=pd.DataFrame(movie_scaled_df,columns=['weighted_average','popularity'])
#print(movie_normalized_df.head())
# after we made a update and scling weighted_average and popularity we put them in our dataset (movies_cleaned_df)
df[['normalized_weight_average','normalized_popularity']]= movie_normalized_df
#print(movies_cleaned_df.head())
df['score'] = df['normalized_weight_average'] * 0.5 + df['normalized_popularity'] * 0.5
df = df.sort_values(['score'], ascending=False)
print("\n")
print("the trending movies\n")
print(df[['original_title'  , 'score']].head(10))

scored_df = df.sort_values('score', ascending=False)

plt.figure(figsize=(16,6))

ax = sns.barplot(x=scored_df['score'].head(10), y=scored_df['original_title'].head(10), data=scored_df, palette='deep')

#plt.xlim(3.55, 5.25)
plt.title('Best Rated & Most Popular Blend', weight='bold')
plt.xlabel('Score', weight='bold')
plt.ylabel('Movie Title', weight='bold')

plt.savefig('scored_movies.png')

########################################################################################################################

#########################################

##Step 2: Select Features

features=['keywords','overview','genres','cast']
for feature in features:
 df[feature]=df[feature].fillna('')
    
    

##Step 3: Create a column in DF which combines all selected features
# Calculate all the components based on the above formula
def combine_features(row):
    try:
     return row['keywords']+" "+row['overview']+" "+row['genres']+" "+row['cast']
    except:
        print("Erorr:",row)
df["combined_features"]=df.apply(combine_features,axis=1)
#print( "combined features:",df ["combined_features"].head())

# use axis=1 to combine only one row in the time 
#we have a problem that we have nan values so we use fillna to solve it 


#if we print it we will have an error (we can't combine strings) so we have aproblem
#in the return of the function

##Step 4: Create count matrix from this new combined column
cv=CountVectorizer()
count_matrix= cv.fit_transform(df["combined_features"])

##Step 5: Compute the Cosine Similarity based on the count_matrix
cosine_sim=cosine_similarity(count_matrix)

#movie_user_likes = "Avatar"
def get_recommended(movie_user_likes):
## Step 6: Get index of this movie from its title
 movie_index=get_index_from_title(movie_user_likes)
 similar_movies=  list (enumerate(cosine_sim[movie_index]))

## Step 7: Get a list of similar movies in descending order of similarity score
 sorted_similar_movies= sorted(similar_movies,key=lambda x:x[1],reverse=True)


# ## Step 8: Print titles of first 50 movies
 i=0
 #print("\n")
 #print("we recommended you :\n")
 recommend_frame = []
 for movie in sorted_similar_movies:
    #print(get_title_from_index(movie[0]))
    recommend_frame.append(get_title_from_index(movie[0]))
    i=i+1
    if(i>10):
        break
 return recommend_frame
    
    
from tkinter import *
top = Tk()
top.title("ContenBased")
top.minsize(500,500)
filmName = Label(text="film name")
filmName.pack()
filmNameEntry = Entry()
filmNameEntry.pack()
def fun() :
    l = get_recommended(filmNameEntry.get())
    print(len(l))
    l = np.array(l)
    for i in range(len(l)) :
        filmName = Label(text=str(i+1)+" : film  "+l[i])
        filmName.pack()
but = Button(text="recommended" , command=fun)
but.pack()


top.mainloop()
    
    
    
    
    