import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from flask import Flask,redirect,url_for,render_template,request
from flask import render_template

movies = pd.read_csv("dataset/movies.csv")
ratings = pd.read_csv("dataset/ratings.csv")

final_dataset = ratings.pivot(index= 'movieId',columns = 'userId',values = 'rating')

# cleaning the dataset i.e. replacing all NaN with 0
final_dataset.fillna(0, inplace = True)

# reducing sparsity to avoid huge computation 

csr_data = csr_matrix(final_dataset.values)
final_dataset.reset_index(inplace = True)

knn = NearestNeighbors(metric = 'cosine', algorithm = 'brute', n_neighbors = 20, n_jobs = -1)
knn.fit(csr_data)

# recommendation algorithm function
# check if movie name input is in database
# if it exists in database, use recommendation system to find similar movies
# then we sort them based on their similarity distance and output top 5

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index1.html')

@app.route('/movie/<string:name>')
def movie(name):
    return 'welcome %s' % name

def get_movie_recommendation(name):
    movie_name=name.title()
    n_movies_to_reccomend = 5
    movie_list = movies[movies['title'].str.contains(movie_name)]
    if len(movie_list):    # if there is a match
        movie_idx = movie_list.iloc[0]['movieId']
        movie_idx = final_dataset[final_dataset['movieId'] == movie_idx].index[0]
        distances, indices = knn.kneighbors(csr_data[movie_idx], n_neighbors = n_movies_to_reccomend + 1)
        rec_movie_indices = sorted(list(zip(indices.squeeze().tolist(), distances.squeeze().tolist())), key = lambda x: x[1])[:0:-1]
        recommend_frame = []
        for val in rec_movie_indices:
            movie_idx = final_dataset.iloc[val[0]]['movieId']
            idx = movies[movies['movieId'] == movie_idx].index
            recommend_frame.append({'Title':movies.iloc[idx]['title'].values[0], 'Distance':val[1]})
        df = pd.DataFrame(recommend_frame, index = range(1, n_movies_to_reccomend + 1))
        x=dict()
        j=0
        for i in recommend_frame:
            x['Title'+str(j)]=i['Title']
            j+=1
        return render_template('index.html',movie=x)
    else:
        return "No movies found. Please try another movie!"