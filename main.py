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