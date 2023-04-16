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