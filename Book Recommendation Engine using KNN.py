import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix

# Load datasets
books = pd.read_csv('BX-Books.csv')
users = pd.read_csv('BX-Users.csv')
ratings = pd.read_csv('BX-Book-Ratings.csv')

# Rename columns for ease of use
books.columns = ['ISBN', 'Book-Title', 'Book-Author', 'Year-Of-Publication', 'Publisher', 'Image-URL-S', 'Image-URL-M', 'Image-URL-L']
users.columns = ['User-ID', 'Location', 'Age']
ratings.columns = ['User-ID', 'ISBN', 'Book-Rating']

# Filter data
ratings = ratings[ratings['Book-Rating'] > 0]

# Merge datasets
data = pd.merge(ratings, books, on='ISBN')

# Create a pivot table
pivot_table = data.pivot(index='Book-Title', columns='User-ID', values='Book-Rating').fillna(0)

# Convert pivot table to sparse matrix
book_sparse_matrix = csr_matrix(pivot_table.values)

# Fit the KNN model
model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
model_knn.fit(book_sparse_matrix)

def get_book_recommendations(book_title, n_neighbors=5):
    book_list = pivot_table.index.tolist()
    
    # Find the index of the book
    try:
        book_idx = book_list.index(book_title)
    except ValueError:
        return f"The book '{book_title}' is not in the dataset."
    
    # Find similar books
    distances, indices = model_knn.kneighbors(book_sparse_matrix[book_idx], n_neighbors=n_neighbors+1)
    
    # Get recommended books
    recommendations = []
    for i in range(1, len(distances.flatten())):
        recommendations.append(book_list[indices.flatten()[i]])
    
    return recommendations

# Test the recommendation system
book_title = 'Harry Potter and the Sorcerer\'s Stone (Book 1)'
print(f"Recommendations for '{book_title}':")
print(get_book_recommendations(book_title))
