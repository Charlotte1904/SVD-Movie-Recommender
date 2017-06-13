import pandas as pd
import numpy as np
from numpy import linalg as LA
import scipy.sparse as ssp
from scipy.sparse import csr_matrix
from scipy.sparse import csc_matrix
from sparsesvd import sparsesvd

r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
df_ratings = pd.read_csv('ml-100k/u.data', sep='\t', names=r_cols,encoding='latin-1')
i_cols = ['movie_id', 'movie_title' ,'release_date','video_release_date', 'IMDb_URL', 'unknown', 'Action', 'Adventure','Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy','Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
items = pd.read_csv('ml-100k/u.item', sep='|', names = i_cols,encoding='latin-1')
'''Drop Timestamp & create a matrix '''
ratings = df_ratings.drop('unix_timestamp',1)
m, n = max(df_ratings['user_id']), max(df_ratings['movie_id'])


class MovieRecommender():

    def __init__(self, ratings):
        self.ratings = ratings
        self.movies_list = movies_list
        self.SVD_matrix = np.zeros(shape = (944, 1683), dtype = np.float32)
        self.user_Rec = dict()
        self.movies_Seen = dict()
        self.movies_pred = dict()
        self.all_movies_rec = dict()
        self.intersection_movies = dict()
        self.accuracy_rate = dict()
        self.rmse = 0 
        self.Movie_Rec = pd.Series(self.user_Rec, name = 'MovieID')

    def modifyRatingsDataset(self): 
        ''' Assign rating values to a matrix (userid, movieid ) '''
        for index, row in self.ratings.iterrows():
            self.SVD_matrix[ int(row[0]) , int(row[1])] = float(row[2])
        return csc_matrix(self.SVD_matrix, dtype=np.float32),self.SVD_matrix



    def UsersRating(self):  
        '''Create an empty dictionary that has key = userid and values = empty list '''
        for index, row in self.ratings.iterrows():
            self.user_Rec[int(row[0])] = list()
            self.movies_pred[int(row[0])] = list()
            self.all_movies_rec[int(row[0])] = list()
            self.intersection_movies[int(row[0])] = list()
            self.accuracy_rate[int(row[0])] = list()
            
        return self.user_Rec , self.movies_pred, self.all_movies_rec,self.intersection_movies,self.accuracy_rate


    def getMoviesSeen(self): 
        ''' Create a dictionary that has key = userid , values = list of movies that they have already rated '''
   
        for index, row in self.ratings.iterrows():
            try:
                self.movies_Seen[int(row[0])].append(int(row[1]))
            except:
                self.movies_Seen[int(row[0])] = list()
                self.movies_Seen[int(row[0])].append(int(row[1]))
        return self.movies_Seen

    
    def computeSVD(self,ratings, K):
        Ut,s,Vt = sparsesvd(ratings, K)

        dim = (len(s), len(s))
        S = np.zeros(dim, dtype = np.float32)
        for i in range(0,len(s)):
            S[i,i] = (s[i])

        U = csr_matrix(Ut.T, dtype=np.float32)
        S = csr_matrix(S, dtype=np.float32)
        Vt = csr_matrix(Vt, dtype=np.float32)
        return U,S,Vt

    def computeEstimatedRatings(self, user_Rec, U, S, Vt): 
        '''Compute Estimated Ratings by taking products of U S Vt '''
        rightTerm = S * Vt 
        estimated_Ratings = np.zeros(shape = (944, 1683), dtype = np.float16)
        for row in self.user_Rec:
            prod = U[row, :] * rightTerm
            estimated_Ratings[row, :] = ssp.coo_matrix.todense(prod)
            recom = (-estimated_Ratings[row, :]).argsort()[:250] #returns indice of values that sorted in order 
            for r in recom:
                self.all_movies_rec[row].append(r)
                if r not in movies_Seen[row]:
                    self.user_Rec[row].append(r)
                    if len(self.user_Rec[row]) == 5:
                        break
        return user_Rec, all_movies_rec,estimated_Ratings
    
    
    def getMovieNames(self,user_Rec, movies_pred, movies_list):
        ''' Replace movieIds with Movie Titles '''
        for key,value in self.user_Rec.items():
            for movie in value:
                self.movies_pred[key].append(movies_list[movie])
        return movies_pred
   

    def testaccuracy(self, intersection_movies, all_movies_rec,accuracy_rate,SVD_matrix,estimated_Ratings): #check to see movies that in moviesSeen list but not in the recommendation
        '''Calculate the Root Mean Square Error '''
        d = []
        for row in self.all_movies_rec:
            self.intersection_movies[row] = set.intersection(*[set(self.movies_Seen[row]), set(self.all_movies_rec[row])])
            for movie in intersection_movies[row]:
                self.accuracy_rate[row].append((estimated_Ratings[row, movie] - SVD_matrix[row, movie]) ** 2)
                d.append((estimated_Ratings[row, movie] - SVD_matrix[row, movie]) ** 2)
            
        self.rmse = np.sqrt(sum(d)/ sum([len(b) for b in self.accuracy_rate.values()]))
       
        return intersection_movies,accuracy_rate, self.rmse



if __name__ == '__main__':
    K = 50 

    r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
    df_ratings = pd.read_csv('ml-100k/u.data', sep='\t', names=r_cols,encoding='latin-1')
    ratings = df_ratings.drop('unix_timestamp',1)
    
    #Reading items file:
    i_cols = ['movie_id', 'movie_title' ,'release_date','video_release_date', 'IMDb_URL', 'unknown', 'Action', 'Adventure','Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy','Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
    items = pd.read_csv('ml-100k/u.item', sep='|', names = i_cols,encoding='latin-1')
    movies_list = dict()
    for index, row in items.iterrows():
        movies_list[int(row[0])] = str(row[1])
        
    #instantiate the class
    pred = MovieRecommender(ratings)
    ratings,SVD_matrix = pred.modifyRatingsDataset()
    U,S,Vt = pred.computeSVD(ratings,K)
    user_Rec,movies_pred, all_movies_rec,intersection_movies,accuracy_rate = pred.UsersRating()
    movies_Seen = pred.getMoviesSeen()
    user_Rec, all_movies_rec,estimated_Ratings = pred.computeEstimatedRatings(user_Rec,U,S,Vt)
    Movie_Rec = pred.getMovieNames(user_Rec,movies_pred, movies_list)
    intersection_movies,accuracy_rate,rmse  = pred.testaccuracy(all_movies_rec,intersection_movies,accuracy_rate,SVD_matrix,estimated_Ratings)

print( "The root squared error of ratings is", rmse)


print(Movie_Rec)