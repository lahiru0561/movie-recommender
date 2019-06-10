import pandas as pd
import graphlab
import graphlab.numpy

# Reading users file:
u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
users = pd.read_csv('u.user', sep='|',
                    names=u_cols, encoding='latin-1')

# Reading ratings file:
r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
ratings = pd.read_csv('u.data', sep='\t',
                      names=r_cols, encoding='latin-1')

# Reading items file:
i_cols = ['movie id', 'movie title', 'release date', 'video release date', 'IMDb URL', 'unknown', 'Action', 'Adventure',
          'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
          'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
items = pd.read_csv('u.item', sep='|', names=i_cols,
                    encoding='latin-1')

print(users.shape)
print(users.head())

print(ratings.shape)
print(ratings.head())


print(items.shape)
print(items.head())


r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
ratings_train = pd.read_csv(
    'ua.base', sep='\t', names=r_cols, encoding='latin-1')
ratings_test = pd.read_csv(
    'ua.test', sep='\t', names=r_cols, encoding='latin-1')
ratings_train.shape, ratings_test.shape


train_data = graphlab.SFrame(ratings_train)
test_data = graphlab.SFrame(ratings_test)

popularity_model = graphlab.popularity_recommender.create(
    train_data, user_id='user_id', item_id='movie_id', target='rating')

    
popularity_recomm = popularity_model.recommend(users=[1, 2, 3, 4, 5], k=5)
popularity_recomm.print_rows(num_rows=25)


# Training the model for cosine similarity
item_sim_model_cosine = graphlab.item_similarity_recommender.create(
    train_data, user_id='user_id', item_id='movie_id', target='rating', similarity_type='cosine')
    
# Training the model
item_sim_model_jaccard = graphlab.item_similarity_recommender.create(
    train_data, user_id='user_id', item_id='movie_id', target='rating', similarity_type='jaccard')

item_sim_model_pearson = graphlab.item_similarity_recommender.create(
    train_data, user_id='user_id', item_id='movie_id', target='rating', similarity_type='pearson')

# Making recommendations
item_sim_recomm = item_sim_model_cosine.recommend(users=[1, 2, 3, 4, 5], k=5)
item_sim_recomm.print_rows(num_rows=25)


print("RMSE  ")
eval_popularity_model = popularity_model.evaluate_rmse(test_data, target='rating')
print("RMSE eval_popularity_model :%s"%eval_popularity_model['rmse_overall'])
#eval_popularity_model['rmse_by_item'].save('eval_popularity_model.csv',format='csv')
eval_popularity_model['rmse_by_item'].show()

eval_rmse_result_cosine = item_sim_model_cosine.evaluate_rmse(test_data, target='rating')
print("RMSE eval_rmse_result_cosine :%s"%eval_rmse_result_cosine['rmse_overall'])
#eval_rmse_result_cosine['rmse_by_item'].save('eval_rmse_result_cosine.csv',format='csv')

eval_rmse_result_jaccard = item_sim_model_jaccard.evaluate_rmse(test_data, target='rating')
print("RMSE eval_rmse_result_jaccard :%s"%eval_rmse_result_jaccard['rmse_overall'])
#eval_rmse_result_jaccard['rmse_by_item'].save('eval_rmse_result_jaccard.csv',format='csv')

eval_rmse_result_pearson = item_sim_model_pearson.evaluate_rmse(test_data, target='rating')
print("RMSE eval_rmse_result_pearson :%s"%eval_rmse_result_pearson['rmse_overall'])
#eval_rmse_result_pearson['rmse_by_item'].save('eval_rmse_result_pearson.csv',format='csv')

print("------------------------------------------------------------------")
#
#model_performance_1 = graphlab.compare(
#    test_data, [popularity_model,item_sim_model_cosine,item_sim_model_jaccard,item_sim_model_pearson])
#    
#graphlab.show_comparison(model_performance_1,[popularity_model,item_sim_model_cosine,item_sim_model_jaccard,item_sim_model_pearson])

#change to rating mean
ratings_train['rating'] = (ratings_train['rating'] -
                           ratings_train['rating'].mean())/ratings_train['rating'].std()
ratings_test['rating'] = (ratings_test['rating'] -
                          ratings_test['rating'].mean())/ratings_test['rating'].std()
train_data_mean = graphlab.SFrame(ratings_train)
test_data_mean = graphlab.SFrame(ratings_test)


# Training the model
item_sim_model_cosine_mean = graphlab.item_similarity_recommender.create(
    train_data_mean, user_id='user_id', item_id='movie_id', target='rating', similarity_type='cosine')
    
# Training the model
item_sim_model_jaccard_mean = graphlab.item_similarity_recommender.create(
    train_data_mean, user_id='user_id', item_id='movie_id', target='rating', similarity_type='jaccard')

item_sim_model_pearson_mean = graphlab.item_similarity_recommender.create(
    train_data_mean, user_id='user_id', item_id='movie_id', target='rating', similarity_type='pearson')

# Making recommendations
#item_sim_recomm_mean = item_sim_model_cosine_mean.recommend(users=[1, 2, 3, 4, 5], k=5)
#item_sim_recomm_mean.print_rows(num_rows=25)


print("RMSE after mean ")


eval_rmse_result_cosine_mean = item_sim_model_cosine_mean.evaluate_rmse(test_data_mean, target='rating')
print("RMSE eval_rmse_result_cosine_mean :%s"%eval_rmse_result_cosine_mean['rmse_overall'])
#eval_rmse_result_cosine_mean['rmse_by_item'].save('eval_rmse_result_cosine_mean.csv',format='csv')

eval_rmse_result_jaccard_mean = item_sim_model_jaccard_mean.evaluate_rmse(test_data_mean, target='rating')
print("RMSE eval_rmse_result_jaccard_mean :%s"%eval_rmse_result_jaccard_mean['rmse_overall'])
#eval_rmse_result_jaccard_mean['rmse_by_item'].save('eval_rmse_result_jaccard_mean.csv',format='csv')

eval_rmse_result_pearson_mean = item_sim_model_pearson_mean.evaluate_rmse(test_data_mean, target='rating')
print("RMSE eval_rmse_result_pearson_mean :%s"%eval_rmse_result_pearson_mean['rmse_overall'])
#eval_rmse_result_pearson_mean['rmse_by_item'].save('eval_rmse_result_pearson_mean.csv',format='csv')

print("------------------------------------------------------------------")

#
#model_performance_2 = graphlab.compare(
#    test_data, [popularity_model_mean,item_sim_model_cosine_mean,item_sim_model_jaccard_mean,item_sim_model_pearson_mean])
#    

#graphlab.show_comparison(model_performance_2)