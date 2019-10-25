import numpy as np
from lightfm.datasets import fetch_movielens
from lightfm import LightFM

#fetch date and format it
data = fetch_movielens(min_rating=5.0)

#pring training and testing data
print(repr(data['train']))
print(repr(data['test']))
	#dictionnaries with strings to retrive data USER with rating 

#create model
model = LightFM(loss="warp")
	#weighted approxiamte-rank pairwise, create recomm, uses the gradient descent algorithm to predict iteratively 3.50

#train model
model.fit(data['train'],epochs=30,num_threads=2)

def sample_recommendation(model, data, user_ids):
	#number of user and movies in training date
	n_users, n_items = data['train'].shape

	#generate recommendation for each user we input
	for user_id in user_ids:

		#movies they already like
		known_positives = data['item_labels'][data['train'].tocsr()[user_id].indices]

        #movies our model predicts they will like
        scores = model.predict(user_id, np.arange(n_items))
        #rank them in order of most liked to least
        top_items = data['item_labels'][np.argsort(-scores)]

        #print out the results
        print("User %s" % user_id)
        print("     Known positives:")

        for x in known_positives[:3]:
            print("        %s" % x)

        print("     Recommended:")

        for x in top_items[:3]:
			print("        %s" % x)


sample_recommendation(model, data, [3, 25, 23])