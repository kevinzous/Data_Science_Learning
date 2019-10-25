from sklearn import tree
from sklearn import svm
from sklearn.neighbors.nearest_centroid import NearestCentroid

# [height, weight, shoe_size]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male']

# Tree 

clf1=tree.DecisionTreeClassifier()

# kNeighbours CLassifier Nearest Centroid Classifier
clf2 = NearestCentroid()

# Linear SVC Support vector machines (SVMs)
clf3 = svm.SVC()

# CLF=[clf1,clf2,clf3]
# for clf in CLF:
for clf in [clf1,clf2,clf3]:

	clf=clf.fit(X,Y)
	prediction = clf.predict([[169,60,42]])

	print prediction