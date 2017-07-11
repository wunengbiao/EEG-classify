from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import svm
from sklearn.externals import joblib

# parameter named save in functions below indicate saving train model in a file.
def userRandomForestClassifier(X_train,X_test,y_train,y_test,save):
    if save:
        clf = RandomForestClassifier(n_estimators=101, criterion='gini', min_samples_split=5, max_depth=10,
                                     min_samples_leaf=1, max_features=0.7, n_jobs=3)
        clf.fit(X_train, y_train)
        # joblib.dump(clf,'rf.model')
        r = clf.score(X_test, y_test)
        return clf,r,clf.predict(X_train),clf.predict(X_test)
    else:
        clf=joblib.load('rf.model')
        clf.fit(X_train, y_train)
        r = clf.score(X_test, y_test)
        return clf,r,clf.predict(X_train),clf.predict(X_test)

def userGradientBoostingClassifier(X_train,X_test,y_train,y_test,save):
    if save:
        clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.2,
                                         max_depth=6, random_state=0).fit(X_train, y_train)
        # joblib.dump(clf,'gb.model')
        ret = clf.score(X_test, y_test)
        return clf,ret,clf.predict(X_train),clf.predict(X_test)
    else:
        clf=joblib.load('gb.model')
        ret = clf.score(X_test, y_test)
        return clf,ret,clf.predict(X_train),clf.predict(X_test)


def userGradientBoostingRegressor(X_train,X_test,y_train,y_test,save):
    if save:
        clf =GradientBoostingRegressor(n_estimators=100, max_depth=6, random_state=0)
        clf.fit(X_train,y_train)
        ret = clf.predict(X_test)
        count = 0
        l = len(y_test)
        count = 0
        for i in range(l):
            if (ret[i] < 0.5 and y_test[i] == 0) or (ret[i] >= 0.5 and y_test[i] == 1):
                count += 1

        # joblib.dump(clf,'gbr.model')
        return clf,count/l,clf.predict(X_train),clf.predict(X_test)
    else:
        clf = joblib.load('gbr.model')
        clf.fit(X_train, y_train)
        ret = clf.predict(X_test)
        count = 0
        l = len(y_test)
        count = 0
        for i in range(l):
            if (ret[i] < 0.5 and y_test[i] == 0) or (ret[i] >= 0.5 and y_test[i] == 1):
                count += 1
        return clf,count / l,clf.predict(X_train),clf.predict(X_test)


def userSVMClassifier(X_train,X_test,y_train,y_test,save):
    if save:
        clf=svm.SVC(C=1.0,cache_size=200, coef0=0.0, degree=3,gamma='auto',kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False)
        clf.fit(X_train,y_train)
        result = clf.predict(X_test)
        count = 0
        for i in range(len(y_test)):
            if result[i] == y_test[i]:
                count += 1
        # joblib.dump(clf,'svm.model')
        return clf,count/len(y_test),clf.predict(X_train),clf.predict(X_test)
    else:
        clf = joblib.load('svm.model')
        clf.fit(X_train, y_train)
        result = clf.predict(X_test)
        count = 0
        for i in range(len(y_test)):
            if result[i] == y_test[i]:
                count += 1
        return clf,count / len(y_test),clf.predict(X_train),clf.predict(X_test)



