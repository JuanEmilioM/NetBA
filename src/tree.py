from pandas import read_csv
from sklearn import tree
from sklearn.ensemble import BaggingRegressor, RandomForestClassifier, AdaBoostClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
import numpy as np

def preproccesing (data, split, normalization='', shuffle=True):
    if shuffle:
        np.random.shuffle(data)
    
    N = data.shape[0]
    xtr = data[:int(split*N),:-1]
    xte = data[int(split*N):,:-1]
    ytr = data[:int(split*N):, -1:]
    yte = data[int(split*N):, -1:]
    
    if normalization == 'min_max':
        scaler = MinMaxScaler(feature_range=(0.,1.))
        xtr = scaler.fit_transform(xtr)
        xte = scaler.fit_transform(xte)
    elif normalization == 'standard':
        standarizer = StandardScaler()
        xtr = standarizer.fit_transform(xtr)
        xte = standarizer.fit_transform(xte)
    elif normalization != '':
        raise RuntimeError('normalization argument not valid')
    
    return xtr, ytr, xte, yte


def plot_tree (tree_, title, labels, class_names = None):
    figure = plt.figure()
    figure.dpi = 600
    figure.figsize = [20,20]
    
    tree.plot_tree(
        tree_, 
        max_depth = None,
        proportion = True,
        label = None, 
        feature_names = labels,
        class_names = class_names,
        filled = True, 
        rounded = True,
        fontsize = 6,
        impurity = False
    ) 
    
    figure.savefig('images/' + title + '.jpg')
    plt.close()

def main ():
    data = read_csv('files/2016-17.csv')
    features = list(data.columns)
    data = data.to_numpy()
    xtr, ytr, xte, yte  = preproccesing(data, 0.8)
    ytr, yte = ytr.ravel(), yte.ravel()
    
    # random forest classifier
    forest = RandomForestClassifier(
        n_estimators = 150,
        max_depth = 8, 
        min_samples_split = 2,
        max_features = 'auto',
        bootstrap = True,
        random_state = 0,
        verbose = 0,        
    )
    forest.fit(xtr, ytr)
    pred = forest.predict(xte)
    print('Random forest: {}'.format((pred == yte).mean()))
       
     # ada boost classifier
    hada = AdaBoostClassifier(
        # base_estimator = tree.DecisionTreeClassifier(max_depth=8),
        n_estimators = 500, 
        learning_rate = .05,
        random_state = 0
    )
    hada.fit(xtr, ytr)
    pred = hada.predict(xte)
    print('AdaBoost: {}'.format((pred == yte).mean()))
    
    return 0

if __name__ == '__main__':
    main()
