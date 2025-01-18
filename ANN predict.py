test=pd.read_csv('./xx.csv', engine='python')
X, Y=prepare_x(test)
nb_features = 1015 
X_test = np.zeros((len(X), nb_features, 1))
X_test[:, :, 0] = X[:, :nb_features]
species_prediction=model.predict(X_test)
