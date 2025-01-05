
def main():
    #import necessary libraries 
    from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay, accuracy_score
    from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.decomposition import PCA
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

    #data wrangling steps:
    #load file and skip commented out rows
    file_path = 'cumulative_2023.11.07_13.44.30.csv'
    df_kepler = pd.read_csv(file_path, skiprows=41)
    #print(df_kepler.info())
    #drop duplicates
    df_kepler = df_kepler.drop_duplicates()
    #print(df_kepler.info())
    #map candidate and false positive to 1 or 0
    df_kepler['koi_pdisposition'] = df_kepler['koi_pdisposition'].map({'CANDIDATE': 1, 'FALSE POSITIVE': 0})
    #column filter- only include those listed in assignment description 
    df_kepler = df_kepler[['koi_pdisposition',
    'koi_period',
    'koi_eccen',
    'koi_duration',
    'koi_prad',
    'koi_sma',
    'koi_incl',
    'koi_teq',
    'koi_dor',
    'koi_steff',
    'koi_srad',
    'koi_smass'
]]
    print(df_kepler)
   #drop null columns then rows
    df_kepler = df_kepler.dropna(axis=1, how='all').dropna(how='any') 
    #Drop columns that are all zero
    df_kepler= df_kepler.loc[:, (df_kepler != 0).any(axis=0)]
#drop column disposition (not needed)- added column filter instead
    #df_kepler = df_kepler.drop(columns='koi_disposition')
#check data
    print(df_kepler.info())

    #separate feature and target values 
    X = df_kepler.drop(columns=['koi_pdisposition'])
    y = df_kepler['koi_pdisposition']
    print(df_kepler)

    #test train split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    #calculate the total combinations and n_iter for each model:
    #define parameter grids for each model based on hyperparameters given
    param_grid_lr = {}  #empty- no hyperparameters

    param_grid_knn = {
        'model__n_neighbors': range(1, int(1.5 * np.sqrt(len(X_train))) + 1)
    }

    param_grid_dt = {
        'model__criterion': ['gini', 'entropy'],
        'model__max_depth': range(3, 16), 
        'model__min_samples_leaf': range(1, 11) 
    }

    param_grid_svc = {
        'model__kernel': ['rbf'],
        'model__C': [0.1, 1, 10, 100],
        'model__gamma': [0.1, 1, 10]
    }
    #logistic regression n_iter- no hyperparameters so n_iter is 1
    n_iter_lr = 1

    #K-nearest neighbors n_iter
    knn_combinations = len(param_grid_knn['model__n_neighbors'])
    n_iter_knn = max(1, int(0.1 * knn_combinations)+1)  #searching 10% of total combinations(+1 is because it rounds down if decimal and cannot run model decimal amount- at LEAST 10%)

    #decision Tree n_iter
    dt_combinations = (
        len(param_grid_dt['model__criterion']) *
        len(param_grid_dt['model__max_depth']) *
        len(param_grid_dt['model__min_samples_leaf'])
    )
    n_iter_dt = max(1, int(0.1 * dt_combinations)+1)  #searching 10% of total combinations(+1 is because it rounds down if decimal and cannot run model decimal amount- at LEAST 10%)

    #SVC n_iter
    svc_combinations = (
        len(param_grid_svc['model__kernel']) *
        len(param_grid_svc['model__C']) *
        len(param_grid_svc['model__gamma'])
    )
    n_iter_svc = max(1, int(0.1 * svc_combinations)+1)  #searching 10% of total combinations(+1 is because it rounds down if decimal and cannot run model decimal amount- at LEAST 10%)

    #results for all n_iter values for diff. models 
    print(f"n_iter for Logistic Regression: {n_iter_lr}")
    print(f"n_iter for K-Nearest Neighbors: {n_iter_knn}")
    print(f"n_iter for Decision Tree: {n_iter_dt}")
    print(f"n_iter for SVC: {n_iter_svc}")
    print('\n ------------- \n')

#store model names and accuracies for NON PCA 
    model_names = []
    model_accuracies = []

    #logistic Regression No PCA
    print("Logistic Regression No PCA")
    #make pipeline 
    pipeline_lr = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LogisticRegression())
    ])
    #random search for hyper parameter tuning
    random_search_lr = RandomizedSearchCV(
        estimator=pipeline_lr,
        param_distributions=param_grid_lr,
        n_iter=n_iter_lr,
        scoring='accuracy',
    )
    #fit the RandomizedSearchCV
    random_search_lr.fit(X_train, y_train)
    #print best hyperparameters
    print("Best Parameters:", random_search_lr.best_params_)
    #predictions with best model 
    best_lr = random_search_lr.best_estimator_
    y_pred_lr = best_lr.predict(X_test)
    #calc and print accuracy
    accuracy = accuracy_score(y_test, y_pred_lr)
    print("Logistic Regression Accuracy:", accuracy)
#append to arrays
    model_names.append("Logistic Regression No PCA")
    model_accuracies.append(accuracy)


    #K-Nearest Neighbors No PCA
    print("K-Nearest Neighbors No PCA")
    #make mdoel pipeline
    pipeline_knn = Pipeline([
        ('scaler', StandardScaler()),
        ('model', KNeighborsClassifier())
    ])
  #random search for hyper parameter tuning
    random_search_knn = RandomizedSearchCV(
        estimator=pipeline_knn,
        n_iter=n_iter_knn,
        param_distributions=param_grid_knn,
        scoring='accuracy',
    )
    #fit the RandomizedSearchCV
    random_search_knn.fit(X_train, y_train)
#print best hyperparameters
    print("Best Parameters:", random_search_knn.best_params_)
    #predictions with best model 
    best_knn = random_search_knn.best_estimator_
    y_pred_knn = best_knn.predict(X_test)
    #calc and print accuracy
    accuracy = accuracy_score(y_test, y_pred_knn)
    print("K-Nearest Neighbors Accuracy:", accuracy)
    #append to arrays
    model_names.append("K-Nearest Neighbors No PCA")
    model_accuracies.append(accuracy)


    #decision Tree No PCA
    print("Decision Tree No PCA")
    #make mdoel pipeline
    pipeline_dt = Pipeline([
        ('scaler', StandardScaler()),
        ('model', DecisionTreeClassifier())
    ])
    #random search for hyper parameter tuning  
    random_search_dt = RandomizedSearchCV(
        estimator=pipeline_dt,
        n_iter=n_iter_dt,
        param_distributions=param_grid_dt,
        scoring='accuracy',
    )
    #fit the RandomizedSearchCV   
    random_search_dt.fit(X_train, y_train)
    #print best parameters
    print("Best Parameters:", random_search_dt.best_params_)
    #predictions with best model 
    best_dt = random_search_dt.best_estimator_
    y_pred_dt = best_dt.predict(X_test)
    #calc and print accuracy
    accuracy = accuracy_score(y_test, y_pred_dt)
    print("Decision Tree Accuracy:", accuracy)
#append to arrays
    model_names.append("Decision Tree No PCA")
    model_accuracies.append(accuracy)

    #SVC No PCA
    print("SVC No PCA")
    #make pipeline
    pipeline_svc = Pipeline([
        ('scaler', StandardScaler()),
        ('model', SVC())
    ])
    #random search for hyper parameter tuning  
    random_search_svc = RandomizedSearchCV(
        estimator=pipeline_svc,
        n_iter=n_iter_svc,
        param_distributions=param_grid_svc,
        scoring='accuracy',
    )
    #fit the RandomizedSearchCV 
    random_search_svc.fit(X_train, y_train)
    #print best parameters
    print("Best Parameters:", random_search_svc.best_params_)
    #predictions with best model 
    best_svc = random_search_svc.best_estimator_
    y_pred_svc = best_svc.predict(X_test)
    #calc and print accuracy   
    accuracy = accuracy_score(y_test, y_pred_svc)
    print("SVC Accuracy:", accuracy)
    #append to arrays
    model_names.append("SVC No PCA")
    model_accuracies.append(accuracy)

    #find the best model
    best_model_index = model_accuracies.index(max(model_accuracies))
    best_model_name = model_names[best_model_index]
    best_model_accuracy = model_accuracies[best_model_index]
#print best model and accuracy for non pca models 
    print("\nBest Model:")
    print(f"{best_model_name} with Accuracy: {best_model_accuracy:.4f}")
    print('\n ------------- \n')

    #plot PCA components and variance to find the elbow value:
    #standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    #PCA
    pca = PCA()
    pca.fit(X_scaled)
    #variance - x axis
    cumulative_explained_variance = pca.explained_variance_ratio_.cumsum()

    #create plot of variance and n componenrs
    fig, ax = plt.subplots()
    ax.plot(range(1, len(cumulative_explained_variance) + 1), cumulative_explained_variance)
    ax.set_xlabel('Number of Components')
    ax.set_ylabel('Cumulative Explained Variance')
    ax.set_title('Explained Variance vs. Number of Components')
    plt.tight_layout()
    plt.savefig('elbow.png')
    plt.show()
    
    #initialize arrays to store model names and accuracies
    pca_model_names = []
    pca_model_accuracies = []

    #MODELS WITH PCA
    #Logistic Regression WITH PCA
    print("Logistic Regression with PCA")
    #create pipeline
    pca_pipeline_lr = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=6)),
        ('model', LogisticRegression())
    ])
    #random search for hyper parameter tuning    
    random_search_lr = RandomizedSearchCV(
        estimator=pca_pipeline_lr,
        param_distributions=param_grid_lr,
        n_iter=n_iter_lr,
        scoring='accuracy',
    )
    #fit the RandomizedSearchCV 
    random_search_lr.fit(X_train, y_train)
    #print best parameters
    print("Best Parameters:", random_search_lr.best_params_)
#predictions with best model 
    best_lr = random_search_lr.best_estimator_
    y_pred_lr = best_lr.predict(X_test)
    #calc and print accuracy
    accuracy = accuracy_score(y_test, y_pred_lr)
    print("Logistic Regression Accuracy:", accuracy)
    #append to array
    pca_model_names.append("Logistic Regression PCA")
    pca_model_accuracies.append(accuracy)


    # K-Nearest Neighbors WITH PCA
    print("K-Nearest Neighbors with PCA")
    #create pipeline
    pca_pipeline_knn = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=6)),
        ('model', KNeighborsClassifier())
    ])
    #random search for hyper parameter tuning    
    random_search_knn = RandomizedSearchCV(
        estimator=pca_pipeline_knn,
        n_iter=n_iter_knn,
        param_distributions=param_grid_knn,
        scoring='accuracy',
    )
    #fit the RandomizedSearchCV    
    random_search_knn.fit(X_train, y_train)
    #print best parameters
    print("Best Parameters:", random_search_knn.best_params_)
#predictions with best model 
    best_knn = random_search_knn.best_estimator_
    y_pred_knn = best_knn.predict(X_test)
    #calc and print accuracy
    accuracy = accuracy_score(y_test, y_pred_knn)
    print("K-Nearest Neighbors Accuracy:", accuracy)
    #append to arrays
    pca_model_names.append("K-Nearest Neighbors PCA")
    pca_model_accuracies.append(accuracy)


    # Decision Tree WITH PCA
    print("Decision Tree with PCA")
#make pipeline
    pca_pipeline_dt = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=6)),
        ('model', DecisionTreeClassifier())
    ])
    #random search for hyper parameter tuning      
    random_search_dt = RandomizedSearchCV(
        estimator=pca_pipeline_dt,
        n_iter=n_iter_dt,
        param_distributions=param_grid_dt,
        scoring='accuracy',
    )
    #fit the RandomizedSearchCV   
    random_search_dt.fit(X_train, y_train)
    #print best parameters
    print("Best Parameters:", random_search_dt.best_params_)
#predictions with best model
    best_dt = random_search_dt.best_estimator_
    y_pred_dt = best_dt.predict(X_test)
    #calc and print accuracy
    accuracy = accuracy_score(y_test, y_pred_dt)
    print("Decision Tree Accuracy:", accuracy)
    #append to array
    pca_model_names.append("Decision Tree PCA")
    pca_model_accuracies.append(accuracy)


    # SVC WITH PCA
    print("SVC with PCA")
    #make pipeline 
    pca_pipeline_svc = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=6)),
        ('model', SVC())
    ])
    #random search for hyper parameter tuning    
    random_search_svc = RandomizedSearchCV(
        estimator=pca_pipeline_svc,
        n_iter=n_iter_svc,
        param_distributions=param_grid_svc,
        scoring='accuracy',
    )
    #fit the RandomizedSearchCV   
    random_search_svc.fit(X_train, y_train)
    #print best parameters
    print("Best Parameters:", random_search_svc.best_params_)
#predictions with best model
    best_svc = random_search_svc.best_estimator_
    y_pred_svc = best_svc.predict(X_test)
    #calc and print accuracy
    accuracy = accuracy_score(y_test, y_pred_svc)
    print("SVC Accuracy:", accuracy)
    #append to arrays
    pca_model_names.append("SVC PCA")
    pca_model_accuracies.append(accuracy)

    #find the best model WITH pca
    pca_best_model_index = pca_model_accuracies.index(max(pca_model_accuracies))
    pca_best_model_name = pca_model_names[pca_best_model_index]
    pca_best_model_accuracy = pca_model_accuracies[pca_best_model_index]

#print best model with PCA Data
    print("\nBest Model:")
    print(f"{pca_best_model_name} with Accuracy: {pca_best_model_accuracy:.4f}")
    print('\n ------------- \n')

#compare models and print accuracies
    print("\nComparison of Best Models:")
    print(f"Best Non-PCA Model: {best_model_name} with Accuracy: {best_model_accuracy:.4f}")
    print(f"Best PCA Model: {pca_best_model_name} with Accuracy: {pca_best_model_accuracy:.4f}")

#compare the best PCA and non-PCA models
 #Check if the highest accuracy of non-PCA models is better than PCA models
    if max(model_accuracies) > max(pca_model_accuracies):
        #execute if non pca if larger/better
        print("Best Model: Non-PCA")
        #find model name
        final_model_name = model_names[model_accuracies.index(max(model_accuracies))]
        #select corresponding pipeline
        final_pipeline = [pipeline_lr, pipeline_knn, pipeline_dt, pipeline_svc][model_accuracies.index(max(model_accuracies))]
        #select corresponding param grid 
        final_param_grid = [param_grid_lr, param_grid_knn, param_grid_dt, param_grid_svc][model_accuracies.index(max(model_accuracies))]
    else:
    #execute if pca if larger/netter
        print("Best Model: PCA")
        #find model name
        final_model_name = pca_model_names[pca_model_accuracies.index(max(pca_model_accuracies))]
        #select corresponding pipeline
        final_pipeline = [pca_pipeline_lr, pca_pipeline_knn, pca_pipeline_dt, pca_pipeline_svc][pca_model_accuracies.index(max(pca_model_accuracies))]
        #select corresponding param grid
        final_param_grid = [param_grid_lr, param_grid_knn, param_grid_dt, param_grid_svc][pca_model_accuracies.index(max(pca_model_accuracies))]
#print final mdoel
    print(f"Selected Model: {final_model_name}")

    print(f"\nGrid Search on {final_model_name}:")
    #perform GridSearchCV on the selected model
    grid_search = GridSearchCV(
        estimator=final_pipeline,
        param_grid=final_param_grid,
        scoring='accuracy',
    )
    #fit gridsearch cv
    grid_search.fit(X_train, y_train)

#evaluate the final model
    final_model = grid_search.best_estimator_
    y_pred = final_model.predict(X_test)
#define accuracy
    accuracy = accuracy_score(y_test, y_pred)
#print final optimized model, accuracy and parameters
    print(f"\nFinal Optimized Model: {final_model_name}")
    print(f"Accuracy: {accuracy:.4f}")
    print("Best Parameters:", grid_search.best_params_)

#display confusion matrix:
#plot confusion matriz 
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    disp.plot(cmap="Blues", ax=ax)  
    ax.set_title(f"Confusion Matrix: {final_model_name}")
    plt.show() 
    plt.savefig('confusion') 

#classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
#this is for question 4 to find which attribute most significantly influences whether or not object is exoplanet
    from sklearn.inspection import permutation_importance 
    res = permutation_importance(final_model, X, y, n_repeats=10, random_state=0,scoring='accuracy')
    print(res.importances_mean)

    #add feature names and importance mean scores into df and sort ascending
    importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance Mean': res.importances_mean
    }).sort_values(by='Importance Mean', ascending=False)
    print(importance_df)

#find feature with highest importance mean- most influence
    highest_importance_row = importance_df.iloc[0] #first row of df

    #extract the feature name and importance mean
    highest_feature = highest_importance_row['Feature']
    highest_importance_mean = highest_importance_row['Importance Mean']

    #print results
    print(f"The feature with the highest importance mean is: {highest_feature}")
    print(f"Importance Mean: {highest_importance_mean:.4f}")

    print(
#answer reflection questions
    """
    Reflection:
    1.PCA did not improve the results compared to the non-PCA models. The best non-PCA model, the Decision Tree classifier, achieved an accuracy of 82.34%, while the best PCA-based model, the SVC classifier, achieved an accuracy of 76.09%. After hyperparameter optimization using GridSearchCV, the Decision Tree model without PCA was still better with an optimized accuracy of 81.74%.
    2.PCA didn't improve the results because of this dataset's specific characteristics and how the models work. PCA simplifies the data by combining features into new components that capture the most variation, but this process can hide important details specific to the problem, which are usually important for classification. For example, attributes like planetary radius (koi_prad) and distance over star radius (koi_dor) are directly interpretable and important for identifying exoplanets. By combining these attributes into different principal components, PCA may decrease their individual significance separately. PCA works best when the dataset has many similar or highly related features, but that doesn't seem to be the case here. The features in this dataset are on their own distinct and already well-suited for the classification task, which is why PCA did not offer much improvement.
    3.The model performed well in classifying the two labels but showed small differences in performance. For label 0 (false positive/not exoplanet), precision the percentage of correctly predicted false positives out of all predicted false positives was 0.84, while recall the percentage of actual false positives correctly identified was slightly lower at 0.76. This resulted in an F1-score, a balance of precision and recall, of 0.80. For label 1 (candidate/planet), precision was 0.80, recall was higher at 0.87, and the F1-score was 0.83. These results indicate that the model performed slightly better for label 1 because it was more effective at identifying all candidates (higher recall). The lower recall for label 0 suggests that the model has more false negatives. This imbalance is also evident in the confusion matrix, where label 1 was classified more accurately. Overall, the model achieved good results but could improve its ability to identify false positives.
    4.Based on the results of permutation importance analysis, the most influential attribute for determining whether an object is an exoplanet is koi_prad (planetary radius), which had the highest importance mean score of 0.1875. This score indicates that koi_prad has the strongest impact on the model's accuracy when its values are shuffled, showing its importance in the classification process.
    5.The koi_prad attribute represents the planetary radius of the object. This attribute is likely the most influential because the size of a celestial object is an important characteristic in distinguishing exoplanets from other types of objects taht are not actual planets, such as stars or noise in the data. Exoplanets are probably all within similar size ranges, with smaller radii suggesting smaller planets and larger radii indicating larger planets. Its importance is shown by its logical link to the process of classifying planets (lookign at size of an object) and its high score in the permutation importance analysis.
    """
    )
  

if __name__ == '__main__':
    main()
git remote add origin https://github.com/zfournierswire/NASA-Kepler.git
git branch -M main
git push -u origin main
