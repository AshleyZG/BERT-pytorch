# hold out seeds
# for evaluation
wrangling = ['pandas.DataFrame.dropna'，
             'pandas.DataFrame.groupby',
             'pandas.DataFrame.stack']
exploring = ['seaborn.clustermap',
             'seaborn.FacetGrid',
             'seaborn.jointplot',
             'seaborn.countplot']
modeling = ['sklearn.cluster.spectral_clustering',
            'sklearn.linear_model.SGDClassifier',
            'sklearn.covariance.GraphicalLassoCV',
            'sklearn.linear_model.MultiTaskLasso']
evaluation = ['sklearn.model_selection.validation_curve',
              'sklearn.metrics.f1_score',
              'sklearn.metrics.log_loss',
              'sklearn.metrics.precision_score']
# ======================
# 'sklearn.cluster.spectral_clustering'
# 'sklearn.covariance.GraphicalLassoCV'
# why not in test data??
