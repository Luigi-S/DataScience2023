import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge

from sklearn.neighbors import KNeighborsRegressor
from yellowbrick.model_selection import ValidationCurve

def knn_tuning(X_train, y_train, min_k=1, max_k =10):
    # Set up the KNeighborsRegressor model
    model = KNeighborsRegressor()

    # Set the parameter range for tuning
    param_range = np.arange(min_k, max_k)

    # Create a ValidationCurve visualizer
    visualizer = ValidationCurve(
        model, param_name="n_neighbors", param_range=param_range,
            scoring="r2", n_jobs=-1
    )

    # Fit the visualizer and display the plot
    visualizer.fit(X_train, y_train)
    visualizer.show()



def ridge_tuning(X_train, y_train):
    # Set up the Ridge model
    model = Ridge()

    # Set the parameter range for tuning
    param_range = np.logspace(-6, 6, 13)

    # Create a ValidationCurve visualizer
    visualizer = ValidationCurve(
        model, param_name="alpha", param_range=param_range,
        scoring="r2", cv=5, n_jobs=-1,
        logx=True
    )

    # Fit the visualizer and display the plot
    visualizer.fit(X_train, y_train)
    visualizer.show()
    # Print the optimal value
    optimal_alpha = visualizer.optimal_value_
    print(f"Optimal alpha value: {optimal_alpha}")