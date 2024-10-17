import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.inspection import DecisionBoundaryDisplay

# Load dataset
data = pd.read_csv('Heart.csv')

# Drop nan values 
print('data shape before: ', data.shape)
data = data.dropna()
print('data shape after: ', data.shape)

x1 = data['Age']
x2 = data['MaxHR']
y = data['AHD']
# Convert 'Yes' and 'No' to 1 and 0
y = y.replace('Yes', 1)
y = y.replace('No', 0)

# Split dataset into training set and test set
X = np.array(list(zip(x1, x2)))
Y = np.array(y)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

# Plot the data training and test set
fig, (ax1, ax2) = plt.subplots(1, 2)

scatter = ax1.scatter(X_train[:, 0], X_train[:, 1], c=Y_train, s=30, cmap=plt.cm.Paired)
handles, labels = scatter.legend_elements(num=1, prop="colors")
labels = ['Yes' if label == '$\\mathdefault{1}$' else 'No' for label in labels]
legend = ax1.legend(handles, labels, title="AHD", loc='upper right')
ax1.add_artist(legend)
ax1.set_xlabel('Age')
ax1.set_ylabel('MaxHR')
ax1.set_title('Training set')

scatter = ax2.scatter(X_test[:, 0], X_test[:, 1], c=Y_test, s=30, cmap=plt.cm.Paired)
handles, labels = scatter.legend_elements(num=1, prop="colors")
labels = ['Yes' if label == '$\\mathdefault{1}$' else 'No' for label in labels]
legend = ax2.legend(handles, labels, title="AHD", loc='upper right')
ax2.add_artist(legend)
ax2.set_xlabel('Age')
ax2.set_ylabel('MaxHR')
ax2.set_title('Test set')

plt.tight_layout()
plt.savefig('feauture_space.png')
#plt.show()
plt.clf()


# Create a SVM Classifier
clf_linear = svm.SVC(kernel='linear', C=1.0)
clf_linear.fit(X_train, Y_train)
Y_pred = clf_linear.predict(X_test)
accuracy_linear = accuracy_score(Y_test, Y_pred)
print(f'Accuracy of linear kernel: {100*accuracy_linear:.2f}%')

clf_polynomial = svm.SVC(kernel='poly', degree=3, C=1.0)
clf_polynomial.fit(X_train, Y_train)
Y_pred = clf_polynomial.predict(X_test)
accuracy_polynomial = accuracy_score(Y_test, Y_pred)
print(f'Accuracy of polynomial kernel: {100*accuracy_polynomial:.2f}%')

clf_radial = svm.SVC(kernel='rbf', C=1.0)
clf_radial.fit(X_train, Y_train)
Y_pred = clf_radial.predict(X_test)
accuracy_radial = accuracy_score(Y_test, Y_pred)
print(f'Accuracy of radial kernel: {100*accuracy_radial:.2f}%')

# Save accuracy to a file
with open('accuracy.txt', 'w') as f:
    f.write(f'Accuracy of linear kernel: {100*accuracy_linear:.2f}%\n')
    f.write(f'Accuracy of polynomial kernel: {100*accuracy_polynomial:.2f}%\n')
    f.write(f'Accuracy of radial kernel: {100*accuracy_radial:.2f}%\n')

scatter = plt.scatter(X_train[:, 0], X_train[:, 1], c=Y_train, s=30, cmap=plt.cm.Paired)
handles, labels = scatter.legend_elements(num=1, prop="colors")
labels = ['Yes' if label == '$\\mathdefault{1}$' else 'No' for label in labels]
legend = plt.legend(handles, labels, title="AHD", loc='upper right')
plt.gca().add_artist(legend)
ax = plt.gca()
DecisionBoundaryDisplay.from_estimator(
        clf_linear,
        X_train,
        plot_method='contour',
        colors='k',
        levels=[-1, 0, 1],
        alpha=0.5,
        linestyles=['--', '-', '--'],
        ax=ax,
)

# Plot support vectors for linear kernel
ax.scatter(
        clf_linear.support_vectors_[:, 0],
        clf_linear.support_vectors_[:, 1],
        s=100,
        linewidth=1,
        facecolors='none',
        edgecolors='k',
)
plt.xlabel('Age')
plt.ylabel('MaxHR')
plt.savefig('linear_kernel.png')
#plt.show()
plt.clf()

scatter = plt.scatter(X_train[:, 0], X_train[:, 1], c=Y_train, s=30, cmap=plt.cm.Paired)
handles, labels = scatter.legend_elements(num=1, prop="colors")
labels = ['Yes' if label == '$\\mathdefault{1}$' else 'No' for label in labels]
legend = plt.legend(handles, labels, title="AHD", loc='upper right')
plt.gca().add_artist(legend)
ax = plt.gca()
DecisionBoundaryDisplay.from_estimator(
        clf_polynomial,
        X_train,
        plot_method='contour',
        colors='k',
        levels=[-1, 0, 1],
        alpha=0.5,
        linestyles=['--', '-', '--'],
        ax=ax,
)

# Plot support vectors for polynomial kernel
ax.scatter(
        clf_polynomial.support_vectors_[:, 0],
        clf_polynomial.support_vectors_[:, 1],
        s=100,
        linewidth=1,
        facecolors='none',
        edgecolors='k',
)
plt.xlabel('Age')
plt.ylabel('MaxHR')
plt.savefig('polynomial_kernel.png')
#plt.show()
plt.clf()

scatter = plt.scatter(X_train[:, 0], X_train[:, 1], c=Y_train, s=30, cmap=plt.cm.Paired)
handles, labels = scatter.legend_elements(num=1, prop="colors")
labels = ['Yes' if label == '$\\mathdefault{1}$' else 'No' for label in labels]
legend = plt.legend(handles, labels, title="AHD", loc='upper right')
plt.gca().add_artist(legend)
ax = plt.gca()
DecisionBoundaryDisplay.from_estimator(
        clf_radial,
        X_train,
        plot_method='contour',
        colors='k',
        levels=[-1, 0, 1],
        alpha=0.5,
        linestyles=['--', '-', '--'],
        ax=ax,
)

# Plot support vectors for radial kernel
ax.scatter(
        clf_radial.support_vectors_[:, 0],
        clf_radial.support_vectors_[:, 1],
        s=100,
        linewidth=1,
        facecolors='none',
        edgecolors='k',
)
plt.xlabel('Age')
plt.ylabel('MaxHR')
plt.savefig('radial_kernel.png')
#plt.show()
plt.clf()

# Title for the plots
titles = ['SVC with linear kernel',
            'SVC with polynomial kernel',
            'SVC with radial basis function kernel',
]

clf_list = [clf_linear, clf_polynomial, clf_radial]

# Plot the decision boundary for training set
fig, sub = plt.subplots(1, 3, figsize=(10, 5))
plt.subplots_adjust(wspace=0.4, hspace=0.4)

for clf, title, ax in zip(clf_list, titles, sub.flatten()):
    DecisionBoundaryDisplay.from_estimator(
            clf,
            X_train,
            response_method='predict',
            cmap=plt.cm.coolwarm,
            alpha=0.8,
            ax=ax,
            xlabel='Age',
            ylabel='MaxHR',
    )
    scatter = ax.scatter(
            X_train[:, 0],
            X_train[:, 1],
            c=Y_train,
            cmap=plt.cm.coolwarm,
            s=20,
            edgecolors='k',
    )
    
    handles, labels = scatter.legend_elements(num=1, prop="colors")
    labels = ['Yes' if label == '$\\mathdefault{1}$' else 'No' for label in labels]
    legend = ax.legend(handles, labels, title="AHD", loc='upper right', framealpha=0.5)
    ax.add_artist(legend)
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(title)

plt.savefig('training_decision_boundary.png')
#plt.show()
plt.clf()

# Plot the decision boundary for test set
fig, sub = plt.subplots(1, 3, figsize=(10, 5))
plt.subplots_adjust(wspace=0.4, hspace=0.4)

for clf, title, ax in zip(clf_list, titles, sub.flatten()):
    DecisionBoundaryDisplay.from_estimator(
            clf,
            X_test,
            response_method='predict',
            cmap=plt.cm.coolwarm,
            alpha=0.8,
            ax=ax,
            xlabel='Age',
            ylabel='MaxHR',
    )
    scatter = ax.scatter(
            X_test[:, 0],
            X_test[:, 1],
            c=Y_test,
            cmap=plt.cm.coolwarm,
            s=20,
            edgecolors='k',
    )
    
    handles, labels = scatter.legend_elements(num=1, prop="colors")
    labels = ['Yes' if label == '$\\mathdefault{1}$' else 'No' for label in labels]
    legend = ax.legend(handles, labels, title="AHD", loc='upper right', framealpha=0.5)
    ax.add_artist(legend)
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(title)

plt.savefig('test_decision_boundary.png')
#plt.show()
plt.clf()

#end of svm.py

