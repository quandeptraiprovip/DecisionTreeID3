import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_graphviz
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import category_encoders as ce
import graphviz
import seaborn as sns
import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings('ignore')

def load_data(file_path):

  data = pd.read_csv(file_path)
  return data

def train_decision_tree(X_train, y_train, random_state):
  clf = DecisionTreeClassifier(random_state=random_state)
  clf.fit(X_train, y_train)
  return clf

def visualize_decision_trees(max_depths, clf_list, feature_names, class_names):
    
  for i in range(len(max_depths)):
    class_names_str = class_names.astype(str)  # Convert class names to strings
    dot_data = export_graphviz(clf_list[i], out_file=None, feature_names=feature_names, class_names=class_names_str, filled=True)
    graph = graphviz.Source(dot_data)
    # graph.render(f"decision_tree_max_depth_{max_depths[i]}", format="png")
    graph = graphviz.Source(dot_data, format='png')
    graph.render(filename=f"decision_tree_max_depth_{max_depths[i]}", directory='.', cleanup=True)

def heatmap(evaluates):
  for i ,eval in enumerate(evaluates):
    data = eval
    data_array = np.array(data)

    # Create a heatmap
    sns.heatmap(data_array, annot=True, fmt='.1f', cmap='viridis')
    plt.savefig(f"heatmap{i}.png")

    # Show the plot
    # plt.show()

def accuracy_depth(X_train, y_train, X_test, y_test, max_depths):

  accuracy_scores= []
  evaluates = []
  reports = []
  clf_list = []

  for md in max_depths:
    clf = DecisionTreeClassifier(max_depth=md, random_state=0)
    clf.fit(X_train, y_train)
    clf_list.append(clf)
    y_pred= clf.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    accuracy_scores.append(accuracy_score(y_test, y_pred))
    evaluates.append(cm)
    reports.append(classification_report(y_test, y_pred))

  return accuracy_scores, evaluates, reports, clf_list

def main():
  df = load_data("nursery/nursery.data")

  col_names = ['parents', 'hase_nurs', 'form', 'children', 'housing', 'finance', 'social', 'health', 'class']
  df.columns = col_names

  X = df.drop(['class'], axis=1)
  y = df['class']

  train_proportions = [0.4, 0.6, 0.8, 0.9]
  test_proportions = [1 - prop for prop in train_proportions]

  subsets = []
  encoder = ce.OrdinalEncoder(cols=['parents', 'hase_nurs', 'form', 'children', 'housing', 'finance', 'social', 'health'])

  for train_prop, test_prop in zip(train_proportions, test_proportions):
    for _ in range(4):  
      X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_prop, random_state=42, shuffle = True)
      X_train = encoder.fit_transform(X_train)
      X_test = encoder.transform(X_test)
      subsets.append({
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'train_size': train_prop,
        'test_size': test_prop
      })



  max_depths = [None, 2, 3, 4, 5, 6, 7]
  accuracy_scores, evaluates, reports, clf_list = accuracy_depth(subsets[0]['X_train'], subsets[0]['y_train'], subsets[0]['X_test'], subsets[0]['y_test'], max_depths)
  # print(accuracy_scores)
  # for evaluate in evaluates:
  #   for row in evaluate:
  #     print(row)

  # print(accuracy_scores)
  # print(reports[0])
  # print(evaluates[0])
  # print(len(evaluates))
  visualize_decision_trees(max_depths, clf_list, df.columns[:-1], clf_list[0].classes_)
  heatmap(evaluates)


if __name__ == "__main__":
  main()