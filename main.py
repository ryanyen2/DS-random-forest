import pandas as pd
from sklearn import preprocessing, tree
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
import graphviz
import timeit
import matplotlib.pyplot as plt
from statistics import mean

from sklearn.preprocessing import StandardScaler


def k_folding(ts, data_frame):
    train_data, test_data, train_label, test_label = train_test_split(data_frame.iloc[:, :-1], data_frame.iloc[:, -1],
                                                                      test_size=ts, random_state=10)
    feature_scaler = StandardScaler()
    X_train = feature_scaler.fit_transform(train_data)
    # X_test = feature_scaler.transform(test_data)
    result = []
    indexs = []
    for j in range(0, 20):
        x = 0
        for i in range(10, 1000, 20):
            classifier = RandomForestClassifier(n_estimators=i, random_state=0)
            all_accuracies = cross_val_score(estimator=classifier, X=X_train, y=train_label, cv=5)
            if j == 0:
                indexs.append(i)
            if j > 0:
                result[x] = ((result[x] + all_accuracies.mean()) / 2)
            else:
                result.append(all_accuracies.mean())
            x += 1
        # result.append((i, all_accuracies.mean(), all_accuracies.std()))
        # print(f'\n----- ^n_estimators = {i} ----\n')

    # print(sorted(result, key=lambda tup: tup[1]))
    plot_data(indexs, result, 'n_estimators', 'mean_acc_score', 'kfold-nest-acc', 20)

    # grid_param = {
    #     'n_estimators': [100, 300, 500, 800, 1000],
    #     'criterion': ['gini', 'entropy'],
    #     'max_depth': [1, 2, 3, 4, 5, 6, 7],
    #     'min_samples_leaf': [1, 2, 3, 4, 5, 6],
    # }
    # gd_sr = GridSearchCV(estimator=classifier,
    #                      param_grid=grid_param,
    #                      scoring='accuracy',
    #                      cv=5,
    #                      n_jobs=-1)
    #
    # gd_sr.fit(X_train, train_label)
    # best_parameters = gd_sr.best_params_
    # best_result = gd_sr.best_score_
    # print(best_parameters)
    # print(best_result)
    #
    # # {'criterion': 'gini', 'max_depth': 4, 'min_samples_leaf': 6, 'n_estimators': 100}
    # # 0.8305306122448979
    # df = pd.concat([pd.DataFrame(gd_sr.cv_results_["params"]),
    #                 pd.DataFrame(gd_sr.cv_results_["mean_test_score"], columns=["Accuracy"])], axis=1)
    #
    # print(df.to_csv(f'output/cv_best_{str(ts)}.csv'))


def plot_data(d1, d2, n1, n2, graph_name, times=None):
    _, ax = plt.subplots(1, 1)

    ax.plot(d1, d2)
    ax.set_title(graph_name + str(times), fontsize=18, fontweight='bold')
    ax.set_xlabel(n1, fontsize=14)
    ax.set_ylabel(n2, fontsize=14)
    ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), fontsize=13)
    ax.grid('on')
    plt.savefig(f'output/{graph_name}_{n1}_{n2}_{str(times) if times else ""}.png')
    plt.show()


def plot_bar_data(d1, d2, n1, n2, graph_name):
    # d2 = sorted(list(set(d2)), reverse=True)
    if len(d1) > 10:
        d1 = d1[0:10]
        d2 = d2[0:10]

    plt.barh(d1, d2)
    plt.xlabel(n2)
    plt.ylabel(n1)
    plt.title(graph_name)
    plt.xlim(d2[-1]-0.01, d2[0]+0.02)
    plt.grid('on')
    plt.savefig(f'output/{graph_name}_{n1}_{n2}.png')
    plt.show()


def loop_nest(data_frame, loop_times):
    train_data, test_data, train_label, test_label = train_test_split(data_frame.iloc[:, :-1], data_frame.iloc[:, -1],
                                                                      test_size=0.2)
    grid1 = []
    grid2 = []
    for j in range(0, loop_times):
        x = 0
        for i in range(10, 1000, 10):
            clf = RandomForestClassifier(n_estimators=i)
            clf.fit(train_data, train_label)
            prediction = clf.predict(test_data)
            acc_score = accuracy_score(test_label, prediction)
            if j > 0:
                grid2[x] = ((grid2[x] + acc_score) / 2)
            else:
                grid2.append(acc_score)

            if j == 0:
                grid1.append(i)
            x += 1
        print(grid2)
        # print(confusion_matrix(test_label, prediction))

    plot_data(grid1, grid2, 'n_estimators', 'Accuracy Score', 'n_estimators with accuracy', loop_times)


def construct_tree(ns, data_frame):
    train_data, test_data, train_label, test_label = train_test_split(data_frame.iloc[:, :-1], data_frame.iloc[:, -1],
                                                                      test_size=0.2, random_state=1)
    feature_names = ["pelvic incidence", "pelvic tilt", "lumbar lordosis angle", "sacral slope", "pelvic radius", "grade of spondylolisthesis"]
    class_names = ["Disk Hernia (DH)", "Spondylolisthesis (SL)", "Normal (NO)", "Abnormal (AB)"]

    for n in ns:
        clf = RandomForestClassifier(n_estimators=n)
        clf.fit(train_data, train_label)

        le = preprocessing.LabelEncoder()
        label = le.fit_transform(test_label)
        prediction = clf.predict(test_data)
        print(f'\n=================={n}===========\n')
        print(accuracy_score(test_label, prediction))
        print(confusion_matrix(test_label, prediction))
        print(clf.feature_importances_)

        print("estimators")

        prediction_scores = []
        for i, v in enumerate(clf.estimators_):
            temp_pred = v.predict(test_data)
            prediction_scores.append((i, accuracy_score(label, temp_pred), confusion_matrix(label, temp_pred)))

        # print("All_indexes>> ", [acc[0] for acc in prediction_scores])
        # print("All_scores>> ", [acc[1] for acc in prediction_scores])
        prediction_scores = sorted(prediction_scores, key=lambda tup: tup[1], reverse=False)
        plot_bar_data([str(acc[0]) for acc in prediction_scores], [float("{:.5f}".format(acc[1])) for acc in prediction_scores], 'n_estimators', 'accuracy_score', f'{n}_tree_nest_acc_worst')

        for estimator in prediction_scores[0:3]:
            print("index: ", estimator[0], "\naccuracy_score: ", estimator[1], "\nconfusion_matrix: ", estimator[2])
            dot_data = tree.export_graphviz(clf.estimators_[estimator[0]], out_file=None,
                                            feature_names=feature_names,
                                            class_names=class_names,
                                            filled=True, rounded=True, special_characters=True)

            graph = graphviz.Source(dot_data)
            graph.render(f'output/{str(n)}_tree_{str(estimator[0])}', view=True)
            print(clf.estimators_[estimator[0]].feature_importances_)


def plot_time_acc(index, rf_data, nb_data, name):
    print(index, '\n', rf_data, '\n', nb_data)
    fig, ax = plt.subplots()
    ax.plot(index, rf_data, marker="o")
    ax.set_xlabel("estimators(n)")
    ax.set_ylabel("accuracy score" if name == 'ac' else "time (100s)")
    ax.plot(index, nb_data, marker="o")
    plt.title(f'Naive Bayes v.s Random Forest - {name}')
    plt.grid('on')
    ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), fontsize=13)
    plt.savefig(f'output/nb_vs_rf_{name}.png')
    plt.show()


def naive_bayes(data_frame):
    train_data, test_data, train_label, test_label = train_test_split(data_frame.iloc[:, :-1], data_frame.iloc[:, -1],
                                                                      test_size=0.2, random_state=1)

    feature_scaler = StandardScaler()
    X_train = feature_scaler.fit_transform(train_data)
    rf_result = []
    nb_result = []
    rf_time = []
    nb_time = []
    index = []

    for j in range(0, 30):
        x = 0
        for i in range(10, 1000, 50):
            # rf_start = timeit.default_timer()
            clf = RandomForestClassifier(n_estimators=i, random_state=0)
            all_accuracies = cross_val_score(estimator=clf, X=X_train, y=train_label, cv=5)
            # clf.fit(train_data, train_label)
            # rf_pred = clf.predict(test_data)
            if j == 0:
                index.append(i)
                # nb_start = timeit.default_timer()
                nb = GaussianNB()
                nb = nb.fit(train_data, train_label)
                nb_pred = nb.predict(test_data)
                # nb_stop = timeit.default_timer()
                nb_result.append(accuracy_score(test_label, nb_pred))
            if j > 0:
                rf_result[x] = ((rf_result[x] + all_accuracies.mean()) / 2)
            else:
                rf_result.append(all_accuracies.mean())
            x += 1
            # rf_stop = timeit.default_timer()

            # rf_result.append(accuracy_score(test_label, rf_pred))

            # rf_time.append((rf_stop-rf_start)*100)
            # nb_time.append((nb_stop-nb_start)*100)

            # print("Accuracy Score:\t", accuracy_score(test_label, nb_pred), '\t', accuracy_score(test_label, rf_pred))
            # print("Run Time:\t", nb_stop-nb_start, '\t', rf_stop-rf_start)
            # # if not j:
            # print("Confusion Matrix:\t", confusion_matrix(test_label, nb_pred), '\t', confusion_matrix(test_label, rf_pred))

    plot_time_acc(index, rf_result, nb_result, 'ac')
    # plot_time_acc(index, rf_time, nb_time, 't')
    # print("Mean Random forest accuracy: ", mean(rf_result))


if __name__ == '__main__':
    vertebral_data = pd.read_csv('vertebral_column_data/column_3C.dat', header=None, sep=' ')
    # loop_nest(vertebral_data, 10)
    # k_folding(0.2, vertebral_data)
    # construct_tree([800], vertebral_data)
    naive_bayes(vertebral_data)
