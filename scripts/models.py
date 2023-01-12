import collections
import gc
import random

import lace
import nest_asyncio
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_federated as tff
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from config import Config
from EPM import EffortAwareEvaluation

nest_asyncio.apply()

# == FLR ==============================
def FL_FLR(train_dfs, train_targets, train_data, train_tloc, \
            test_dfs, test_targets, test_data, test_tloc, \
            test_project_name, i=0):
    cols = 0
    num_clients = 1
    element_spec = 0

    def preprocess(dataset):
        def element_fn(element):
            return collections.OrderedDict([
                ('x', tf.reshape(element['features'], [-1])),
                ('y', tf.reshape(element['target'], [1])),
            ])
        return dataset.repeat(Config.NUM_EPOCHS).map(element_fn).shuffle(Config.SHUFFLE_BUFFER).batch(Config.BATCH_SIZE)

    def make_federated_data(client_data, num_clients_list):
        return [preprocess(client_data[x]) for x in num_clients_list]

    def create_keras_model():
        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(
                1, activation=tf.nn.sigmoid, kernel_initializer='zeros', input_shape=(cols,))])
        return model

    def model_fn():
        keras_model = create_keras_model()
        return tff.learning.from_keras_model(
            keras_model=keras_model,
            loss=tf.keras.losses.BinaryCrossentropy(),
            input_spec=element_spec,
            metrics=[tf.keras.metrics.BinaryAccuracy(),tf.keras.metrics.AUC(),tf.keras.metrics.Precision(),tf.keras.metrics.Recall()])

    random.seed(i)
    cols = len(train_dfs[0].columns)
    num_clients = len(train_data)
    print(f'NUM_CLIENTS : {num_clients}')

    federated_train_data = make_federated_data(train_data, range(num_clients))
    element_spec = federated_train_data[0].element_spec

    tf.keras.backend.set_floatx('float32')
    iterative_process = tff.learning.build_federated_averaging_process(
        model_fn=model_fn,
        client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=Config.CLIENT_LEARNING_RATE),
        server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=Config.SERVER_LEARNING_RATE),
        use_experimental_simulation_loop=True
        )
    state = iterative_process.initialize()

    # learn
    for round_num in range(1, Config.NUM_ROUNDS):
        # selected_clients = range(num_clients)
        # federated_train_data = make_federated_data(train_data, selected_clients)
        state, metrics = iterative_process.next(state, federated_train_data)
        print('round {:2d}, clients num: {} metrics={}'.format(round_num, num_clients, metrics))

    # evaluate
    evaluation = tff.learning.build_federated_evaluation(model_fn, use_experimental_simulation_loop=True)

    # test
    federated_test_data = make_federated_data(test_data, range(1))
    print(federated_train_data)
    print(federated_test_data)
    test_metrics = evaluation(state.model, federated_test_data)

    # EPM
    model_for_inference = create_keras_model()
    state.model.assign_weights_to(model_for_inference)
    predictions = model_for_inference.predict_on_batch(test_dfs[0])
    prob_list = np.array(predictions).flatten()
    label_list = np.array(test_targets[0])
    line_list = np.array(test_tloc).flatten()

    # result
    learning_type = 'FL'
    classifier = 'FLR'
    out_dict = collections.OrderedDict()
    out_od = collections.OrderedDict()
    out_od.update(classifier2OD(f'{test_project_name.replace(".csv", "")}', learning_type, classifier))
    out_od.update(model_no2OD(f'{i:03d}'))
    out_od.update(test_metrics['eval'])
    out_od.update(EPM2OD(prob_list, label_list, line_list))
    # out_od.update(config2OD())
    out_dict[f'{test_project_name}_{learning_type}_{classifier}_[{i:03d}]'] = out_od

    gc.collect()
    return out_dict

# == SL ==============================
def SL_LR(train_dfs, train_targets, train_data, train_tloc, \
            test_dfs, test_targets, test_data, test_tloc, \
            test_project_name, i=0):
    random.seed(i)
    out_dict = collections.OrderedDict()
    train_x = pd.concat(train_dfs)
    train_y = pd.concat(train_targets)
    train_df = pd.concat([train_x, train_y], axis=1)
    test_x = test_dfs[0]
    test_y = test_targets[0]

    lr = LogisticRegression(C=100, max_iter=200, random_state=i)
    lr.fit(train_x, train_y)

    y_pred = lr.predict(test_x)
    y_scores = lr.predict_proba(test_x)
    y_scores = np.array([j[1] for j in y_scores])
    y_true = np.array(test_y)

    accuracy, auc, precision, recall = classification_metrics(y_true, y_pred, y_scores)
    tn, fp, fn, tp = metrics.confusion_matrix(test_y, y_pred).ravel()
    PD, pf, gmeasure, fmeasure = defects_metrics(tn, fp, fn, tp)

    # EPM
    prob_list = y_scores
    label_list = np.array(test_targets[0])
    line_list = np.array(test_tloc).flatten()

    learning_type = 'SL'
    classifier = 'LR'
    out_od = collections.OrderedDict()
    out_od.update(SL_LACE22OD(f'{test_project_name.replace(".csv", "")}', learning_type, classifier, f'{i:03d}', accuracy, auc, precision, recall, PD, pf, gmeasure, fmeasure))
    out_od.update(EPM2OD(prob_list, label_list, line_list))
    out_dict[f'{test_project_name}_{learning_type}_{classifier}_[{i:03d}]'] = out_od

    return out_dict

def SL_RF(train_dfs, train_targets, train_data, train_tloc, \
            test_dfs, test_targets, test_data, test_tloc, \
            test_project_name, i=0):
    random.seed(i)
    out_dict = collections.OrderedDict()
    train_x = pd.concat(train_dfs)
    train_y = pd.concat(train_targets)
    test_x = test_dfs[0]
    test_y = test_targets[0]

    rf = RandomForestClassifier(n_estimators=300, max_depth=20, max_features='log2', random_state=i)
    rf.fit(train_x, train_y)

    y_pred = rf.predict(test_x)
    y_scores = rf.predict_proba(test_x)
    y_scores = np.array([j[1] for j in y_scores])
    y_true = np.array(test_y)

    accuracy, auc, precision, recall = classification_metrics(y_true, y_pred, y_scores)
    tn, fp, fn, tp = metrics.confusion_matrix(test_y, y_pred).ravel()
    PD, pf, gmeasure, fmeasure = defects_metrics(tn, fp, fn, tp)

    # EPM
    prob_list = y_scores
    label_list = np.array(test_targets[0])
    line_list = np.array(test_tloc).flatten()

    learning_type = 'SL'
    classifier = 'RF'
    out_od = collections.OrderedDict()
    out_od.update(SL_LACE22OD(f'{test_project_name.replace(".csv", "")}', learning_type, classifier, f'{i:03d}', accuracy, auc, precision, recall, PD, pf, gmeasure, fmeasure))
    out_od.update(EPM2OD(prob_list, label_list, line_list))
    out_dict[f'{test_project_name}_{learning_type}_{classifier}_[{i:03d}]'] = out_od

    return out_dict

def SL_KNN(train_dfs, train_targets, train_data, train_tloc, \
            test_dfs, test_targets, test_data, test_tloc, \
            test_project_name, i=0):
    random.seed(i)
    out_dict = collections.OrderedDict()
    train_x = pd.concat(train_dfs)
    train_y = pd.concat(train_targets)
    test_x = test_dfs[0]
    test_y = test_targets[0]

    knc = KNeighborsClassifier(n_neighbors=5, weights='uniform')
    knc.fit(train_x, train_y)

    y_pred = knc.predict(test_x)
    y_scores = knc.predict_proba(test_x)
    y_scores = np.array([i[1] for i in y_scores ])
    y_true = np.array(test_y)

    accuracy, auc, precision, recall = classification_metrics(y_true, y_pred, y_scores)
    tn, fp, fn, tp = metrics.confusion_matrix(test_y, y_pred).ravel()
    PD, pf, gmeasure, fmeasure = defects_metrics(tn, fp, fn, tp)
    # EPM
    prob_list = y_scores
    label_list = np.array(test_targets[0])
    line_list = np.array(test_tloc).flatten()

    learning_type = 'SL'
    classifier = 'KNN'
    out_od = collections.OrderedDict()
    out_od.update(SL_LACE22OD(f'{test_project_name.replace(".csv", "")}', learning_type, classifier, f'{i:03d}', accuracy, auc, precision, recall, PD, pf, gmeasure, fmeasure))
    out_od.update(defects_metrics2OD(PD, pf, gmeasure, fmeasure))
    out_od.update(EPM2OD(prob_list, label_list, line_list))
    out_dict[f'{test_project_name}_{learning_type}_{classifier}_[{i:03d}]'] = out_od

    return out_dict

# == LACE2 ==============================
def LACE2(train_dfs, train_targets):
    train_x = pd.concat(train_dfs)
    train_y = pd.concat(train_targets)
    train_df = pd.concat([train_x, train_y], axis=1)
    return lace.lace2_simulator(
                                attribute_names=list(train_df.columns),
                                data_matrix=train_df.values.tolist(),
                                independent_attrs=list(train_x.columns),
                                objective_attr=train_y.name,
                                objective_as_binary=False,
                                cliff_percentage=0.8,
                                morph_alpha=0.15,
                                morph_beta=0.35,
                                number_of_holder=len(train_dfs)
                                )

def LACE2_LR(train_dfs, train_targets, train_data, train_tloc, \
        test_dfs, test_targets, test_data, test_tloc, \
            test_project_name, i=0):
    random.seed(i)
    out_dict = collections.OrderedDict()
    lace2res = LACE2(train_dfs, train_targets)
    train_col = lace2res.pop(0)
    train_df = pd.DataFrame(lace2res, columns=train_col).astype('float64')
    train_y = train_df.pop(pd.concat(train_targets).name)
    train_x = train_df
    test_x = test_dfs[0]
    test_y = test_targets[0]

    lr = LogisticRegression(C=100, max_iter=200, random_state=i)
    lr.fit(train_x, train_y)

    y_pred = lr.predict(test_x)
    y_scores = lr.predict_proba(test_x)
    y_scores = np.array([j[1] for j in y_scores ])
    y_true = np.array(test_y)

    accuracy, auc, precision, recall = classification_metrics(y_true, y_pred, y_scores)
    tn, fp, fn, tp = metrics.confusion_matrix(test_y, y_pred).ravel()
    PD, pf, gmeasure, fmeasure = defects_metrics(tn, fp, fn, tp)
    # EPM
    prob_list = y_scores
    label_list = np.array(test_targets[0])
    line_list = np.array(test_tloc).flatten()

    learning_type = 'LACE2'
    classifier = 'LR'
    out_od = collections.OrderedDict()
    out_od.update(SL_LACE22OD(f'{test_project_name.replace(".csv", "")}', learning_type, classifier, f'{i:03d}', accuracy, auc, precision, recall, PD, pf, gmeasure, fmeasure))
    out_od.update(EPM2OD(prob_list, label_list, line_list))
    out_dict[f'{test_project_name}_{learning_type}_{classifier}_[{i:03d}]'] = out_od

    return out_dict

def LACE2_RF(train_dfs, train_targets, train_data, train_tloc, \
        test_dfs, test_targets, test_data, test_tloc, \
            test_project_name, i=0):
    random.seed(i)
    out_dict = collections.OrderedDict()
    lace2res = LACE2(train_dfs, train_targets)
    train_col = lace2res.pop(0)
    train_df = pd.DataFrame(lace2res, columns=train_col).astype('float64')
    train_y = train_df.pop(pd.concat(train_targets).name)
    train_x = train_df
    test_x = test_dfs[0]
    test_y = test_targets[0]

    rf = RandomForestClassifier(n_estimators=300, max_depth=20, max_features='log2', random_state=i)
    rf.fit(train_x, train_y)

    y_pred = rf.predict(test_x)
    y_scores = rf.predict_proba(test_x)
    y_scores = np.array([j[1] for j in y_scores])
    y_true = np.array(test_y)

    accuracy, auc, precision, recall = classification_metrics(y_true, y_pred, y_scores)
    tn, fp, fn, tp = metrics.confusion_matrix(test_y, y_pred).ravel()
    PD, pf, gmeasure, fmeasure = defects_metrics(tn, fp, fn, tp)

    # EPM
    prob_list = y_scores
    label_list = np.array(test_targets[0])
    line_list = np.array(test_tloc).flatten()

    learning_type = 'LACE2'
    classifier = 'RF'
    out_od = collections.OrderedDict()
    out_od.update(SL_LACE22OD(f'{test_project_name.replace(".csv", "")}', learning_type, classifier, f'{i:03d}', accuracy, auc, precision, recall, PD, pf, gmeasure, fmeasure))
    out_od.update(EPM2OD(prob_list, label_list, line_list))
    out_dict[f'{test_project_name}_{learning_type}_{classifier}_[{i:03d}]'] = out_od

    return out_dict

def LACE2_KNN(train_dfs, train_targets, train_data, train_tloc, \
        test_dfs, test_targets, test_data, test_tloc, \
            test_project_name, i=0):
    random.seed(i)
    out_dict = collections.OrderedDict()
    lace2res = LACE2(train_dfs, train_targets)
    train_col = lace2res.pop(0)
    train_df = pd.DataFrame(lace2res, columns=train_col).astype('float64')
    train_y = train_df.pop(pd.concat(train_targets).name)
    train_x = train_df
    test_x = test_dfs[0]
    test_y = test_targets[0]

    knc = KNeighborsClassifier(n_neighbors=5, weights='uniform')
    knc.fit(train_x, train_y)

    y_pred = knc.predict(test_x)
    y_scores = knc.predict_proba(test_x)
    y_scores = np.array([j[1] for j in y_scores ])
    y_true = np.array(test_y)

    accuracy, auc, precision, recall = classification_metrics(y_true, y_pred, y_scores)
    tn, fp, fn, tp = metrics.confusion_matrix(test_y, y_pred).ravel()
    PD, pf, gmeasure, fmeasure = defects_metrics(tn, fp, fn, tp)

    # EPM
    prob_list = y_scores
    label_list = np.array(test_targets[0])
    line_list = np.array(test_tloc).flatten()

    learning_type = 'LACE2'
    classifier = 'KNN'
    out_od = collections.OrderedDict()
    out_od.update(SL_LACE22OD(f'{test_project_name.replace(".csv", "")}', learning_type, classifier, f'{i:03d}', accuracy, auc, precision, recall, PD, pf, gmeasure, fmeasure))
    out_od.update(EPM2OD(prob_list, label_list, line_list))
    out_dict[f'{test_project_name}_{learning_type}_{classifier}_[{i:03d}]'] = out_od

    return out_dict

# == UL ==============================
def UL_Manualdown(test_dfs, test_targets, test_data, test_tloc, test_project_name):
    out_dict = collections.OrderedDict()
    _, tloc_max, tloc_min = outlier_iqr(test_tloc.copy())
    manual_down = [(x - tloc_min)/(tloc_max-tloc_min) for x in test_tloc]
    test_y = test_targets[0]

    y_true = np.array(test_targets[0])
    y_scores = np.array(manual_down)
    y_pred = np.where(y_scores > 0.5, 1, 0)
    accuracy, auc, precision, recall = classification_metrics(y_true, y_pred, y_scores)
    tn, fp, fn, tp = metrics.confusion_matrix(test_y, y_pred).ravel()
    PD, pf, gmeasure, fmeasure = defects_metrics(tn, fp, fn, tp)

    # EPM
    prob_list = y_scores
    label_list = np.array(test_targets[0])
    line_list = np.array(test_tloc).flatten()

    learning_type = 'UL'
    classifier = 'Manualdown'
    out_od = collections.OrderedDict()
    out_od.update(classifier2OD(f'{test_project_name.replace(".csv", "")}', learning_type, classifier))
    out_od.update(metrics2OD(accuracy, auc, precision, recall))
    out_od.update(defects_metrics2OD(PD, pf, gmeasure, fmeasure))
    out_od.update(EPM2OD(prob_list, label_list, line_list))
    out_dict[f'{test_project_name}_{learning_type}_{classifier}'] = out_od

    return out_dict

# == Func ==============================
def classification_metrics(y_true, y_pred, y_scores):
    return metrics.accuracy_score(y_true, y_pred), \
            metrics.roc_auc_score(y_true, y_scores), \
            metrics.precision_score(y_true, y_pred), \
            metrics.recall_score(y_true, y_pred)

def defects_metrics(tn, fp, fn, tp):
    PD = tp / (tp+fn)
    pf = fp / (fp+tn)
    gmeasure = (2*PD*(100-pf)) / (PD+(100-pf))
    fmeasure = tp / (tp+0.5*(fp+fn))
    return PD, pf, gmeasure, fmeasure

def kwargs2OD(**kwargs):
    kwargs_od = collections.OrderedDict()
    for items in locals().values():
        for key, value in items.items():
            kwargs_od[key] = value
    return kwargs_od

def classifier2OD(test_project_name, learning_type, classifier):
    return kwargs2OD(test_project=test_project_name, learning_type=learning_type, classifier=classifier)

def metrics2OD(accuracy, auc, precision, recall):
    return kwargs2OD(accuracy=accuracy, auc=auc, precision=precision, recall=recall)

def defects_metrics2OD(pd, pf, gmeasure, fmeasure):
    return kwargs2OD(probability_of_detection=pd, probability_of_false_alarm=pf, gmeasure=gmeasure, fmeasure=fmeasure)

def model_no2OD(model_no):
    return kwargs2OD(model_no=model_no)

def SL_LACE22OD(test_project, learning_type, classifier, model_no,
                accuracy, auc, precision, recall,
                probability_of_detection, probability_of_false_alarm, gmeasure, fmeasure):
    SL_LACE_od = collections.OrderedDict()
    SL_LACE_od.update(classifier2OD(test_project, learning_type, classifier))
    SL_LACE_od.update(model_no2OD(model_no))
    SL_LACE_od.update(metrics2OD(accuracy, auc, precision, recall))
    SL_LACE_od.update(defects_metrics2OD(probability_of_detection, probability_of_false_alarm, gmeasure, fmeasure))
    return SL_LACE_od

def outlier_iqr(df_series, threshold=1.5):
    q1 = df_series.describe().loc['25%']
    q3 = df_series.describe().loc['75%']
    iqr = q3 - q1
    outlier_min = q1 - (iqr) * threshold
    outlier_max = q3 + (iqr) * threshold
    df_series[df_series < outlier_min] = None
    df_series[df_series > outlier_max] = None
    return df_series, np.nanmax(df_series), np.nanmin(df_series)

def EPM2OD(prob_list, label_list, line_list):
    ins = EffortAwareEvaluation(prob_list, label_list, line_list)
    return collections.OrderedDict([
        ('IFA', ins.IFA()),
        ('PII@20%', ins.PII(20, prob=True)),
        ('PII@1000', ins.PII(1000)),
        ('PII@2000', ins.PII(2000)),
        ('CostEffort@20%', ins.CostEffort(20, prob=True)),
        ('CostEffort@1000', ins.CostEffort(1000)),
        ('CostEffort@2000', ins.CostEffort(2000)),
        ('Popt@20%', ins.norm_popt(L=20)),
        ('Popt@100%', ins.norm_popt())
    ])
