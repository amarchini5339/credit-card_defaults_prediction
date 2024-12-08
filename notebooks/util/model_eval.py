from sklearn.metrics import classification_report, roc_auc_score, precision_recall_fscore_support

def predict_and_score(model, features_test, targets_test):
    pred = model.predict(features_test)

    report = classification_report(targets_test, pred)
    print("\nClassification Report:")
    print(report)

    roc_auc_score_val = roc_auc_score(targets_test, model.predict_proba(features_test)[:, 1])
    print('ROC-AUC Score: {}'.format(roc_auc_score_val))

    return precision_recall_fscore_support(targets_test, pred), roc_auc_score_val
