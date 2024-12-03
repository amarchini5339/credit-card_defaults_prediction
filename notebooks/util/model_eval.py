from sklearn.metrics import classification_report, roc_auc_score

def predict_and_score(model, features_test, targets_test):
    pred = model.predict(features_test)

    print("\nClassification Report:")
    print(classification_report(targets_test, pred))

    print('ROC-AUC Score: {}'.format(roc_auc_score(targets_test, model.predict_proba(features_test)[:, 1])))