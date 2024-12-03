import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import KMeansSMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, classification_report
from util.get_data import get_features_and_targets
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import VotingClassifier

categorical_features = [
    "GENDER",
    "EDUCATION_LEVEL",
    "MARITAL_STATUS",
    "SEPT_PAY_STATUS",
    "AUG_PAY_STATUS",
    "JULY_PAY_STATUS",
    "JUNE_PAY_STATUS",
    "MAY_PAY_STATUS",
    "APRIL_PAY_STATUS",
]
numeric_features = [
    "AGE",
    "CREDIT_LIMIT",
    "JUNE_BILL",
    "MAY_BILL",
    "APRIL_BILL",
    "SEPT_PAYMENT",
    "AUG_PAYMENT",
    "JULY_PAYMENT",
    "JUNE_PAYMENT",
    "MAY_PAYMENT",
    "APRIL_PAYMENT",
]


def dense_transform(x):
    if isinstance(x, np.ndarray):
        return x
    else:
        return x.toarray()


def make_mnb_pipeline():
    # Pipeline components for MultinomialNB
    categorical_transformer_mnb = Pipeline(
        [
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            )  # Dense output for compatibility with SMOTE
        ]
    )

    preprocessor_mnb = ColumnTransformer(
        [("cat", categorical_transformer_mnb, categorical_features)], remainder="drop"
    )  # Drop numeric features for MultinomialNB

    pipeline_mnb = ImbPipeline(
        [
            ("preprocessor", preprocessor_mnb),
            ("smote", KMeansSMOTE(random_state=0)),
            ("classifier", MultinomialNB()),
        ]
    )

    return pipeline_mnb


def make_gnb_pipeline():
    # Pipeline components for GaussianNB
    numeric_transformer_gnb = Pipeline([("scaler", StandardScaler())])

    preprocessor_gnb = ColumnTransformer(
        [("num", numeric_transformer_gnb, numeric_features)], remainder="drop"
    )  # Drop categorical features for GaussianNB

    pipeline_gnb = ImbPipeline(
        [
            ("preprocessor", preprocessor_gnb),
            (
                "to_dense",
                FunctionTransformer(dense_transform, accept_sparse=True),
            ),  # Ensures data is dense
            (
                "smote",
                KMeansSMOTE(
                    random_state=0, k_neighbors=5, cluster_balance_threshold=0.1
                ),
            ),
            ("classifier", GaussianNB()),
        ]
    )

    return pipeline_gnb


def naive_bayes():
    pipeline_gnb = make_gnb_pipeline()
    pipeline_mnb = make_mnb_pipeline()
    ensemble = VotingClassifier(
        estimators=[("gnb", pipeline_gnb), ("mnb", pipeline_mnb)],
        voting="soft",
        weights=[1, 1],
    )
    return ensemble


def run_naive_bayes(features, targets):
    X, y = features, targets

    # Define the transformation for numeric features
    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])

    # Define the transformation for categorical features
    categorical_transformer = Pipeline(
        steps=[
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            )  # Ensure output is dense
        ]
    )

    # Combine transformations into a preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        sparse_threshold=0,
    )  # Ensures combined output is dense

    # Define an explicit transformation to convert array to dense
    def dense_transform(x):
        if isinstance(x, np.ndarray):
            return x
        else:
            return x.toarray()

    # Build the pipeline
    pipeline_gnb = ImbPipeline(
        [
            ("preprocessor", preprocessor),
            (
                "to_dense",
                FunctionTransformer(dense_transform, accept_sparse=True),
            ),  # Ensures data is dense
            (
                "smote",
                KMeansSMOTE(
                    random_state=0, k_neighbors=5, cluster_balance_threshold=0.1
                ),
            ),
            ("classifier", GaussianNB()),
        ]
    )

    # Now perform your train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Fit and score the pipeline
    pipeline_gnb.fit(X_train, y_train)
    score = pipeline_gnb.score(X_test, y_test)
    print(f"Accuracy of GaussianNB model: {score}")

    # Pipeline components for GaussianNB
    numeric_transformer_gnb = Pipeline([("scaler", StandardScaler())])

    preprocessor_gnb = ColumnTransformer(
        [("num", numeric_transformer_gnb, numeric_features)], remainder="drop"
    )  # Drop categorical features for GaussianNB

    pipeline_gnb = ImbPipeline(
        [
            ("preprocessor", preprocessor),
            (
                "to_dense",
                FunctionTransformer(dense_transform, accept_sparse=True),
            ),  # Ensures data is dense
            (
                "smote",
                KMeansSMOTE(
                    random_state=0, k_neighbors=5, cluster_balance_threshold=0.1
                ),
            ),
            ("classifier", GaussianNB()),
        ]
    )

    # Pipeline components for MultinomialNB
    categorical_transformer_mnb = Pipeline(
        [
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            )  # Dense output for compatibility with SMOTE
        ]
    )

    preprocessor_mnb = ColumnTransformer(
        [("cat", categorical_transformer_mnb, categorical_features)], remainder="drop"
    )  # Drop numeric features for MultinomialNB

    pipeline_mnb = ImbPipeline(
        [
            ("preprocessor", preprocessor_mnb),
            ("smote", KMeansSMOTE(random_state=0)),
            ("classifier", MultinomialNB()),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train both pipelines
    pipeline_gnb.fit(X_train, y_train)
    pipeline_mnb.fit(X_train, y_train)

    # Get probability predictions
    probs_gnb = pipeline_gnb.predict_proba(X_test)
    probs_mnb = pipeline_mnb.predict_proba(X_test)

    # Average the probabilities from both models
    avg_probs = (probs_gnb + probs_mnb) / 2

    # Make final predictions based on the average probabilities
    final_predictions = np.argmax(avg_probs, axis=1)

    # Evaluate the combined model
    accuracy = accuracy_score(y_test, final_predictions)
    print(f"Combined Accuracy of GaussianNB and MultinomialNB: {accuracy}")

    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=0)

    # Initialize lists to store indices for each fold
    train_indices = []
    validation_indices = []

    # Split data into folds
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        train_indices.append(train_idx)
        validation_indices.append(val_idx)
        print(f"Fold {fold + 1}:")
        print(f"  Training indices: {train_idx[:5]}... ({len(train_idx)} samples)")
        print(f"  Validation indices: {val_idx[:5]}... ({len(val_idx)} samples)")

    # Assuming X and y are your complete dataset
    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Store scores for averaging
    train_scores = []
    validation_scores = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"\nFold {fold + 1}/{n_splits}")

        # Split dataset into training and validation sets
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # Ensure y_train and y_val are 1-dimensional if y is a DataFrame with a single target column
        # These will convert DataFrame columns to 1D numpy arrays
        y_train = y_train.squeeze().values
        y_val = y_val.squeeze().values

        # Fit both pipelines on the training data
        pipeline_gnb.fit(X_train, y_train)
        pipeline_mnb.fit(X_train, y_train)

        # Predict on the training data
        y_train_pred_gnb = pipeline_gnb.predict(X_train)
        y_train_pred_mnb = pipeline_mnb.predict(X_train)

        # Average the training predictions
        avg_train_predictions = (y_train_pred_gnb + y_train_pred_mnb) / 2
        avg_train_predictions = np.round(avg_train_predictions).astype(
            int
        )  # Convert averages to discrete class predictions

        # Compute training accuracy
        train_accuracy = accuracy_score(y_train, avg_train_predictions)
        train_scores.append(train_accuracy)

        # Predict probabilities on the validation set and combine predictions
        probs_gnb = pipeline_gnb.predict_proba(X_val)
        probs_mnb = pipeline_mnb.predict_proba(X_val)
        avg_probs = (probs_gnb + probs_mnb) / 2
        final_predictions = np.argmax(avg_probs, axis=1)

        # Compute and append validation accuracy
        validation_accuracy = accuracy_score(y_val, final_predictions)
        validation_scores.append(validation_accuracy)

        print(f"Training Accuracy for current fold: {train_accuracy:.2f}")
        print(f"Validation Accuracy for current fold: {validation_accuracy:.2f}")

    # Calculate average scores across all folds
    average_train_score = np.mean(train_scores)
    average_val_score = np.mean(validation_scores)

    print("\nK-Fold Cross-Validation Results for Ensemble Naive Bayes Model:")
    print(f"Average Training Accuracy: {average_train_score:.2f}")
    print(f"Average Validation Accuracy: {average_val_score:.2f}")

    # Assuming X and y are the complete datasets
    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Store scores for averaging
    gnb_train_scores = []
    gnb_validation_scores = []
    mnb_train_scores = []
    mnb_validation_scores = []

    ensemble_train_scores = []
    ensemble_validation_scores = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"\nFold {fold + 1}/{n_splits}")

        # Split dataset into training and validation sets
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = (
            y.iloc[train_idx].squeeze().values,
            y.iloc[val_idx].squeeze().values,
        )

        # Fit both pipelines on the training data
        pipeline_gnb.fit(X_train, y_train)
        pipeline_mnb.fit(X_train, y_train)

        # Predict on the training data
        y_train_pred_gnb = pipeline_gnb.predict(X_train)
        y_train_pred_mnb = pipeline_mnb.predict(X_train)

        # Calculate individual training accuracies
        gnb_train_accuracy = accuracy_score(y_train, y_train_pred_gnb)
        mnb_train_accuracy = accuracy_score(y_train, y_train_pred_mnb)
        gnb_train_scores.append(gnb_train_accuracy)
        mnb_train_scores.append(mnb_train_accuracy)

        # Average the training predictions for ensemble
        avg_train_predictions = (y_train_pred_gnb + y_train_pred_mnb) / 2
        avg_train_predictions = np.round(avg_train_predictions).astype(int)
        ensemble_train_accuracy = accuracy_score(y_train, avg_train_predictions)
        ensemble_train_scores.append(ensemble_train_accuracy)

        # Predict probabilities on the validation set
        probs_gnb = pipeline_gnb.predict_proba(X_val)
        probs_mnb = pipeline_mnb.predict_proba(X_val)

        # Calculate individual validation accuracies
        y_val_pred_gnb = np.argmax(probs_gnb, axis=1)
        y_val_pred_mnb = np.argmax(probs_mnb, axis=1)
        gnb_val_accuracy = accuracy_score(y_val, y_val_pred_gnb)
        mnb_val_accuracy = accuracy_score(y_val, y_val_pred_mnb)
        gnb_validation_scores.append(gnb_val_accuracy)
        mnb_validation_scores.append(mnb_val_accuracy)

        # Average the validation probabilities for ensemble
        avg_probs = (probs_gnb + probs_mnb) / 2
        final_predictions = np.argmax(avg_probs, axis=1)
        ensemble_validation_accuracy = accuracy_score(y_val, final_predictions)
        ensemble_validation_scores.append(ensemble_validation_accuracy)

        # Output fold results
        print(
            f"GNB Training Accuracy: {gnb_train_accuracy:.2f}, Validation Accuracy: {gnb_val_accuracy:.2f}"
        )
        print(
            f"MNB Training Accuracy: {mnb_train_accuracy:.2f}, Validation Accuracy: {mnb_val_accuracy:.2f}"
        )
        print(
            f"Ensemble Training Accuracy: {ensemble_train_accuracy:.2f}, Validation Accuracy: {ensemble_validation_accuracy:.2f}"
        )

    # Calculate average scores across all folds
    average_gnb_train = np.mean(gnb_train_scores)
    average_gnb_val = np.mean(gnb_validation_scores)
    average_mnb_train = np.mean(mnb_train_scores)
    average_mnb_val = np.mean(mnb_validation_scores)

    average_ensemble_train = np.mean(ensemble_train_scores)
    average_ensemble_val = np.mean(ensemble_validation_scores)

    print("\nFinal K-Fold Cross-Validation Results:")
    print(
        f"GNB Average Training Accuracy: {average_gnb_train:.2f}, Average Validation Accuracy: {average_gnb_val:.2f}"
    )
    print(
        f"MNB Average Training Accuracy: {average_mnb_train:.2f}, Average Validation Accuracy: {average_mnb_val:.2f}"
    )
    print(
        f"Ensemble Average Training Accuracy: {average_ensemble_train:.2f}, Average Validation Accuracy: {average_ensemble_val:.2f}"
    )

    # Use a VotingClassifier to combine the two models
    from sklearn.ensemble import VotingClassifier

    # Define the ensemble model
    ensemble = VotingClassifier(
        estimators=[("gnb", pipeline_gnb), ("mnb", pipeline_mnb)],
        voting="soft",
        weights=[1, 1],
    )

    for fold in range(1, n_splits + 1):
        print(f"\nFold {fold}/{n_splits}")

        # Split dataset into training and validation sets
        X_train, X_val = (
            X.iloc[train_indices[fold - 1]],
            X.iloc[validation_indices[fold - 1]],
        )
        y_train, y_val = (
            y.iloc[train_indices[fold - 1]].squeeze().values,
            y.iloc[validation_indices[fold - 1]].squeeze().values,
        )

        # Fit the ensemble model on the training data
        ensemble.fit(X_train, y_train)

        # Evaluate the ensemble model on the validation data
        val_score = ensemble.score(X_val, y_val)
        print(f"Validation Accuracy: {val_score:.2f}")

        # Perform k-fold cross-validation on the ensemble model
        cv_scores = cross_val_score(ensemble, X, y.squeeze().values, cv=kf)
        avg_cv_score = np.mean(cv_scores)
        print(f"Average Cross-Validation Accuracy: {avg_cv_score:.2f}")
        print(f"Cross-Validation Scores: {cv_scores}")
        # classification report
        y_pred = ensemble.predict(X_val)
        print("Classification Report:")
        print(classification_report(y_val, y_pred))

    # Train a new ensemble model without k-fold cross-validation for comparison
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Fit the ensemble model on the training data
    ensemble.fit(X_train, y_train)

    # Evaluate the ensemble model on the test data
    test_score = ensemble.score(X_test, y_test)
    print(f"Test Accuracy without K-Fold: {test_score:.2f}")

    # Generate classification report
    y_pred = ensemble.predict(X_test)
    print("Classification Report without K-Fold:")
    print(classification_report(y_test, y_pred))
