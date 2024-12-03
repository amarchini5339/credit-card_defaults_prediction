import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Avoid warnings caused by inplace column name reassignment
pd.options.mode.chained_assignment = None

numerical_features = [
    "CREDIT_LIMIT",
    "AGE",
    "SEPT_BILL",
    "AUG_BILL",
    "JULY_BILL",
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


def get_features_and_targets():
    # fetch dataset
    default_of_credit_card_clients = pd.read_csv(
        "../default_of_credit_card_clients.csv"
    )
    default_of_credit_card_clients.drop(
        default_of_credit_card_clients.columns[0], axis=1, inplace=True
    )
    default_of_credit_card_clients.drop(index=0, inplace=True)
    default_of_credit_card_clients = default_of_credit_card_clients.apply(
        pd.to_numeric, errors="coerce"
    )

    # data (as pandas dataframes)
    features = default_of_credit_card_clients[
        ["X{}".format(col_num) for col_num in range(1, 24)]
    ]
    targets = default_of_credit_card_clients[["Y"]]

    features.rename(
        inplace=True,
        columns={
            "X1": "CREDIT_LIMIT",  # Credit limit (NT dollar)
            "X2": "GENDER",  # Gender (1 = male; 2 = female)
            "X3": "EDUCATION_LEVEL",  # Education (1 = graduate school; 2 = university; 3 = high school; 4 = others)
            "X4": "MARITAL_STATUS",  # Marital status (1 = married; 2 = single; 3 = others)
            "X5": "AGE",  # (years)
            # X6 - X11 is repayment status
            # The measurement scale for the repayment status is:
            # -1 = pay duly;
            # 1 = payment delay for one month;
            # 2 = payment delay for two months;
            # . . .;
            # 8 = payment delay for eight months;
            # 9 = payment delay for nine months and above.
            "X6": "SEPT_PAY_STATUS",  # repayment status in September, 2005
            "X7": "AUG_PAY_STATUS",  # repayment status in August, 2005
            "X8": "JULY_PAY_STATUS",  # repayment status in July, 2005
            "X9": "JUNE_PAY_STATUS",  # repayment status in June, 2005
            "X10": "MAY_PAY_STATUS",  # repayment status in May, 2005
            "X11": "APRIL_PAY_STATUS",  # repayment status in April, 2005
            # X12 - X17 is amount of bill statement (NT dollar)
            "X12": "SEPT_BILL",  # amount of bill statement in September, 2005
            "X13": "AUG_BILL",  # amount of bill statement in August, 2005
            "X14": "JULY_BILL",  # amount of bill statement in July, 2005
            "X15": "JUNE_BILL",  # amount of bill statement in June, 2005
            "X16": "MAY_BILL",  # amount of bill statement in May, 2005
            "X17": "APRIL_BILL",  # amount of bill statement in April, 2005
            # X18 - X23 is amount of previous payment (NT dollar)
            "X18": "SEPT_PAYMENT",  # amount paid in September, 2005
            "X19": "AUG_PAYMENT",  # amount paid in August, 2005
            "X20": "JULY_PAYMENT",  # amount paid in July, 2005
            "X21": "JUNE_PAYMENT",  # amount paid in June, 2005
            "X22": "MAY_PAYMENT",  # amount paid in May, 2005
            "X23": "APRIL_PAYMENT",  # amount paid in April, 2005
        },
    )

    targets.rename(inplace=True, columns={"Y": "DEFAULT"})  # Default payment next month

    # Encode categorical variables
    features["GENDER"] = features["GENDER"].map(
        {1: 0, 2: 1}
    )  # 1: Male -> 0, 2: Female -> 1
    print("GENDER column transformed:", features["GENDER"].unique())

    features["EDUCATION_LEVEL"] = features["EDUCATION_LEVEL"].replace(
        {0: 4, 5: 4, 6: 4}
    )  # 4: Others
    print("EDUCATION_LEVEL column transformed:", features["EDUCATION_LEVEL"].unique())

    features["MARITAL_STATUS"] = features["MARITAL_STATUS"].replace({0: 3})  # 3: Others
    print("MARITAL_STATUS column transformed:", features["MARITAL_STATUS"].unique())

    return cap_outliers(features, numerical_features), targets


def cap_outliers(data, columns):
    for column in columns:
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)][
            column
        ]
        print(f"{column}: {len(outliers)} outliers capped.")
        data[column] = data[column].clip(lower=lower_bound, upper=upper_bound)
    return data


def get_normalized_features_and_targets():
    features, targets = get_features_and_targets()

    # Normalize numerical columns using MinMaxScaler
    scaler = MinMaxScaler()
    features[numerical_features] = scaler.fit_transform(features[numerical_features])

    return features, targets
