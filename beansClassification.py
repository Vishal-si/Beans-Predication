import pandas as pd
import numpy as np
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.base import clone
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE


# These size columns are strongly right-skewed, so they are log transformed before scaling.
skew_cols = [
    "Area",
    "Perimeter",
    "MajorAxisLength",
    "MinorAxisLength",
    "ConvexArea",
    "EquivDiameter"
]


def importData(currentPath:str) -> pd.DataFrame | None:
    dataFrame = None
    try:
        currentPath = currentPath + "/" + "Beans.csv"
        dataFrame = pd.read_csv(currentPath)
    except Exception as e:
        print(f"Exception inside when importData : {e}")
    return dataFrame


def exploreEDA(beansRecords: pd.DataFrame):
    try:
        print('######################## INFO ########################')
        print(beansRecords.info())

        print('######################## HEAD ########################')
        print(beansRecords.head())

        print('######################## TAIL ########################')
        print(beansRecords.tail())

        print('######################## SAMPLE ########################')
        print(beansRecords.sample(5))

        print('######################## SHAPE ########################')
        print("Rows:", beansRecords.shape[0])
        print("Columns:", beansRecords.shape[1])

        print('######################## FIND THE NAN VALUE ########################')
        print(beansRecords.isnull().sum())

        print("######################## MANAGE THE DUPLICATE VALUES ########################")
        print(f"Duplicate values = ", beansRecords.duplicated().sum())
        beansRecords.drop_duplicates(inplace=True)
        print(f"Duplicate values = ", beansRecords.duplicated().sum())

        print("######################## DESCRIPTIVE STATISTICS ANALYSIS ########################")
        pd.options.display.float_format = '{:.2f}'.format
        print(beansRecords.describe(include=['int64', 'float64']).T)

        print("######################## CLASS DISTRIBUTION ########################")
        print(beansRecords["Class"].value_counts())
        print("Class percentage:")
        print((beansRecords["Class"].value_counts(normalize=True) * 100).round(2))

        print("######################## AVERAGE AREA BY CLASS ########################")
        area_by_class = beansRecords.groupby("Class")["Area"].mean().sort_values(ascending=False)
        print(area_by_class)
        print(f"Highest average area: {area_by_class.index[0]}")
        print(f"Lowest average area: {area_by_class.index[-1]}")

        print("######################## CORR ########################")
        numerical_cols = beansRecords.drop(columns=["Class"]).columns.tolist()
        corr = beansRecords[numerical_cols].corr()
        print(corr)

        pairs = corr.abs().stack().reset_index()
        pairs.columns = ["Feature 1", "Feature 2", "Correlation"]
        pairs = pairs[pairs["Feature 1"] < pairs["Feature 2"]]
        pairs = pairs.sort_values("Correlation", ascending=False)
        print("######################## TOP CORRELATED FEATURES ########################")
        print(pairs.head(8))

        print("######################## KEY FINDINGS ########################")
        class_counts = beansRecords["Class"].value_counts()
        print(f"Largest class: {class_counts.index[0]} ({class_counts.iloc[0]} rows)")
        print(f"Smallest class: {class_counts.index[-1]} ({class_counts.iloc[-1]} rows)")
        print("No missing values are present.")
        print("DERMASON has the most rows and BOMBAY has the fewest, so the classes are imbalanced.")
        print("Area, Perimeter, MajorAxisLength, ConvexArea and EquivDiameter move together because they all describe bean size.")
        print(f"{area_by_class.index[0]} has the highest average area, so it is the large bean class.")

        print("######################## PLOT NUMERICAL ########################")
        plot_dir = os.path.join(os.path.dirname(__file__), "plots")
        os.makedirs(plot_dir, exist_ok=True)

        fig, axes = plt.subplots(4, 4, figsize=(16, 12))
        axes = axes.flatten()
        for ax, col in zip(axes, numerical_cols):
            sns.histplot(data=beansRecords, x=col, kde=True, bins=30, ax=ax)
            ax.set_title(f"Distribution of {col}")
        plt.tight_layout()
        fig.savefig(os.path.join(plot_dir, "histograms.png"))
        plt.close(fig)

        print("######################## PLOT BOXPLOTS ########################")
        fig, axes = plt.subplots(4, 4, figsize=(16, 12))
        axes = axes.flatten()
        for ax, col in zip(axes, numerical_cols):
            sns.boxplot(data=beansRecords, y=col, ax=ax)
            ax.set_title(col)
        plt.tight_layout()
        fig.savefig(os.path.join(plot_dir, "boxplots.png"))
        plt.close(fig)

        print("######################## PLOT CLASS ########################")
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.countplot(data=beansRecords, x="Class", hue="Class", legend=False, ax=ax)
        ax.set_title("Count of Class")
        ax.tick_params(axis="x", rotation=25)
        plt.tight_layout()
        fig.savefig(os.path.join(plot_dir, "class_count.png"))
        plt.close(fig)

        print("######################## CORRELATION HEATMAP ########################")
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
        ax.set_title("Correlation Heatmap")
        plt.tight_layout()
        fig.savefig(os.path.join(plot_dir, "correlation_heatmap.png"))
        plt.close(fig)

        print("######################## PAIRPLOT ########################")
        sample_records = beansRecords.sample(500, random_state=42)
        pair_cols = ["Area", "Perimeter", "AspectRation", "roundness", "Compactness", "Class"]
        graph = sns.pairplot(sample_records[pair_cols], hue="Class")
        graph.savefig(os.path.join(plot_dir, "pairplot.png"))
        plt.close("all")
        print("Plots saved in the plots folder.")
        print("Pairplot uses 500 sample rows and 5 features so the chart stays readable.")

    except Exception as e:
        print(f"Exception inside exploreEDA Part {e}")


def outlierTreatment(records : pd.DataFrame) -> pd.DataFrame | None:
    beansRecords = None
    try:
        print("######################## MISSING VALUES ########################")
        print(records.isnull().sum())
        print("Missing values do not need to be filled.")

        print("######################## OUTLIER TREATMENT ########################")
        numerical_cols = records.drop(columns=["Class"]).columns.tolist()
        outlier_mask = pd.Series(False, index=records.index)

        for col in numerical_cols:
            Q1 = records[col].quantile(0.25)
            Q3 = records[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            col_mask = (records[col] < lower) | (records[col] > upper)
            print(f"{col} outlier rows: {int(col_mask.sum())}")
            outlier_mask = outlier_mask | col_mask

        print("Rows:", len(records))
        print("Rows outside the IQR fence:", int(outlier_mask.sum()))
        print("Class count of those rows:")
        print(records.loc[outlier_mask, "Class"].value_counts())

        # Bombay beans are naturally large, so a global IQR cut removes that whole class.
        # The rows are kept so all 7 bean types can still be classified.
        print("Outlier rows are kept.")
        print("Removing them with one global IQR rule deletes the BOMBAY class.")
        beansRecords = records.copy()

    except Exception as e:
        print(f"Exception inside the outlierTreatment {e}")

    return beansRecords


def featureEngineering(records: pd.DataFrame) -> pd.DataFrame | None:
    beansRecords = None
    try:
        beansRecords = records.copy()

        print("######################## ENCODING ########################")
        print("All feature columns are numeric.")
        print("Class is the target, so no feature encoding is needed.")

        numerical_cols = beansRecords.drop(columns=["Class"]).columns.tolist()
        print("######################## SKEWNESS ########################")
        print("Skewness before treatment:")
        print(beansRecords[numerical_cols].skew().sort_values(ascending=False))

        area_before = beansRecords["Area"].copy()
        for col in skew_cols:
            beansRecords[col] = np.log1p(beansRecords[col])

        print("Skewness after treatment:")
        print(beansRecords[numerical_cols].skew().sort_values(ascending=False))
        print("Log transformed columns:", skew_cols)

        plot_dir = os.path.join(os.path.dirname(__file__), "plots")
        os.makedirs(plot_dir, exist_ok=True)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        sns.histplot(area_before, kde=True, ax=axes[0])
        axes[0].set_title("Original Area")
        sns.histplot(beansRecords["Area"], kde=True, ax=axes[1])
        axes[1].set_title("Log Transformed Area")
        plt.tight_layout()
        fig.savefig(os.path.join(plot_dir, "area_skew.png"))
        plt.close(fig)

        print(beansRecords.head())

    except Exception as e:
        print(f"Exception Inside featureEngineering {e}")

    return beansRecords


def models(records: pd.DataFrame) -> str | None:
    best_name = None
    try:
        x = records.drop(['Class'], axis=1)
        y = records['Class']

        print("######################## TRAIN TEST SPLIT ########################")
        X_train, X_test, y_train, y_test = train_test_split(
            x, y, test_size=0.3, random_state=42, stratify=y
        )
        print("Train size:", len(y_train))
        print("Test size:", len(y_test))
        print("Train class count:")
        print(y_train.value_counts())

        print("######################## SCALING ########################")
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        print("StandardScaler is fit on the training data only.")

        model_list = [
            LogisticRegression(max_iter=2000, random_state=42),
            DecisionTreeClassifier(random_state=42),
            RandomForestClassifier(random_state=42),
            KNeighborsClassifier(),
            LinearSVC(max_iter=5000, dual=False, random_state=42),
            GradientBoostingClassifier(random_state=42),
            GaussianNB()
        ]

        results = []
        best_f1 = -1
        labels = sorted(y.unique())

        print("######################## MODEL BUILDING ########################")
        for model in model_list:
            model_test = model
            model_test.fit(X_train, y_train)
            y_pred = model_test.predict(X_test)

            train_acc = model_test.score(X_train, y_train)
            test_acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average="macro")
            cv_scores = cross_val_score(
                clone(model_test), X_train, y_train, cv=3, scoring="accuracy"
            )

            if (train_acc - test_acc) > 0.05:
                overfit = "Y"
            else:
                overfit = "N"

            print("------------------------------------------------")
            print(model_test.__class__.__name__)
            print(classification_report(y_test, y_pred, digits=4))
            print("Confusion Matrix")
            print(labels)
            print(confusion_matrix(y_test, y_pred, labels=labels))
            print(f"Cross validation accuracy: {cv_scores.mean():.4f}")

            results.append({
                "Model": model_test.__class__.__name__,
                "Train Accuracy": round(train_acc, 4),
                "Test Accuracy": round(test_acc, 4),
                "F1 Score": round(f1, 4),
                "Overfitting (Y/N)": overfit,
                "CV Accuracy": round(cv_scores.mean(), 4)
            })

            if f1 > best_f1:
                best_f1 = f1
                best_name = model_test.__class__.__name__

        results_df = pd.DataFrame(results)
        pd.options.display.float_format = '{:.4f}'.format
        print("######################## MODEL COMPARISON ########################")
        print(results_df.to_string(index=False))
        print("Best Model:", best_name)
        print("Best F1 Score:", round(best_f1, 4))

    except Exception as e:
        print(f"Exception inside the models {e}")

    return best_name


def handleImbalance(records: pd.DataFrame):
    try:
        print("######################## CLASS IMBALANCE ########################")
        x = records.drop(['Class'], axis=1)
        y = records['Class']

        X_train, X_test, y_train, y_test = train_test_split(
            x, y, test_size=0.3, random_state=42, stratify=y
        )

        print("Class count before balancing:")
        print(y_train.value_counts())

        results = []

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model_original = RandomForestClassifier(random_state=42)
        model_original.fit(X_train_scaled, y_train)
        pred = model_original.predict(X_test_scaled)
        report = classification_report(y_test, pred, output_dict=True)
        results.append({
            "Method": "Original",
            "Test Accuracy": round(accuracy_score(y_test, pred), 4),
            "F1 Score": round(f1_score(y_test, pred, average="macro"), 4),
            "BOMBAY Recall": round(report["BOMBAY"]["recall"], 4),
            "BARBUNYA Recall": round(report["BARBUNYA"]["recall"], 4)
        })
        print("Original classification report:")
        print(classification_report(y_test, pred, digits=4))

        train_data = X_train.copy()
        train_data["Class"] = y_train.to_numpy()
        max_count = int(train_data["Class"].value_counts().max())
        parts = []
        for label, group in train_data.groupby("Class"):
            if len(group) < max_count:
                extra = resample(group, replace=True, n_samples=max_count, random_state=42)
                parts.append(extra)
            else:
                parts.append(group)
        over_data = pd.concat(parts)
        X_over = over_data.drop("Class", axis=1)
        y_over = over_data["Class"]
        print("Class count after random oversampling:")
        print(y_over.value_counts())

        scaler_over = StandardScaler()
        X_over_scaled = scaler_over.fit_transform(X_over)
        X_test_over = scaler_over.transform(X_test)
        model_over = RandomForestClassifier(random_state=42)
        model_over.fit(X_over_scaled, y_over)
        pred_over = model_over.predict(X_test_over)
        report_over = classification_report(y_test, pred_over, output_dict=True)
        results.append({
            "Method": "Random Oversampling",
            "Test Accuracy": round(accuracy_score(y_test, pred_over), 4),
            "F1 Score": round(f1_score(y_test, pred_over, average="macro"), 4),
            "BOMBAY Recall": round(report_over["BOMBAY"]["recall"], 4),
            "BARBUNYA Recall": round(report_over["BARBUNYA"]["recall"], 4)
        })
        print("Random Oversampling classification report:")
        print(classification_report(y_test, pred_over, digits=4))

        smote = SMOTE(random_state=42)
        X_smote, y_smote = smote.fit_resample(X_train, y_train)
        print("Class count after SMOTE:")
        print(pd.Series(y_smote).value_counts())

        scaler_smote = StandardScaler()
        X_smote_scaled = scaler_smote.fit_transform(X_smote)
        X_test_smote = scaler_smote.transform(X_test)
        model_smote = RandomForestClassifier(random_state=42)
        model_smote.fit(X_smote_scaled, y_smote)
        pred_smote = model_smote.predict(X_test_smote)
        report_smote = classification_report(y_test, pred_smote, output_dict=True)
        results.append({
            "Method": "SMOTE",
            "Test Accuracy": round(accuracy_score(y_test, pred_smote), 4),
            "F1 Score": round(f1_score(y_test, pred_smote, average="macro"), 4),
            "BOMBAY Recall": round(report_smote["BOMBAY"]["recall"], 4),
            "BARBUNYA Recall": round(report_smote["BARBUNYA"]["recall"], 4)
        })
        print("SMOTE classification report:")
        print(classification_report(y_test, pred_smote, digits=4))

        model_weight = RandomForestClassifier(random_state=42, class_weight="balanced")
        model_weight.fit(X_train_scaled, y_train)
        pred_weight = model_weight.predict(X_test_scaled)
        report_weight = classification_report(y_test, pred_weight, output_dict=True)
        results.append({
            "Method": "Class Weight Balanced",
            "Test Accuracy": round(accuracy_score(y_test, pred_weight), 4),
            "F1 Score": round(f1_score(y_test, pred_weight, average="macro"), 4),
            "BOMBAY Recall": round(report_weight["BOMBAY"]["recall"], 4),
            "BARBUNYA Recall": round(report_weight["BARBUNYA"]["recall"], 4)
        })
        print("Class Weight classification report:")
        print(classification_report(y_test, pred_weight, digits=4))

        results_df = pd.DataFrame(results)
        pd.options.display.float_format = '{:.4f}'.format
        print("######################## IMBALANCE COMPARISON ########################")
        print(results_df.to_string(index=False))
        print("Balancing is applied to the training data only.")

    except Exception as e:
        print(f"Exception inside the handleImbalance {e}")


def model(records: pd.DataFrame, best_name: str | None):
    try:
        print("######################## HYPERPARAMETER TUNING ########################")
        x = records.drop(['Class'], axis=1)
        y = records['Class']

        X_train, X_test, y_train, y_test = train_test_split(
            x, y, test_size=0.3, random_state=42, stratify=y
        )

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        base_model = RandomForestClassifier(random_state=42)
        base_model.fit(X_train, y_train)
        base_pred = base_model.predict(X_test)
        base_f1 = f1_score(y_test, base_pred, average="macro")
        print(f"Default Random Forest F1: {base_f1:.4f}")

        param_grid = {
            "n_estimators": [100, 200],
            "max_depth": [10, None],
            "class_weight": [None, "balanced"]
        }

        grid = GridSearchCV(
            RandomForestClassifier(random_state=42),
            param_grid,
            cv=3,
            scoring="f1_macro",
            n_jobs=-1
        )
        grid.fit(X_train, y_train)

        print("Best parameters:")
        print(grid.best_params_)
        print(f"Best CV F1: {grid.best_score_:.4f}")

        best_model = grid.best_estimator_
        y_pred = best_model.predict(X_test)
        train_acc = best_model.score(X_train, y_train)
        test_acc = best_model.score(X_test, y_test)
        tuned_f1 = f1_score(y_test, y_pred, average="macro")

        print(f"Train Accuracy: {train_acc:.4f}")
        print(f"Test Accuracy: {test_acc:.4f}")
        print(f"F1 Score: {tuned_f1:.4f}")
        print(f"F1 improvement over default Random Forest: {tuned_f1 - base_f1:.4f}")
        if (train_acc - test_acc) > 0.05:
            print("Overfitting (Y/N): Y")
        else:
            print("Overfitting (Y/N): N")

        labels = sorted(y.unique())
        print(classification_report(y_test, y_pred, digits=4))
        print("Confusion Matrix")
        print(labels)
        print(confusion_matrix(y_test, y_pred, labels=labels))

        selected_model = best_model
        selected_name = "RandomForestClassifier"
        selected_f1 = tuned_f1

        model_map = {
            "LogisticRegression": LogisticRegression(max_iter=2000, random_state=42),
            "DecisionTreeClassifier": DecisionTreeClassifier(random_state=42),
            "RandomForestClassifier": RandomForestClassifier(random_state=42),
            "KNeighborsClassifier": KNeighborsClassifier(),
            "LinearSVC": LinearSVC(max_iter=5000, dual=False, random_state=42),
            "GradientBoostingClassifier": GradientBoostingClassifier(random_state=42),
            "GaussianNB": GaussianNB()
        }

        # Keep the tuned forest unless another compared model has a higher test F1.
        if best_name in model_map and best_name != "RandomForestClassifier":
            other_model = model_map[best_name]
            other_model.fit(X_train, y_train)
            other_pred = other_model.predict(X_test)
            other_f1 = f1_score(y_test, other_pred, average="macro")
            print(f"{best_name} F1: {other_f1:.4f}")
            if other_f1 > selected_f1:
                selected_model = other_model
                selected_name = best_name
                selected_f1 = other_f1

        print("Saved model:", selected_name)
        print(f"Saved model F1: {selected_f1:.4f}")

        currentPath = os.path.dirname(__file__)
        joblib.dump(scaler, f"{currentPath}//scaler.pkl")
        joblib.dump(selected_model, f"{currentPath}//model.pkl")
        print("Scaler and model saved successfully.")

    except Exception as e:
        print(f"Exception inside the model {e}")


if __name__ == "__main__":
    currentPath = os.path.dirname(__file__)
    records = importData(currentPath)
    if records is None:
        print("Beans Records Is None")
    else:
        exploreEDA(beansRecords = records)
        records = outlierTreatment(records= records)

        if records is None:
            print("Records is None After outlier")
        else:
            records = featureEngineering(records= records)

            if records is None:
                print("Records is None After feature engineering")
            else:
                best_name = models(records=records)
                handleImbalance(records=records)
                model(records, best_name)
