import streamlit as st
import pickle
import pandas as pd
import numpy as np
from matminer.datasets import load_dataset
from pymatgen.core.composition import Composition
from matminer.featurizers.composition import ElementProperty, ElementFraction
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import logging

# Initialize logging
logging.basicConfig(level=logging.INFO)

class DataLoader:
    def __init__(self, file, num_samples):
        self.num_samples = num_samples
        self.file = file

    def load_data(self):
        try:
            data = pd.read_csv(self.file)
            data = data.iloc[:self.num_samples]
            ef_ftd = pd.read_csv("df_sc_ef_ElementFraction_ftd.csv").dropna().reset_index()
            ep_ftd = pd.read_csv("df_sc_ep_ElementProperty_Magpie_ftd.csv").dropna().reset_index()
            ef_ftd = ef_ftd.drop_duplicates(subset=["Critical Temp", "_Composition"])
            ep_ftd = ep_ftd.drop_duplicates(subset=["Critical Temp", "_Composition"])
            return data, ep_ftd, ef_ftd
        except Exception as e:
            logging.error(f"Error loading data: {e}")
            st.error(f"Error loading data: {e}")
            return None, None, None

    def prepare_compositions(self):
        data, ep_ftd, ef_ftd = self.load_data()
        if data is not None:
            data['_Composition'] = data['composition'].apply(self.create_composition)
            data = data.dropna().drop_duplicates()
            return data, ep_ftd, ef_ftd
        return None, None, None

    @staticmethod
    def create_composition(formula):
        try:
            return Composition(formula)
        except ValueError:
            logging.warning(f"Error parsing formula: {formula}")
            return None

class FeatureExtractor:
    def featurize(self, data):
        data = data.dropna()
        ep_featurizer = ElementProperty.from_preset('magpie')
        ep_ftd = ep_featurizer.featurize_dataframe(data, col_id='_Composition', ignore_errors=True)
        ef_featurizer = ElementFraction()
        ef_ftd = ef_featurizer.featurize_dataframe(data, col_id='_Composition', ignore_errors=True)
        return ep_ftd, ef_ftd

class ModelManager:
    def __init__(self):
        self.models = {
            'model_ep': 'sc_ep_rf_cl.pkl',
            'model_efep': 'sc_efep_rf_cl.pkl',
            'model_ef': 'sc_ef_rf_cl.pkl',
            'regmodel_efep': 'sc_efep_best_model_yet.pkl'
        }
        self.load_models()

    def load_models(self):
        for name, path in self.models.items():
            try:
                with open(path, 'rb') as file:
                    self.models[name] = pickle.load(file)
            except FileNotFoundError:
                logging.error(f"Model file {path} not found.")
                st.error(f"Model file {path} not found.")

    def predict(self, model_name, X):
        model = self.models.get(model_name)
        if model:
            return model.predict(X)
        else:
            raise ValueError(f"Model '{model_name}' not found")

    def evaluate(self, pred_type, y_true, y_pred):
        if pred_type == "C":
            return {
                "accuracy": accuracy_score(y_true, y_pred),
                "confusion_matrix": confusion_matrix(y_true, y_pred),
                "precision": precision_score(y_true, y_pred),
                "recall": recall_score(y_true, y_pred),
                "f1_score": f1_score(y_true, y_pred),
                "roc_auc_score": roc_auc_score(y_true, y_pred)
            }
        elif pred_type == "R":
            return {
                "Mean Squared Error (MSE)": mean_squared_error(y_true, y_pred),
                "R-squared": r2_score(y_true, y_pred),
                "Mean Absolute Error (MAE)": mean_absolute_error(y_true, y_pred),
                "Root Mean Squared Error (RMSE)": np.sqrt(mean_squared_error(y_true, y_pred)),
                "Adjusted R-squared": r2_score(y_true, y_pred, multioutput='uniform_average')
            }

import numpy as np

class SuperconductorPredictor:
    def __init__(self, data_loader, feature_extractor, model_manager):
        self.data_loader = data_loader
        self.feature_extractor = feature_extractor
        self.model_manager = model_manager
        self.data, self.ep_ftd, self.ef_ftd = self.data_loader.prepare_compositions()
        if self.data is not None:
            self.final_ep_ftd, self.final_ef_ftd = self.record_indices()

    def record_indices(self):
        compositions = self.data["composition"].tolist()
        present_indices = []
        missing_indices = []
        
        for idx, composition in enumerate(compositions):
            if composition in self.ep_ftd['composition'].tolist():
                found_index = self.ep_ftd[self.ep_ftd['composition'] == composition]["index"].tolist()
                present_indices.extend(found_index)
            else:
                missing_indices.append(idx)

        present_ep_ftd = self.ep_ftd.loc[self.ep_ftd["index"].isin(present_indices)]
        present_ef_ftd = self.ef_ftd.loc[self.ef_ftd["index"].isin(present_indices)]

        if missing_indices:
            missing_data = self.data.iloc[missing_indices]
            missing_ep_ftd, missing_ef_ftd = self.feature_extractor.featurize(missing_data)
            final_ep_ftd = pd.concat([present_ep_ftd, missing_ep_ftd], ignore_index=True)
            final_ef_ftd = pd.concat([present_ef_ftd, missing_ef_ftd], ignore_index=True)
        else:
            final_ep_ftd = present_ep_ftd
            final_ef_ftd = present_ef_ftd

        return final_ep_ftd.dropna(), final_ef_ftd.dropna()

    def preprocess_data(self):
        def preprocess_for_model1(ep_ftd):
            ep_ftd["having_tc"] = (ep_ftd["Critical Temp"] >= 10).astype(int)
            ep_X = ep_ftd.iloc[:, 4:-1]
            ep_y = ep_ftd['having_tc']
            return ep_X, ep_y

        def preprocess_for_model3(ef_ftd):
            ef_ftd["having_tc"] = (ef_ftd["Critical Temp"] >= 10).astype(int)
            ef_X = ef_ftd.iloc[:, 4:-1]
            ef_y = ef_ftd['having_tc']
            return ef_X, ef_y

        def preprocess_for_model2(ep_ftd, ef_ftd):
            ef_ftd = ef_ftd.iloc[:, 2:]
            ep_ftd = ep_ftd.iloc[:, 2:]
            
            merged_df = pd.merge(ef_ftd, ep_ftd, left_on=["Critical Temp", "_Composition"], right_on=["Critical Temp", "_Composition"], how="inner")
            merged_df["having_tc"] = (merged_df["Critical Temp"] >= 10).astype(int)

            efep_X = merged_df.iloc[:, 2:-1]
            efep_y = merged_df['having_tc']    
            return efep_X, efep_y, merged_df

        ep_X, ep_y = preprocess_for_model1(self.final_ep_ftd)
        efep_X, efep_y, _ = preprocess_for_model2(self.final_ep_ftd, self.final_ef_ftd)
        ef_X, ef_y = preprocess_for_model3(self.final_ef_ftd)
        
        return ep_X, ep_y, efep_X, efep_y, ef_X, ef_y

    def preprocess_for_regmodel(self, result_df):
        indices = result_df.index
        ef_ftd = self.ef_ftd.iloc[indices, 2:]
        ep_ftd = self.ep_ftd.iloc[indices, 2:]
        
        merged_df = pd.merge(ef_ftd, ep_ftd, left_on=["Critical Temp", "_Composition"], right_on=["Critical Temp", "_Composition"], how="inner")
    
        efep_X = merged_df.iloc[:, 2:]
        efep_y = merged_df['Critical Temp']    
        
        return efep_X, efep_y

    def ensemble_predict(self):
        ep_X, ep_y, efep_X, efep_y, ef_X, ef_y = self.preprocess_data()
        
        pred_ep = self.model_manager.predict('model_ep', ep_X)
        pred_efep = self.model_manager.predict('model_efep', efep_X)
        pred_ef = self.model_manager.predict('model_ef', ef_X)
        
        n_samples = len(pred_ep)
        ensemble_pred = np.zeros(n_samples)
        for i in range(n_samples):
            class_counts = np.bincount([pred_ep[i], pred_efep[i], pred_ef[i]])
            ensemble_pred[i] = np.argmax(class_counts)

        return ensemble_pred.astype(int), ep_y, efep_y, ef_y

    def reg_predict(self, result_df):
        efep_X, efep_y = self.preprocess_for_regmodel(result_df)
        efep_pred = self.model_manager.predict('regmodel_efep', efep_X)
        return efep_pred, efep_y

    def evaluate_ensemble(self, pred_type, y_true, y_pred):
        return self.model_manager.evaluate(pred_type, y_true, y_pred)

    def create_result_dataframe(self, ensemble_pred):
        # Identify compositions that were missing
        missing_compositions = self.final_ef_ftd["_Composition"].isnull()

        # Create result dataframe with placeholders for missing compositions
        result_df = pd.DataFrame({
            "composition": self.final_ef_ftd["_Composition"],
            "Actual Tc": np.where(missing_compositions, "Unknown", self.final_ef_ftd["Critical Temp"]),
            "prediction": ensemble_pred
        })

        result_df = result_df[result_df["prediction"] == 1]
        self.result_df = result_df.copy()
        
        # Predicting Tc for high probability superconductors
        efep_pred, efep_y = self.reg_predict(result_df)
        result_df["Predicted Tc"] = efep_pred

        return result_df



# Streamlit App
st.title("Superconductor Predictor")

# User input for CSV file upload
uploaded_file = st.file_uploader("Upload your input CSV file", type=["csv"])

if uploaded_file is not None:
    # User input for number of samples
    num_samples = st.number_input("Enter number of samples to load", min_value=1, max_value=16000, value=10000)
    
    # Show loading spinner while processing
    with st.spinner('Processing...'):
        # Initialize the components
        data_loader = DataLoader(uploaded_file, num_samples)
        feature_extractor = FeatureExtractor()
        model_manager = ModelManager()

        predictor = SuperconductorPredictor(data_loader, feature_extractor, model_manager)

        if predictor.data is not None:
            ensemble_pred, ep_y, efep_y, ef_y = predictor.ensemble_predict()
            ensemble_eval = predictor.evaluate_ensemble("C", ep_y, ensemble_pred)
            result_dataframe = predictor.create_result_dataframe(ensemble_pred)

            efep_pred, efep_y = predictor.reg_predict(result_dataframe)
            reg_eval = predictor.evaluate_ensemble("R", efep_y, efep_pred)
            result_dataframe["Predicted Tc"] = efep_pred

            # Display result dataframe
            st.subheader("Result DataFrame")
            st.write(result_dataframe)

            # Display classification results
            st.subheader("Ensemble Classification Accuracy")
            st.write(ensemble_eval)

            # Display regression results
            st.subheader("Regression Accuracy")
            st.write(reg_eval)

        else:
            st.write("Failed to load data. Please check the input file and try again.")
else:
    st.write("Please upload a CSV file to proceed.")
